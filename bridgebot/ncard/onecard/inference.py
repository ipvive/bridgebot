import os
import queue
import socket
import threading
import time

import absl.app
import absl.flags as flags
import absl.logging as logging
import grpc
import numpy as np

from bridge import game as bridgegame
from bridge import tokens
import pb.alphabridge_pb2 as alphabridge_pb2
import pb.alphabridge_pb2_grpc as alphabridge_pb2_grpc

class InferenceConfig:
    def __init__(self):
        self.embedding_size = self.mubert_config.representation.hidden_width
        self.max_seq_length = FLAGS.max_seq_length
        self.hidden_length = FLAGS.hidden_length
        self.latent_vectors_length = self.max_seq_length + self.hidden_length
        self.micro_batch_size = FLAGS.num_parallel_inferences
        self.estimator_config = EstimatorConfig()
        self.max_action_lookahead = FLAGS.max_lookahead_steps
        self.check_model_interval = 1800


class Inference:
    def __init__(self, config):
        self.config = config
        self.tokenizer = tokens.Tokenizer()
        model_fn = self._model_fn_builder()
        params = {"hidden_length": self.config.hidden_length}
        self._estimator = get_estimator(self.config.estimator_config,
                                       model_fn, params)
        self._next_model_update_time = None
        self._current_checkpoint = None
        self._core_batches_inflight_lock = threading.Lock()
        self._core_batches_inflight = 0

    def run_predict(self, pipe):
        while True:
            self._set_current_model()
            data_gen_builder = self._data_builder(pipe)
            input_fn = self._input_fn_builder(data_gen_builder)
            predictions_gen = self._estimator.predict(input_fn,
                    yield_single_examples=False)
            for pred in predictions_gen:
                msgs = list(self._predictions_to_protos(pred))
                for msg in msgs:
                    pipe.PutPredictionsBatch(msg)  # Don't wait_for_ready.
                with self._core_batches_inflight_lock:
                    self._core_batches_inflight -= len(msgs)
                    logging.debug("Pipeline has %d core batches inflight",
                            self._core_batches_inflight)
                    if self._core_batches_inflight == 0:
                        if self._needs_model_update():
                            logging.info("Pipeline empty. Restarting now.")
                            break

    def _input_fn_builder(self, data_gen_builder):
        def input_fn(params):
            types = {
                "unique_id": tf.uint32,
                "view_ids": tf.int32,
                "action_ids": tf.int32,
            }
            shapes = {
                "unique_id": tf.TensorShape([params["batch_size"]]),
                "view_ids": tf.TensorShape([
                    params["batch_size"],
                    self.config.max_seq_length]),
                "action_ids": tf.TensorShape([
                    params["batch_size"],
                    self.config.micro_batch_size,
                    self.config.max_action_lookahead])
            }

            def examples_fn():
                data_gen = data_gen_builder(params["batch_size"])
                return tf.data.Dataset.from_generator(
                        data_gen, output_types=types, output_shapes=shapes)

            if self.config.estimator_config.use_tpu:
                examples = tpudata.ControllerDataset(examples_fn)
            else:
                examples = examples_fn()

            examples = examples.prefetch(2)
            return examples
        return input_fn

    def _model_fn_builder(self):
        """Returns `model_fn` closure for TPUEstimator."""

        mubert_config = self.config.mubert_config
        use_tpu = self.config.estimator_config.use_tpu
        use_one_hot_embeddings = use_tpu

        #TODO: check if we can't move/reuse the model_fn to/from elsewhere
        def model_fn(features, labels, mode, params):
            """The `model_fn` for TPUEstimator."""

            if mode != tf.estimator.ModeKeys.PREDICT:
                raise ValueError("Only PREDICT modes are supported: %s" % (mode))

            for name in sorted(features.keys()):
                logging.info("  name = %s, shape = %s" %
                             (name, features[name].shape))

            view_ids = features["view_ids"]
            action_ids = features["action_ids"]

            id_embedding_table = mubert.embedding_table(
                vocab_size=mubert_config.representation.vocab_size,
                embedding_size=mubert_config.representation.hidden_width)

            score_ix1 = bridgegame.first_score_id
            score_ix2 = score_ix1 + bridgegame.num_scores
            score_embedding_table = id_embedding_table[score_ix1:score_ix2, :]

            policy_ix1 = bridgegame.first_action_id
            policy_ix2 = policy_ix1 + bridgegame.num_actions
            policy_embedding_table = id_embedding_table[policy_ix1:policy_ix2, :]

            nta_ix1 = bridgegame.first_action_verb_id
            nta_ix2 = nta_ix1 + bridgegame.num_action_verbs
            next_to_act_embedding_table = id_embedding_table[nta_ix1:nta_ix2, :]

            representation = mubert.MuBertRepresentationModel(
                config=mubert_config.representation,
                is_training=False,
                use_one_hot_embeddings=use_one_hot_embeddings,
                embedding_table=id_embedding_table)

            hidden_pad_id = self.tokenizer.tokens_to_ids(["[HIDDEN]"])[0]
            hidden_pad_ids_batch = tf.fill(
                dims=[params["batch_size"], params["hidden_length"]],
                value=self.tokenizer.tokens_to_ids(["[HIDDEN]"])[0])
            latent_vectors = representation(hidden_pad_ids_batch, view_ids)

            # replicate latent vectors, and merge *batch dimensions
            assert action_ids.shape[1] == self.config.micro_batch_size
            latent_vectors = tf.repeat(latent_vectors, self.config.micro_batch_size,
                    axis=0)
            action_ids = tf.reshape(action_ids,
                    shape=[params["batch_size"] * self.config.micro_batch_size,
                        self.config.max_action_lookahead])

            dynamics = mubert.MuBertDynamicsModel(
                config=mubert_config.dynamics,
                is_training=False,
                use_one_hot_embeddings=use_one_hot_embeddings,
                embedding_table=id_embedding_table)
            to_act_id = self.tokenizer.tokens_to_ids(["[TO_ACT]"])[0]
            to_act_ids_batch = tf.fill(
                dims=[params["batch_size"] * self.config.micro_batch_size, 1],
                value=to_act_id)

            def any_dynamics_update_needed(i, l, n):
                logging.warning(f"any_update: i={i} action_ids[:,i]={action_ids[:, i]}")
                return tf.reduce_any(
                    tf.math.not_equal(action_ids[:, i], tf.constant(0)))

            def dynamics_update(i, l, n):
                dyn_seq_in = tf.concat(
                        [action_ids[:, i:i+1], to_act_ids_batch], axis=1)
                dyn_l, dyn_seq_out = dynamics(l, dyn_seq_in)
                dyn_l = tf.reshape(dyn_l, shape=l.shape)
                dyn_n = dyn_seq_out[:, 1, :]
                dyn_n = tf.reshape(dyn_n, shape=n.shape)
                logging.info(f"update: dyn_l.shape={dyn_l.shape} dyn_n.shape={dyn_n.shape}")
                needs_update = tf.math.not_equal(
                        action_ids[:, i], tf.constant(0))
                needs_update = tf.expand_dims(needs_update, -1)
                logging.info(f"update: needs_update.shape={needs_update.shape}")
                logging.info(f"update: action_ids.shape={action_ids.shape}")
                next_l = tf.where(tf.expand_dims(needs_update, -1), dyn_l, l)
                next_n = tf.where(needs_update, dyn_n, n)
                logging.info(f"update: i={i}")
                logging.info(f"update: l.shape={l.shape} next_l.shape={next_l.shape}")
                logging.info(f"update: n.shape={n.shape} next_n.shape={next_n.shape}")
                return tf.add(i, 1), next_l, next_n

            next_to_act_out = tf.zeros(
                shape=[params["batch_size"] * self.config.micro_batch_size,
                    latent_vectors.shape[-1]],
                dtype=tf.float32)
            i = tf.constant(0)
            _, latent_vectors, next_to_act_out = tf.while_loop(
                cond=any_dynamics_update_needed,
                body=dynamics_update,
                loop_vars=[i, latent_vectors, next_to_act_out],
                maximum_iterations=action_ids.shape[1] - 1)
            next_to_act_logits = mubert.get_logits(
                next_to_act_out, next_to_act_embedding_table, "next_to_act_logits")
            next_to_act = tf.math.argmax(next_to_act_logits, axis=1)

            prediction = mubert.MuBertPredictionModel(
                config=mubert_config.prediction,
                is_training=False,
                use_one_hot_embeddings=use_one_hot_embeddings,
                embedding_table=id_embedding_table)
            prediction_query_ids = self.tokenizer.tokens_to_ids(["[SCORE]", "[ACTION]"])
            prediction_query_ids_batch = tf.tile(
                tf.constant([prediction_query_ids]),
                tf.constant([params["batch_size"] * self.config.micro_batch_size, 1]))
            _, prediction_sequence_output = prediction(
                latent_vectors, prediction_query_ids_batch)

            score_logits = mubert.get_logits(
                prediction_sequence_output[:, 0, :], score_embedding_table,
                "score_logits")
            policy_logits = mubert.get_logits(
                prediction_sequence_output[:, 1, :], policy_embedding_table,
                "policy_logits")

            score_probs = tf.nn.softmax(score_logits, axis=-1)
            score_values = tf.reshape(tf.range(-24.0, 25.0), [-1, 1])
            value = tf.tensordot(score_probs, score_values, axes=1)
            value = tf.reshape(value, [-1])

            policy_probs = tf.nn.softmax(policy_logits)

            # reshape outputs to split batch dimensions
            next_to_act = tf.reshape(next_to_act,
                    shape=(-1, self.config.micro_batch_size))
            value = tf.reshape(value, (-1, self.config.micro_batch_size))
            policy_probs = tf.reshape(policy_probs,
                    shape=(-1, self.config.micro_batch_size, policy_probs.shape[-1]))

            tvars = tf.compat.v1.trainable_variables()
            initialized_variable_names = {}

            tf.compat.v1.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name,
                                          var.shape, init_string)

            predictions = {
                "next_to_act": next_to_act,
                "policy": policy_probs,
                "value": value,
                "unique_id": features["unique_id"],
            }

            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions)
            return output_spec

        return model_fn

    def _needs_model_update(self):
        if not self._next_model_check_time:
            return True
        t = time.time()
        if t < self._next_model_check_time:
            return False
        checkpoint = self._estimator.latest_checkpoint()
        if checkpoint != self._current_checkpoint:
            self._next_model_check_time = None
            return True
        else:
            self._next_model_check_time = t + self.config.check_model_interval
            return False

    def _set_current_model(self):
        t = time.time()
        self._next_model_check_time = t + self.config.check_model_interval
        self._current_checkpoint = self._estimator.latest_checkpoint()

    def _data_builder(self, pipe):
        def gen_builder(batch_size):
            req = alphabridge_pb2.FeaturesBatchRequest(batch_size=batch_size)
            def gen():
                    while True:
                        with self._core_batches_inflight_lock:
                            core_batches = self._core_batches_inflight
                        if core_batches % 8 == 0 and self._needs_model_update():
                            time.sleep(30)
                        logging.debug("Asking pipe for features batch")
                        yield self._proto_to_features(
                                pipe.GetFeaturesBatch(req, wait_for_ready=True),
                                batch_size)
                        with self._core_batches_inflight_lock:
                            self._core_batches_inflight += 1
            return gen
        return gen_builder

    def _proto_to_features(self, features_batch_msg, batch_size):
        logging.debug("Proto to features")
        fb = features_batch_msg
        view_ids = [self.tokenizer.tokens_to_ids(fmb.view_token)
                for fmb in fb.micro]
        action_offset = bridgegame.first_action_id 
        action_ids = [
                [
                    [action_offset + idx for idx in aip.action_index]
                    for aip in fmb.action_path
                ] for fmb in fb.micro
        ]
        return {
                "unique_id": np.repeat(fb.unique_id, batch_size),
                "view_ids": self._view_ids_padded_array(view_ids, batch_size),
                "action_ids": self._action_ids_padded_array(
                    action_ids, batch_size),
        }

    def _action_ids_padded_array(self, action_ids, batch_size):
        return np.array(_nested_pad(action_ids,
            shape=[
                batch_size,
                self.config.micro_batch_size,
                self.config.max_action_lookahead,
            ],
            fill=0))

    def _predictions_to_protos(self, predictions_dict):
        next_to_act = predictions_dict["next_to_act"]
        value = predictions_dict["value"]
        policy = predictions_dict["policy"]
        batch_size, micro_batch_size = value.shape

        micro = [
                alphabridge_pb2.PredictionsMicroBatch(
                    prediction=[
                        alphabridge_pb2.Prediction(
                            next_to_act=next_to_act[i][j],
                            value=value[i][j],
                            policy=policy[i][j][:]
                        ) for j in range(micro_batch_size)
                    ]
                ) for i in range(batch_size)
        ]
        logging.debug(f"unique_id = {predictions_dict['unique_id']}")
        uids = predictions_dict['unique_id']
        for uid in np.unique(uids):
            sel = (uids == uid)
            uid_micro = [m for i, m in enumerate(micro) if sel[i]]
            yield alphabridge_pb2.PredictionsBatch(
                unique_id=uid,
                micro=uid_micro
        )


FLAGS = flags.FLAGS


# Files and paths
flags.DEFINE_string(
    "bridgebot_config_file", None,
    "The config json file. This specifies the model architecture.")

flags.DEFINE_string(
    "model_dir", None,
    "The output directory where the model checkpoints will be written.")

# Training metaparameters
flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

# Other metaparameters
flags.DEFINE_integer(
    "hidden_length", 16, "Length of hidden state")

flags.DEFINE_integer(
    "num_parallel_inferences", 1, "Number of parallel inferences.")

flags.DEFINE_integer(
    "max_lookahead_steps", 20,
    "Maximum lookahead steps in simulation.")
