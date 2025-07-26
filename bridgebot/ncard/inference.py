import pathlib
import socket
import threading
import time

import absl.app
import absl.flags as flags
import absl.logging as logging
import grpc
import tensorflow as tf
import numpy as np
import jaxline
import jax

from bridgebot.ncard import game as bridgegame
from bridgebot.ncard import model
from bridgebot.ncard import experiment
from bridgebot.ncard import chords
import bridgebot.pb.alphabridge_pb2 as alphabridge_pb2
import bridgebot.pb.alphabridge_pb2_grpc as alphabridge_pb2_grpc

import pdb


def _pad(v, n, fill):
    if len(v) >= n:
        return v[:n]
    else:
        return v + [fill] * (n - len(v))

def _nested_pad(v, shape, fill):
    if len(shape) == 1:
        return _pad(v, shape[0], fill)
    n = shape[0]
    if len(v) >= n:
        return [_nested_pad(vv, shape[1:], fill) for vv in v[:n]]
    else:
        return ([_nested_pad(vv, shape[1:], fill) for vv in v] +
                [_nested_pad([], shape[1:], fill) for _ in range(n - len(v))])


class Inference:
    def __init__(self, config, game):
        self.game = game
        self.tokenizer = bridgegame.Tokenizer(game)
        self.config = config
        self._next_model_update_time = None
        self._current_checkpoint = None
        self.e = experiment.Experiment(
                mode='infer', init_rng=jax.random.PRNGKey(config.random_seed),
                config=config.experiment_kwargs.config)
        self.rng = jax.random.PRNGKey(config.random_seed)

        checkpointer = experiment.NaiveDiskCheckpointer(config, None)
        state = checkpointer.get_experiment_state("latest")
        state.experiment_module = self.e
        experiment.checkpoint_dir = "/home/njt/bridgebot/data/c0f"
        checkpointer.restore("latest")
        self.e._initialize_train()

    def predictions_gen(self, pipe):
        # TODO: prefetch to keep the GPU busy.
        while True:
            batch_size = 10  # TODO: make this a flag
            req = alphabridge_pb2.FeaturesBatchRequest(batch_size=batch_size)
            batch_msg = pipe.GetFeaturesBatch(req, wait_for_ready=True)
            print(batch_msg)
            pdb.set_trace()  # TODO: remove after fixing bugs for batch > 1.
            features = self._proto_to_features(batch_msg.micro], batch_msg.unique_id)
            predictions = self._features_to_predictions(features)
            for proto in self._predictions_to_protos(predictions):
                yield proto

    def _proto_to_features(self, msg, unique_id):
        max_legal_actions = 35 + 2
        max_seq_length = 256
        max_chord_width = 5
        target_mask = np.array(
            [1.] * (len(msg.queries)) + \
            [0.] * (max_legal_actions - len(msg.queries)))
        other_tokens = [["value_gt"], ["value_geq"], ["outcome"]]
        other_tokens = self.tokenizer.tokens_to_ids(other_tokens)
        features = {
            "unique_id": unique_id,
            "input_view_ids": _nested_pad([list(c.micro_token_id) for c in msg.view_chords],
                (max_seq_length, max_chord_width), 0),
            "input_par_outcome": _pad(list(msg.par_outcome.micro_token_id),
                                      max_chord_width, 0),
            "query_ids": _nested_pad(other_tokens + \
                    [list(c.micro_token_id) for c in msg.queries],
                                     (3 + max_legal_actions, max_chord_width), 0),
            "target_mask": target_mask,
        }
        features = {k: np.array([v]) for k, v in features.items()}
        return features

    def _features_to_predictions(self, features):
        (logits, t), state = self.e.forward.apply(
                self.e._params, self.e._state, self.rng, features, False)
        value_gt = logits[:,0,2] - logits[:,0,3]
        value_geq = logits[:,1,2] - logits[:,1,3]
        outcome = [chords.generate(l[2], self.e.codecs["outcome"])[0]
                   for l in logits]
        policy_ll = -chords.batch_yes_log_likelihood(
            logits[:,3:,:], YES=2, NO=3)
        policy = jax.nn.softmax(policy_ll, where=features["target_mask"], axis=1)
        return {
                'unique_id': features['unique_id'],
                'value_gt': value_gt,
                'value_geq': value_geq,
                'outcome': outcome,
                'policy': policy,
                }

    def run_predict(self, pipe):
        while True:
            self._set_current_model()
            for msg in self.predictions_gen(pipe):
                pipe.PutPredictionsBatch(msg)  # Don't wait_for_ready.

    def _last_checkpoint(self):
        checkpoint_dir = pathlib.Path(
                FLAGS.base_model_dir, FLAGS.config.checkpoint_dir)
        checkpoints_gen = checkpoint_dir.glob('latest-*')
        checkpoints = [x for x in checkpoints_gen]
        if not checkpoints:
            logging.error("model_dir is empty")
        checkpoint = sorted(checkpoints)[-1]
        return checkpoint

    def _needs_model_update(self):
        if not self._next_model_check_time:
            return True
        t = time.time()
        if t < self._next_model_check_time:
            return False
        checkpoint = self.last_checkpoint()
        if checkpoint != self._current_checkpoint:
            self._next_model_check_time = None
            return True
        else:
            self._next_model_check_time = t + self.config.check_model_interval
            return False

    def _set_current_model(self):
        t = time.time()
        self._current_checkpoint = self._last_checkpoint()
        self.experiment = experiment.Experiment(
                mode='infer', init_rng=None, config=self.config)

    def _predictions_to_protos(self, predictions_dict):
        """package predictions dict into series of PredictionsBatches."""
        value_gt = predictions_dict["value_gt"]
        value_geq = predictions_dict["value_geq"]
        outcome = predictions_dict["outcome"]
        policy = predictions_dict["policy"]
        batch_size, micro_batch_size = value_gt.shape[0], 1
        print(f"batch_size={batch_size} mbs={micro_batch_size}")

        micro = [
                alphabridge_pb2.PredictionsMicroBatch(
                    prediction=[
                        alphabridge_pb2.Prediction(
                            value_gt=value_gt[i],
                            value_geq=value_geq[i],
                            par_outcome={'micro_token_id': outcome[0]},
                            policy=policy[i][:]
                        ) for j in range(micro_batch_size)
                    ]
                ) for i in range(batch_size)
        ]
        logging.error(f"unique_id = {predictions_dict['unique_id']}")
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
    "base_model_dir", None,
    "The base directory where the model checkpoints will be read from.")

# Training metaparameters
flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

# Other metaparameters
flags.DEFINE_integer(
        "embedding_width", 24, "number of dimensions per token")

flags.DEFINE_integer(
    "hidden_length", 16, "Length of hidden state")

flags.DEFINE_integer(
    "num_parallel_inferences", 1, "Number of parallel inferences.")

flags.DEFINE_integer(
    "max_lookahead_steps", 20,
    "Maximum lookahead steps in simulation.")

# Selfplay simulation metaparameters
flags.DEFINE_integer("simulation_batch_size", 8, "Total batch size for selfplay simulation.")

# Networking and GCE settings
flags.DEFINE_string("inference_pipe_address", "localhost:20000",
        "Address of inference pipe service")

def main(_):
    game = bridgegame.Game(n=2)
    config = jaxline.platform._CONFIG.value
    jaxline.base_config.validate_config(config)
    inference = Inference(config, game)

    # TODO: remove after https://github.com/grpc/grpc/issues/24018 is resolved.
    # os.environ["GRPC_DNS_RESOLVER"] = "native"
    # os.environ["GRPC_VERBOSITY"] = "debug"

    logging.info("connecting to inference pipe at %s", FLAGS.inference_pipe_address)
    pipe_channel = grpc.insecure_channel(FLAGS.inference_pipe_address)
    pipe = alphabridge_pb2_grpc.InferencePipeStub(pipe_channel)

    # wait for the inference pipe to be ready.
    batch = pipe.GetFeaturesBatch(
            alphabridge_pb2.FeaturesBatchRequest(batch_size=0),
            wait_for_ready=True)

    logging.info("inference pipe ready: %s", batch)
    inference.run_predict(pipe)

if __name__ == "__main__":
  logging.set_verbosity(logging.DEBUG)
  flags.mark_flag_as_required("config")
  flags.mark_flag_as_required("base_model_dir")

  absl.app.run(main)
