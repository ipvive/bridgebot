import logging
import numpy as np
import queue
import random
import tensorflow as tf

import bridge.game as bridgegame
from bridge import tokens
import tpudata
from mubert import modeling
from mubert import mubert
from mubert import optimization
import time

import pdb


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bridgebot_config_file", "mubert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string("model_dir", None, "/path/to/networks_model_directory")

flags.DEFINE_integer("batch_size", 8, "")

flags.DEFINE_integer("num_examples", 64, "")

flags.DEFINE_integer("num_actions", 64, "")

flags.DEFINE_bool("use_coordinator", False, "")

flags.DEFINE_integer("max_seq_length", 140, "")

flags.DEFINE_integer("hidden_state_length", 128, "")


def input_fn_builder(embedding_size):
  def input_fn(params):
    input_ids = tf.zeros(
        shape=[FLAGS.max_seq_length],
        dtype=tf.int32)
    action_ids = tf.zeros(
        shape=[FLAGS.num_actions],
        dtype=tf.int32)

    d = {
        "input_ids": input_ids,
        "action_ids": action_ids,
    }
    ds = tf.data.Dataset.from_tensors(d)
    ds = ds.repeat()
    ds = ds.batch(params["batch_size"])
    ds = ds.prefetch(1)
    return ds

  def controller_input_fn(params):
    return tpudata.ControllerDataset(lambda: input_fn(params))

  if FLAGS.use_coordinator:
    return controller_input_fn
  else:
    return input_fn


def get_estimator(mubert_config):

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, coordinator_name='coordinator')

  is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.compat.v1.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(
      mubert_config=mubert_config,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=FLAGS.batch_size,
      train_batch_size=FLAGS.batch_size,
      params={"hidden_length": FLAGS.hidden_length})

  return estimator


def model_fn_builder(mubert_config, use_tpu, use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  # TODO(njt): Remove lengths from config
  mubert_config.representation.hidden_state_length = FLAGS.hidden_state_length
  mubert_config.dynamics.hidden_state_length = FLAGS.hidden_state_length
  mubert_config.prediction.hidden_state_length = FLAGS.hidden_state_length
  mubert_config.representation.max_seq_length = FLAGS.max_seq_length
  mubert_config.dynamics.max_seq_length = FLAGS.max_seq_length
  mubert_config.prediction.max_seq_length = FLAGS.max_seq_length

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT modes are supported: %s" % (mode))

    tokenizer = tokens.Tokenizer()
    for name in sorted(features.keys()):
      logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    hidden_pad = tf.constant(np.zeros(shape=(params["batch_size"], FLAGS.hidden_state_length)), dtype=tf.int32)
    input_ids = features["input_ids"]
    action_ids = features["action_ids"]

    query_tokens = ["[SCORE]", "[ACTION]"]
    query_ids_batch = tf.constant(np.repeat(
        [tokenizer.tokens_to_ids(query_tokens)], params["batch_size"], axis=0),
        dtype=tf.int32)

    dynamics_query_tokens = ["[TO_ACT]"]
    dynamics_query_ids_batch = tf.constant(np.repeat(
        [tokenizer.tokens_to_ids(dynamics_query_tokens)],
        params["batch_size"], axis=0), dtype=tf.int32)

    id_embedding_table = mubert.embedding_table(
        vocab_size=mubert_config.representation.vocab_size,
        embedding_size=mubert_config.representation.hidden_width)
    representation = mubert.MuBertRepresentationModel(
                        config=mubert_config.representation,
                        is_training=False,
                        use_one_hot_embeddings=use_one_hot_embeddings,
                        embedding_table=id_embedding_table)
    dynamics = mubert.MuBertDynamicsModel(
                        config=mubert_config.dynamics,
                        is_training=False,
                        use_one_hot_embeddings=use_one_hot_embeddings,
                        embedding_table=id_embedding_table)
    prediction = mubert.MuBertPredictionModel(
                        config=mubert_config.prediction,
                        is_training=False,
                        use_one_hot_embeddings=use_one_hot_embeddings,
                        embedding_table=id_embedding_table)

    i1 = bridgegame.first_score_id
    i2 = i1 + bridgegame.num_scores
    score_embedding_table = id_embedding_table[i1:i2, :]

    i1 = bridgegame.first_action_id
    i2 = i1 + bridgegame.num_actions
    policy_embedding_table = id_embedding_table[i1:i2, :]

    i1 = bridgegame.first_action_verb_id
    i2 = i1 + bridgegame.num_action_verbs
    next_to_act_embedding_table = id_embedding_table[i1:i2, :]

    latent_state = representation(hidden_pad, input_ids)
    for i in range(FLAGS.num_actions):
      action_id = action_ids[:, i:i+1]
      a = tf.concat([action_id, dynamics_query_ids_batch], axis=1)
      latent_state, next_to_act_seq_out = dynamics(latent_state, a)
      next_to_act_logits = mubert.get_logits(
              next_to_act_seq_out[:,1,:], next_to_act_embedding_table,
              "next_to_act")
      next_to_act = tf.math.argmax(next_to_act_logits, axis=1)
    _, sequence_output = prediction(latent_state, query_ids_batch)

    score_logits = mubert.get_logits(
        sequence_output[:,0,:], score_embedding_table, "score")
    policy_logits = mubert.get_logits(
        sequence_output[:,1,:], policy_embedding_table, "policy")

    score_values = tf.reshape(tf.range(-24.0, 25.0), [1, -1])
    value = tf.tensordot(tf.nn.softmax(score_logits),
        score_values, axes=[[1],[1]])
    policy_probs = tf.nn.softmax(policy_logits)

    predictions = {
        "next_to_act": next_to_act,
        "policy_probs": policy_probs,
        "value": value
    }

    output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions)
    return output_spec

  return model_fn


def main(_):
  mubert_config = modeling.MuBertConfig.from_json_file(FLAGS.bridgebot_config_file)
  embedding_size = mubert_config.representation.hidden_width
  estimator = get_estimator(mubert_config)
  input_fn = input_fn_builder(embedding_size)
  predictions = estimator.predict(input_fn, yield_single_examples=False)
  for _ in range(FLAGS.num_examples // FLAGS.batch_size):
    p = next(predictions)
    print("{} infer batch ready".format(time.clock_gettime(time.CLOCK_MONOTONIC)))


if __name__ == "__main__":
  tf.compat.v1.app.run()
