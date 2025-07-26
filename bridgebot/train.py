# coding=utf-8
# Copyright 2020 ipvive.
# Portions Copyright 2018 The Google AI Language Team Authors.
"""Train BERT-inspired model for bridge."""
import apache_beam as beam
import grpc
import absl.logging as logging
import numpy as np
import os
import random
import socket
import tensorflow as tf
import tensorflow_datasets as tfds

import mubert.modeling as modeling
import mubert.optimization as optimization
import mubert.mubert as mubert
import pb.alphabridge_pb2 as alphabridge_pb2
import pb.alphabridge_pb2_grpc as alphabridge_pb2_grpc
import bridge.game as bridgegame
import bridge.tokens as tokens
import training_pipeline

import pdb


flags = tf.compat.v1.flags

FLAGS = flags.FLAGS


def features_to_predictions(features, params, mubert_config,
    use_one_hot_embeddings, is_training):
  """ Processes input features and returns prediction tensors."""
  tokenizer = tokens.Tokenizer()
  for name in sorted(features.keys()):
    logging.info("  name = %s, shape = %s" % (name, features[name].shape))

  input_ids = features["input_ids"]
  actions = features["actions"]

  id_embedding_table = mubert.embedding_table(
      vocab_size=mubert_config.representation.vocab_size,
      embedding_size=mubert_config.representation.hidden_width)

  representation = mubert.MuBertRepresentationModel(
                      config=mubert_config.representation,
                      is_training=is_training,
                      use_one_hot_embeddings=use_one_hot_embeddings,
                      embedding_table=id_embedding_table)

  hidden_pad = ["[HIDDEN]"] * params["hidden_length"]
  hidden_pad_batch = tf.constant(np.repeat(
      [tokenizer.tokens_to_ids(hidden_pad)], params["batch_size"], axis=0),
      dtype=tf.int32)

  latent_vectors  = representation(hidden_pad_batch, input_ids)

  prediction = mubert.MuBertPredictionModel(
                      config=mubert_config.prediction,
                      is_training=is_training,
                      use_one_hot_embeddings=use_one_hot_embeddings,
                      embedding_table=id_embedding_table)
  dynamics = mubert.MuBertDynamicsModel(
                      config=mubert_config.dynamics,
                      is_training=is_training,
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

  query_tokens = ["[SCORE]", "[ACTION]"]
  query_ids_batch = tf.constant(np.repeat(
      [tokenizer.tokens_to_ids(query_tokens)], params["batch_size"], axis=0),
      dtype=tf.int32)

  masked_output, prediction_output = prediction(latent_vectors, query_ids_batch)

  score_logits = mubert.get_logits(
      prediction_output[:,0,:], score_embedding_table, "score_logits")
  policy_logits = mubert.get_logits(
      prediction_output[:,1,:], policy_embedding_table, "policy_logits")

  dynamics_query_tokens = ["[TO_ACT]"]
  dynamics_query_ids_batch = tf.constant(np.repeat(
      [tokenizer.tokens_to_ids(dynamics_query_tokens)],
      params["batch_size"], axis=0), dtype=tf.int32)

  # Recurrent steps, from action and previous hidden state.
  num_lookahead_steps = actions.shape.as_list()[1] - 1

  predictions = [(1.0, None, score_logits, policy_logits)]
  for i in range(num_lookahead_steps):
    a = tf.concat([actions[:,i:i+1], dynamics_query_ids_batch], axis=1)

    latent_vectors, next_to_act_seq_out = dynamics(latent_vectors, a)

    next_to_act_logits = mubert.get_logits(
            next_to_act_seq_out[:,1,:], next_to_act_embedding_table,
            "next_to_act_logits")

    latent_output, prediction_output = prediction(latent_vectors, query_ids_batch)
    masked_output = latent_output[:, params["hidden_length"]:, :]

    score_logits = mubert.get_logits(
        prediction_output[:,0,:], score_embedding_table, "score_logits")
    policy_logits = mubert.get_logits(
        prediction_output[:,1,:], policy_embedding_table, "policy_logits")

    predictions.append((
      1.0 / num_lookahead_steps, next_to_act_logits,
      score_logits, policy_logits))
    # TODO: Experiment with better gradient_scale weights or
    # explain why we definitely do not need them here
    # latent_vectors = scale_gradient(latent_vectors, 0.5)

  return (predictions, masked_output, id_embedding_table)


def target_losses(features, predictions):
  """ Returns target_loss and eval losses for training model."""
  target_actors = features["target_actors"] - bridgegame.first_action_verb_id
  target_score = features["target_score"] - bridgegame.first_score_id
  # target policies are probs over the list of actions
  # one-hot for supervised and mcts probs for reinfurcement
  target_policies = features["target_policies"]

  target_loss = tf.constant(0.0, dtype=tf.float32)
  losses = {}
  for idx, prediction in enumerate(predictions):
    gradient_scale, next_to_act_logits, score_logits, policy_logits = prediction
    score_loss = mubert.logistic_loss(
                    score_logits,
                    target_score,
                    bridgegame.num_scores)
    policy_loss = tf.nn.softmax_cross_entropy_with_logits(
                    logits=policy_logits, labels=target_policies[:,idx,:])

    next_to_act_loss = tf.constant(0.0, dtype=tf.float32)
    if next_to_act_logits is not None:
      next_to_act_loss = mubert.logistic_loss(
                    next_to_act_logits,
                    target_actors[:, idx],
                    bridgegame.num_action_verbs)

    score_loss = scale_gradient(score_loss, gradient_scale)
    policy_loss = scale_gradient(policy_loss, gradient_scale)
    next_to_act_loss = scale_gradient(next_to_act_loss, gradient_scale)

    losses["score_loss_{}".format(idx)] = tf.compat.v1.metrics.mean(score_loss)
    losses["policy_loss_{}".format(idx)] = tf.compat.v1.metrics.mean(policy_loss)
    if next_to_act_logits is not None:
      losses["next_to_act_loss_{}".format(idx)] = tf.compat.v1.metrics.mean(next_to_act_loss)

    target_loss += tf.nn.compute_average_loss(
        score_loss + policy_loss + next_to_act_loss)

  return target_loss, losses


def eval_accuracies(features, predictions):
  """ Returns target accuracies for eval."""
  target_actors = features["target_actors"] - bridgegame.first_action_verb_id
  target_score = features["target_score"] - bridgegame.first_score_id
  # target policies are probs over the list of actions
  # one-hot for supervised and mcts probs for reinfurcement
  target_policies = features["target_policies"]

  accuracies = {}
  for idx, prediction in enumerate(predictions):
    gradient_scale, next_to_act_logits, score_logits, policy_logits = prediction

    accuracies["score_accuracy_{}".format(idx)] = tf.compat.v1.metrics.mean(
       tf.metrics.sparse_categorical_accuracy(target_score, score_logits))

    target_policy = tf.argmax(target_policies[:,idx,:],
                              axis=-1, output_type=tf.int32)

    accuracies["policy_accuracy_{}".format(idx)] = tf.compat.v1.metrics.mean(
       tf.metrics.sparse_categorical_accuracy(target_policy, policy_logits))

    if next_to_act_logits is not None:
      accuracies["next_to_act_accuracy_{}".format(idx)] =\
        tf.compat.v1.metrics.mean(
          tf.metrics.sparse_categorical_accuracy(
            target_actors[:, idx], next_to_act_logits))

  return accuracies


def init_from_checkpoint(init_checkpoint, use_tpu):
  """ Initialize variables from checkpount scaffold_fn."""
  tvars = tf.compat.v1.trainable_variables()
  initialized_variable_names = {}
  scaffold_fn = None
  if init_checkpoint:
    (assignment_map, initialized_variable_names
    ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    if use_tpu:
      def tpu_scaffold():
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.compat.v1.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

  return scaffold_fn


def model_fn_builder(mubert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    is_eval = (mode == tf.estimator.ModeKeys.EVAL)

    predictions, masked_output, id_embedding_table =\
        features_to_predictions(features, params, mubert_config,
            use_one_hot_embeddings, is_training)

    masked_positions = features["masked_positions"]
    masked_ids = features["masked_ids"]
    masked_weights = features["masked_weights"]

    (masked_loss, masked_example_loss,
         masked_log_probs) = mubert.get_masked_loss(
         mubert_config.representation, masked_output, id_embedding_table,
         masked_positions, masked_ids, masked_weights)

    target_loss, losses = target_losses(features, predictions)
    total_loss = target_loss + masked_loss / FLAGS.max_predictions_per_seq

    scaffold_fn = init_from_checkpoint(init_checkpoint, use_tpu)
    accuracies = eval_accuracies(features, predictions)
    accuracies["masked_accuracy"] = tf.compat.v1.metrics.mean(
        tf.metrics.sparse_categorical_accuracy(
            tf.reshape(masked_ids,[-1]), masked_log_probs))

    losses["masked_mean_loss"] = tf.compat.v1.metrics.mean(
       values=masked_example_loss, weights=tf.reshape(masked_weights, [-1]))

    def metric_fn(losses, accuracies):
      """Returns the loss and accuracy of the model."""
      # NOTE: this won't run on TPU, (pair of dict args not supported.
      result = {}
      result.update(losses)
      result.update(accuracies)
      return result

    eval_metrics = (metric_fn, [losses, accuracies])

    train_op = None
    if is_training:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

    output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
        eval_metrics=eval_metrics,
        scaffold_fn=scaffold_fn)

    return output_spec

  return model_fn


def scale_gradient(tensor, scale):
  """Scales the gradient for the backward pass."""
  return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def input_fn_builder(max_seq_length,
                     max_predictions_per_seq,
                     num_lookahead_steps,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    beam_options = beam.options.pipeline_options.PipelineOptions(
        [
          "--runner={}".format(FLAGS.beam_runner),
          "--job_name={}".format(FLAGS.beam_job_name),
          "--project={}".format(FLAGS.gcp_project),
          "--zone={}".format(FLAGS.gcp_zone),
          "--network=default",
          "--region={}".format(FLAGS.gcp_region),
          "--network=default",
          "--temp_location={}".format(FLAGS.beam_temp_location),
          "--staging_location={}".format(FLAGS.beam_staging_location),
          "--direct_num_workers={}".format(FLAGS.beam_num_workers),
          "--direct_running_mode={}".format(FLAGS.beam_direct_run_mode),
          "--num_workers={}".format(FLAGS.beam_num_workers),
          "--setup_file={}".format(FLAGS.setup_py_path),
        ])
    if FLAGS.learn_supervised:
      config = training_pipeline.SupervisedConfig(name='bridgebot-sl',
          train_input_files=FLAGS.train_input_files,
          eval_input_files=FLAGS.eval_input_files,
          max_seq_length=FLAGS.max_seq_length,
          max_predictions_per_seq=FLAGS.max_predictions_per_seq,
          num_lookahead_steps=FLAGS.num_lookahead_steps,
          num_actions=bridgegame.num_actions,
          masked_prob=FLAGS.masked_prob)
      data_dir = FLAGS.supervised_dataset_location
      if not data_dir:
          data_dir = FLAGS.beam_temp_location
      builder = training_pipeline.SupervisedBuilder(
          data_dir=data_dir, config=config)
      download_config = tfds.download.DownloadConfig(beam_options=beam_options)
      builder.download_and_prepare(download_config=download_config)
      d = builder.as_dataset()
      if is_training:
        d = d["train"]
        d = d.shuffle(10000, reshuffle_each_iteration=True)
      else:
        d = d["test"]
      d = d.batch(params["batch_size"], drop_remainder=True)
      d = d.prefetch(1)
      return d
    else:
      config = training_pipeline.ReinforcementConfig(
          replay_buffer_address=FLAGS.replay_buffer_address,
          tfrecord_temp_dir=FLAGS.tfrecord_temp_dir,
          sample_batch_size=FLAGS.sample_batch_size,
          num_batches=FLAGS.reinforcement_num_batches,
          beam_options=beam_options,
          use_tpu=FLAGS.use_tpu,
          max_seq_length=FLAGS.max_seq_length,
          max_predictions_per_seq=FLAGS.max_predictions_per_seq,
          num_lookahead_steps=FLAGS.num_lookahead_steps,
          num_actions=bridgegame.num_actions,
          masked_prob=FLAGS.masked_prob,
          num_shards=FLAGS.beam_num_workers)
      builder = training_pipeline.ReinforcementBuilder(config)
      d = builder.as_dataset()
      d = d.batch(params["batch_size"], drop_remainder=True)
      d = d.prefetch(2)
      return d

  return input_fn


def start_local_tf_server():
    hostname = socket.gethostname()
    hostaddr = socket.gethostbyname(hostname)
    svr = tf.distribute.Server({'local': ['%s:0' % hostaddr]},
                                      protocol='grpc',
                                      config=None,
                                      start=True)
    splits = svr.target.split(':')
    svr_port = splits[2]
    svr_address = '%s:%s' % (hostaddr, svr_port)
    logging.info("listening on %s (%s)" % (svr_address, svr.target))
    return svr_address


def run_train():
  if FLAGS.debug:
    logging.getLogger().setLevel(logging.DEBUG)
  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  mubert_config = modeling.MuBertConfig.from_json_file(FLAGS.bridgebot_config_file)

  tf.compat.v1.gfile.MakeDirs(FLAGS.model_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_name = FLAGS.tpu_name
    if len(tpu_name.split("/")) > 1:
      tpu_name = tpu_name.split("/")[-1]
    local_server_address = start_local_tf_server()
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu_name, zone=FLAGS.gcp_zone, project=FLAGS.gcp_project,
        coordinator_name="coordinator",
        coordinator_address=local_server_address)

  is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.compat.v1.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(
      mubert_config=mubert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      params={"hidden_length": FLAGS.hidden_length})

  if FLAGS.do_train:
    logging.info("***** Running training *****")
    logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        num_lookahead_steps=FLAGS.num_lookahead_steps,
        is_training=True)

  if FLAGS.do_eval:
    logging.info("***** Running evaluation *****")
    logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    eval_input_fn = input_fn_builder(
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        num_lookahead_steps=FLAGS.num_lookahead_steps,
        is_training=False)

  if FLAGS.do_train and FLAGS.do_eval:
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=FLAGS.max_eval_steps,
            throttle_secs=10, start_delay_secs=10)
    logging.info("EvalSpec: %s", eval_spec)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  elif FLAGS.do_train:
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)
  elif FLAGS.do_eval:
    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)
    output_eval_file = os.path.join(FLAGS.model_dir, "eval_results.txt")
    with tf.io.gfile.GFile(output_eval_file, "w") as writer:
      logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
