# TODO(njt): consolidate copied code from various model_fn in `../*py` here.
from __future__ import absolute_import
import tensorflow as tf
import sys
import mubert.modeling as modeling


def embedding_table(vocab_size,
                    embedding_size=128,
                    initializer_range=0.02,
                    word_embedding_name="word_embeddings"):
    with tf.compat.v1.variable_scope("mubert"):
      with tf.compat.v1.variable_scope("embeddings"):
        embedding_table = tf.compat.v1.get_variable(
            name=word_embedding_name,
            shape=[vocab_size, embedding_size],
            initializer=modeling.create_initializer(initializer_range))
    return embedding_table


class MuBertRepresentationModel(object):
    def __init__(self, config, is_training, use_one_hot_embeddings=False,
        embedding_table=None):
        self.config = config
        self.is_training = is_training
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.scope = "representation"
        self.embedding_table = embedding_table

    def __call__(self, hidden_pad, input_ids):
        """Returns hidden_state, embedding_out, masked_sequence_out."""
        mu_model = modeling.MuBertModel(
               config=self.config,
               is_training=self.is_training,
               input_vectors=None,
               input_ids=tf.concat([hidden_pad, input_ids], axis=1),
               use_one_hot_embeddings=self.use_one_hot_embeddings,
               embedding_table=self.embedding_table,
               scope=self.scope)

        return mu_model.get_sequence_output()


class MuBertPredictionModel(object):
    def __init__(self, config, is_training, use_one_hot_embeddings=False,
            embedding_table=None):
        self.config = config
        self.is_training = is_training
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.scope = "prediction"
        self.embedding_table = embedding_table

    def __call__(self, input_vectors, query_ids):
        """Returns seq_out after dropping hidden_state."""
        mu_model = modeling.MuBertModel(
               config=self.config,
               is_training=self.is_training,
               input_vectors=input_vectors,
               input_ids=query_ids,
               use_one_hot_embeddings=self.use_one_hot_embeddings,
               embedding_table=self.embedding_table,
               scope=self.scope)

        latent_output_end = input_vectors.shape[1]
        full_seq_output = mu_model.get_sequence_output()
        latent_output = full_seq_output[:, :latent_output_end, :]
        prediction_output = full_seq_output[:, latent_output_end:, :]

        return latent_output, prediction_output


class MuBertDynamicsModel(object):
    def __init__(self, config, is_training, use_one_hot_embeddings=False,
            embedding_table=None):
        self.config = config
        self.is_training = is_training
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.scope = "dynamics"
        self.embedding_table = embedding_table

    def __call__(self, input_vectors, action_query_ids):
        """Returns hidden_state, seq_out."""
        mu_model = modeling.MuBertModel(
               config=self.config,
               is_training=self.is_training,
               input_vectors=input_vectors,
               input_ids=action_query_ids,
               use_one_hot_embeddings=self.use_one_hot_embeddings,
               embedding_table=self.embedding_table,
               scope=self.scope)

        output_vectors_length = input_vectors.shape[1]
        full_seq_output = mu_model.get_sequence_output()
        output_vectors = full_seq_output[:, :output_vectors_length, :]
        response = full_seq_output[:, output_vectors_length:, :]

        return output_vectors, response


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch.
     Return a tensor of shape [batch_size*len(positions), width], where
     width is usually config.hidden_width in the bert implementation."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                     [batch_size * seq_length, width])

    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor

def get_masked_loss(bert_config, input_tensor, embedding_table, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)
  # The input tensor now has shape [batch_size*len(positions), width]
  with tf.compat.v1.variable_scope("masked_predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.compat.v1.variable_scope("transform"):
      input_tensor = tf.compat.v1.layers.dense(
          input_tensor,
          units=bert_config.hidden_width,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # We use the embedding table to get tensor of shape [batch_size*len(positions), vocab_size]
    logits = tf.matmul(input_tensor, embedding_table, transpose_b=True)
    # We add an output-only bias for each token.
    output_bias = tf.compat.v1.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

    return (loss, per_example_loss, log_probs)

def get_logits(input_tensor, special_embedding_table, scope):
    """Return a tensor of log_probs using the given embedding table"""
    # The input tensor has shape [batch_size, width]
    logits = tf.matmul(input_tensor, special_embedding_table, transpose_b=True)
    # We add an output-only bias for each token.
    special_vocab_size = modeling.get_shape_list(logits, expected_rank=2)[1]
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        output_bias = tf.compat.v1.get_variable(
            "bias",
            shape=[special_vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.nn.bias_add(logits, output_bias)
    return logits


def get_log_probs(input_tensor, special_embedding_table, scope):
    logits = get_logits(input_tensor, special_embedding_table, scope)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    return log_probs


def logistic_loss(logits, label_indices, depth):
    labels = tf.one_hot(label_indices, depth=depth, dtype=tf.float32)
    return tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
