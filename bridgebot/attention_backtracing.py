import tensorflow as tf
import mubert.modeling as modeling

def gradient_test():
    nano_config = modeling.BertConfig(
                    vocab_size=8,
                    hidden_width=2,
                    num_hidden_layers=1,
                    num_attention_heads=1,
                    intermediate_size=6,
                    max_position_embeddings=4,
                    hidden_state_length=4)
    with  tf.name_scope("grad_test") as scope:
        input_vectors = tf.constant([[[-1.,-2.]]])
        input_ids = tf.constant([[1,2,3,3]])
        model = modeling.MuBertModel(
                       config=nano_config,
                       is_training=False,
                       input_vectors=input_vectors,
                       input_ids=input_ids,
                       scope=scope)
        all_out = model.get_sequence_output()
        return tf.gradients(all_out, [input_vectors, model.embedded_ids])

if __name__ == "__main__":
  # AY test code seems to run without this, but we probably need it
  tf.compat.v1.disable_eager_execution()
  with tf.compat.v1.Session() as sess:
    grads = gradient_test()
    sess.run(tf.compat.v1.initialize_all_variables())
    result = sess.run(grads)
    print(result)
