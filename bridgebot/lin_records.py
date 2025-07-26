import logging
import tensorflow as tf


flags = tf.compat.v1.flags


FLAGS = flags.FLAGS


flags.DEFINE_string("input_files", None,
    "Input .lin file or fileglob.")

flags.DEFINE_string("output_file", None,
    "Output tfrecord file.")


def main(_):
  output_files = FLAGS.output_file.split(",")
  writers = []
  for output_file in output_files:
    writers.append(tf.io.TFRecordWriter(output_file))
  writer_index = 0

  input_files = []
  for input_pattern in FLAGS.input_files.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))

  for input_file in input_files:
    with tf.io.gfile.GFile(input_file, "r") as reader:
      s = input_file + "\n" + reader.read()
      writers[writer_index].write(s.encode("utf-8"))
      writer_index = (writer_index + 1) % len(writers)

  for writer in writers:
    writer.close()


if __name__ == '__main__':
  flags.mark_flag_as_required("input_files")
  tf.compat.v1.app.run()
