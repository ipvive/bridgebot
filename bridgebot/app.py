"""Application for machine learning of the game bridge."""
import os
import time

import absl.flags as flags
import absl.logging as logging
import absl.app
import tensorflow as tf

import bridge.game as bridgegame
from pb import alphabridge_pb2
import train


FLAGS = flags.FLAGS


# Booleans
flags.DEFINE_bool("debug", False, "show debug logs")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_eval_forever", False, "Whether to run eval forever on the dev set.")

flags.DEFINE_bool("learn_supervised", True, "Whether to learn from the supervised dataset.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")


# Files and paths
flags.DEFINE_string(
    "bridgebot_config_file", None,
    "The config json file. This specifies the model architecture.")

flags.DEFINE_string(
    "model_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "tfrecord_temp_dir", "/tmp", "Location where reinforcement learning tfrecord-tf.Example files are stored.")

flags.DEFINE_string(
    "supervised_dataset_location",
    "gs://njt-serene-epsilion/sl-dataset",
    "Location where supervised dataset is stored.")

flags.DEFINE_string(
    "beam_temp_location", None,
    "Location where beam temporary files are stored.")

flags.DEFINE_string(
    "beam_staging_location", None,
    "Location where beam binaries are stored.")

flags.DEFINE_string(
    "train_input_files", None,
    "Glob of files containing TFRecords of lin data for supervised training.")

flags.DEFINE_string(
    "eval_input_files", None,
    "Glob of files containing TFRecords of lin data for supervised evaluation.")


# Training metaparameters
flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")


# Other metaparameters
flags.DEFINE_integer(
    "hidden_length", 16, "Length of hidden state")

flags.DEFINE_integer(
    "num_lookahead_steps", 3,
    "Number of steps ahead to unroll the dynamics model in training.")

flags.DEFINE_integer(
    "max_lookahead_steps", 20,
    "Maximum lookahead steps in simulation.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("sample_batch_size", 1024, "Size for sampling without replacement from replay buffer.")

flags.DEFINE_float("masked_prob", 0.05, "Masked probablity for training.")


# Selfplay simulation metaparameters
flags.DEFINE_integer("simulation_batch_size", 8, "Total batch size for selfplay simulation.")

# Networking and GCE settings
flags.DEFINE_string(
    "replay_buffer_address", "localhost:10000",
    "Address of replay buffer service")

flags.DEFINE_string(
    "gcp_project", "bridgebot-264617",
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "gcp_region", "us-central1",
    "[Optional] GCE region where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_zone", "us-central1-b",
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "beam_job_name", "dataflow-test",
    "name of the Dataflow job being executed.")


# Tensorflow
flags.DEFINE_integer("reinforcement_num_batches", 10000, "Number of batches for reinforcement pipeline")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 5000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 5000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


# Beam
flags.DEFINE_string("beam_runner", "direct", "Runner to use for data preparation.")

flags.DEFINE_integer("beam_num_workers", 1, "...")

flags.DEFINE_string("beam_direct_run_mode", "in_memory", "...")

flags.DEFINE_string("setup_py_path", "./setup.py", "location of bridgebot setup.py")


def eval_forever():
    last_ckpt_line = None
    while True:
        p = os.path.join(FLAGS.model_dir, "checkpoint")
        try:
            ckpt_line = tf.io.gfile.GFile(p).readlines()[0]
            if ckpt_line != last_ckpt_line:
                last_ckpt_line = ckpt_line
                train.run_train()
        except Exception as err:
            logging.warn(err)
        time.sleep(60)
        logging.info("waiting for next checkpoint")


def main(_):
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    if FLAGS.do_eval_forever:
        FLAGS.do_eval = True
        eval_forever()
    elif FLAGS.do_train or FLAGS.do_eval:
        train.run_train()


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__) or ".")
    flags.mark_flag_as_required("bridgebot_config_file")
    flags.mark_flag_as_required("model_dir")
    absl.app.run(main)
