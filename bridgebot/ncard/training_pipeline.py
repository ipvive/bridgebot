from absl import app
import apache_beam as beam
import grpc
import io
import math
import numpy as np
import os.path
import random
import tensorflow as tf
import tensorflow_datasets as tfds
import re

from bridgebot.pb import alphabridge_pb2
from bridgebot.pb import alphabridge_pb2_grpc
from bridgebot.ncard import model
import bridgebot.ncard.game as bridgegame

import pdb

def _pad(v, n, fill):
    if len(v) >= n:
        return v[:n]
    else:
        return v + [fill] * (n - len(v))


def _position_to_features(position, game_obj, max_seq_length, max_chord_width,
                          max_legal_actions):
    tokenizer = bridgegame.Tokenizer(game_obj)
    played_game = position.board.tables[position.table_index]
    deal = game_obj.deal_from_played_game(played_game)
    other_table = np.random.choice([i for i in range(len(position.board.tables))
                                   if i != position.table_index])
    other_table_deal = game_obj.deal_from_played_game(
           position.board.tables[other_table])
    game_obj.compute_score(other_table_deal)
    game_obj.compute_score(deal)
    if not deal.result or not other_table_deal.result:
        pdb.set_trace()
    view = game_obj.kibitzer_view(deal, position.action_index)
    legal_action_ixs = game_obj.possible_action_indices(view)
    inference_features = model.make_inference_features(
            game_obj, tokenizer, view, other_table_deal.result, max_seq_length,
            max_chord_width, max_legal_actions, legal_action_ixs)
    train_features = model.make_train_features(
            game_obj, tokenizer, deal, other_table_deal, view, played_game,
            max_chord_width, max_legal_actions, legal_action_ixs)
    train_features.update(inference_features)
    return train_features


def _features_from_played_position_builder(config):
    rng = random.Random()
    def _features_from_played_position(ser_position):
        position = alphabridge_pb2.PlayedBoardPosition()
        position.ParseFromString(ser_position)
        return (position.board.board_id.source_uri,
                        _position_to_features(position, config.game,
                                              config.max_seq_length,
                                              config.max_chord_width,
                                              config.max_legal_actions))
    return _features_from_played_position


def _features_info(config):
    return tfds.features.FeaturesDict({
        "input_view_ids": tfds.features.Tensor(
                dtype=tf.int32, shape=[config.max_seq_length, config.max_chord_width]),
        "input_par_outcome": tfds.features.Tensor(
            dtype=tf.int32, shape=[1, config.max_chord_width]),
        "query_ids": tfds.features.Tensor(
            dtype=tf.int32, shape=[3 + config.max_legal_actions,
                                   config.max_chord_width]),
        "target_mask": tfds.features.Tensor(
            dtype=tf.float32, shape=[config.max_legal_actions]),
        "target_ids": tfds.features.Tensor(
            dtype=tf.int32, shape=[3 + config.max_legal_actions,
                                   config.max_chord_width]),
        "target_policy": tfds.features.Tensor(
                dtype=tf.float32, shape=[config.max_legal_actions]),
        "is_playing": tfds.features.Tensor(
            dtype=tf.int32, shape=[])
        })


class replay_buffer_sample_batch(beam.DoFn):
    def __init__(self, config):
        self.address = config.replay_buffer_address
        self.sample_batch_size = config.sample_batch_size
        self.replay_buffer = None

    def process(self, _):
        if not self.replay_buffer:
            options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
            channel = grpc.insecure_channel(self.address, options=options)
            self.replay_buffer = alphabridge_pb2_grpc.ReplayBufferStub(channel)

        req = alphabridge_pb2.SampleBatchRequest(batch_size=self.sample_batch_size)
        batch = self.replay_buffer.SampleBatch(req, wait_for_ready=True)
        for position in batch.position:
            yield position.SerializeToString()


def _features_to_example(features):
    def _to_feature(v):
        a = np.array(v)
        if a.dtype == np.int32 or a.dtype == np.int64:
            return tf.train.Feature(int64_list=tf.train.Int64List(
                    value=a.flatten()))
        elif a.dtype == np.float32 or a.dtype == np.float64:
            return tf.train.Feature(float_list=tf.train.FloatList(
                    value=a.flatten()))
        pdb.set_trace()

    example = tf.train.Example(features=tf.train.Features(feature={
        k: _to_feature(v) for k, v in features[1].items()}))
    return example.SerializeToString()

def _reinforcement_learning_filenames(config):
    """Generates filenames: TFRecord/TFExample features from ReplayBuffer."""
    for i in range(config.num_batches):
        batch_dir = os.path.join(config.tfrecord_dir, "batch-{:05d}".format(i))
        prefix = os.path.join(batch_dir, "shard")
        index_filepath = os.path.join(batch_dir, "index")
        with beam.Pipeline(options=config.beam_options) as pipeline :
            (
                    pipeline
                    | 'start' >> beam.Create(list(range(config.num_shards)))
                    | 'sample' >> beam.ParDo(replay_buffer_sample_batch(config))
                    | 'to_features' >>
                        beam.Map(_features_from_played_position_builder(config))
                    | 'serialize' >> beam.Map(_features_to_example)
                    | 'write' >> beam.io.tfrecordio.WriteToTFRecord(
                        file_path_prefix=prefix,
                        file_name_suffix=".rec",
                        num_shards=config.num_shards)
                    | 'write index' >> beam.io.WriteToText(
                            index_filepath, shard_name_template="")
            )
        with tf.io.gfile.GFile(index_filepath, "r") as f:
            yield [l.rstrip() for l in f.readlines()]


class ReinforcementConfig(object):
    def __init__(self, game, replay_buffer_address, tfrecord_temp_dir,
            num_batches=1000000, beam_options=beam.options.pipeline_options.PipelineOptions(['--runner', 'Direct']),
            sample_batch_size=16, num_shards=1,
            max_seq_length=256,
            num_actions=90, masked_prob=0.05):
        self.game = game
        self.sample_batch_size = sample_batch_size
        self.replay_buffer_address = replay_buffer_address
        self.tfrecord_dir = tfrecord_temp_dir
        self.num_shards = num_shards
        self.num_batches = num_batches
        self.beam_options = beam_options

        self.max_seq_length = max_seq_length
        self.max_chord_width = 5
        self.max_legal_actions = 35 + 1
        self.num_actions = num_actions
        self.masked_prob = masked_prob


class ReinforcementBuilder(object):
    def __init__(self, config):
        self.config = config

    def as_dataset(self):
        filenames_gen = lambda: _reinforcement_learning_filenames(self.config)
        filenames_fn = lambda: tf.data.Dataset.from_generator(
                filenames_gen,
                output_types=tf.string,
                output_shapes=self.config.num_shards)
        filenames_ds = filenames_fn()

        filenames_ds = filenames_ds.unbatch()

        raw_ds = tf.data.TFRecordDataset(filenames_ds)

        def _ex_feature_info(fi):
            if fi.dtype == tf.int32:
                return tf.io.FixedLenFeature(fi.shape, tf.int64)
            else:
                return tf.io.FixedLenFeature(fi.shape, fi.dtype)

        features_info = {k: _ex_feature_info(v)
                for k,v in _features_info(self.config).items()}

        def _parse_fn(pb):
            try:
                features = tf.io.parse_single_example(pb, features_info)
            except:
                pdb.set_trace()
            for k in features:
                if features[k].dtype == tf.int64:
                    features[k] = tf.cast(features[k], tf.int32)
            return features

        features_ds = raw_ds.map(_parse_fn)
        return features_ds


def do_generate(_):
    game_obj = bridgegame.Game(n=2)
    cfg = ReinforcementConfig(game_obj, "localhost:10000", "/tmp", sample_batch_size=16)
    b = ReinforcementBuilder(cfg)
    tokenizer = bridgegame.Tokenizer(game_obj)
    i = 1
    for d in b.as_dataset().as_numpy_iterator():
        i += 1


if __name__ =="__main__":
    app.run(do_generate)
