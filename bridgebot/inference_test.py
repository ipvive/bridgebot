import os
import random
import threading
import time
from unittest import mock

import numpy as np
import tensorflow as tf

from pb import alphabridge_pb2
import bridge.game as bridgegame
from bridge import tokens
from mubert import modeling

import inference

class EstimatorTestConfig:
    def __init__(self):
        self.use_tpu = False
        self.model_dir = "/dev/null"
        self.simulation_batch_size = 17


class InferenceTestConfig:
    def __init__(self):
        self.init_checkpoint = ""
        try:
            self.mubert_config = modeling.MuBertConfig.from_json_file(
                    "bridgebot/testdata/bridgebot_config_prime.json")
        except tf.errors.NotFoundError as err:
            print(os.getcwd())
            self.assertFalse(err)
        self.embedding_size = self.mubert_config.representation.hidden_width
        self.max_seq_length = 113
        self.hidden_length = 11
        self.latent_vectors_length = self.max_seq_length + self.hidden_length
        self.micro_batch_size = 13
        self.num_micro_batches_per_mini_batch = 17
        self.num_parallel_simulations = 13 * 17 + 29
        self.max_action_lookahead = 23
        self.estimator_config = EstimatorTestConfig()


class InferenceTest(tf.test.TestCase):
    def setUp(self):
        self.config = InferenceTestConfig()
        self.inference = inference.Inference(self.config)
        game = bridgegame.Game()
        rng = random.Random()
        deal = game.random_deal(rng)
        view = game.actor_view(deal, 0)
        self.tokenizer = tokens.Tokenizer()
        self.view_tokens = self.tokenizer.tokenize_view(view, rng)

    def test_predict_slow(self):
        view_ids = self.tokenizer.tokens_to_ids(self.view_tokens)
        predictions = self.inference.predict_slow(view_ids, [[0, 89]])

    def test_bench_proto_to_features(self):
        self.config.max_action_lookahead = 80
        self.config.micro_batch_size = 10
        actions = np.random.random_integers(0, 90,
                (self.config.micro_batch_size, self.config.max_action_lookahead))
        aips = [alphabridge_pb2.ActionIndexPath(action_index=a.tolist())
                for a in actions]
        fb = alphabridge_pb2.FeaturesBatch(
                micro=[alphabridge_pb2.FeaturesMicroBatch(
                    view_token=self.view_tokens,
                    action_path=aips)])
        ser_fb = fb.SerializeToString()
        start = time.perf_counter()
        for _ in range(1000):
            fb2 = alphabridge_pb2.FeaturesBatch()
            fb2.ParseFromString(ser_fb)
            _ = self.inference._proto_to_features(fb2, 1)
        duration_per = 0.001 * (time.perf_counter() - start)
        print(f"proto_to_features {duration_per}s {1/duration_per}/s")

    def test_bench_predictions_to_proto(self):
        self.config.max_action_lookahead = 80
        self.config.micro_batch_size = 10
        predictions = {
                "unique_id": 1234,
                "next_to_act": np.random.random_integers(0, 4,
                    (1, self.config.micro_batch_size)),
                "value": np.random.random(
                    (1, self.config.micro_batch_size)),
                "policy": np.random.random(
                    (1, self.config.micro_batch_size, 90))}
        start = time.perf_counter()
        for _ in range(1000):
            pb = self.inference._predictions_to_proto(predictions)
            _ = pb.SerializeToString()
        duration_per = 0.001 * (time.perf_counter() - start)
        print(f"predictions_to_proto {duration_per}s {1/duration_per}/s")


if __name__ == "__main__":
    tf.test.main()
