"""Usage:
$ ../../bazel-bin/bridgebot/replay_buffer/linux_amd64_stripped/replay_buffer &
$ python3 replay_buffer_smoke_test
"""
import grpc
import tensorflow as tf
import numpy as np
import logging
import random
import unittest
import unittest.mock

import selfplay
import bridge.game as bridgegame
import inference
import pb.alphabridge_pb2 as alphabridge_pb2
import pb.alphabridge_pb2_grpc as alphabridge_pb2_grpc

class DummyNetworks(object):
    def representation(self, view):
        return ('HIDDEN', random.randrange(bridgegame.num_scores),
                np.random.random(bridgegame.num_actions))

    def dynamics(self, hidden_state, action_idx):
        return ('HIDDEN',
                random.randrange(bridgegame.num_action_verbs),
                random.randrange(bridgegame.num_scores),
                np.random.random(bridgegame.num_actions))


class ReplayBufferPutStressTest(tf.test.TestCase):
    def testBuffer(self):
        channel = grpc.insecure_channel("localhost:10000")
        replay_buffer = alphabridge_pb2_grpc.ReplayBufferStub(channel)
        game = bridgegame.Game()
        rng = random.Random()
        for i in range(1000000):
            if i > 0 and i % 100 == 0:
                logging.info("Put {} boards".format(i))
            deal = game.random_deal(rng)
            while not deal.is_final():
                while True:
                    ix = rng.randint(0, bridgegame.num_actions - 1)
                    deal = game.execute_action_index(deal, ix)
                    if not deal.error:
                        break
                    deal.error = None
            played_board = alphabridge_pb2.PlayedBoard(
                    tables=[game.played_game_from_deal(deal)])
            game.score_played_board(played_board)
            replay_buffer.Put(played_board)


if __name__ == "__main__":
    tf.test.main()
