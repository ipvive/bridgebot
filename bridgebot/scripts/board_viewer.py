import tensorflow as tf
import bridge.game as bridgegame
import pb.alphabridge_pb2 as alphabridge_pb2

import sys

dataset = tf.data.TFRecordDataset(sys.argv[1])

game = bridgegame.Game()
played_board = alphabridge_pb2.PlayedBoard()
for i, data in enumerate(dataset):
    print("*" * 10 + f" Played Board {i} " + "*" * 10)
    played_board.ParseFromString(data.numpy())
    for j, played_game in enumerate(played_board.tables):
        print(f"Table {j}")
        print(game.deal_from_played_game(played_game))
        print(played_game.result)
