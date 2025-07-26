from absl import flags, app
import bridge.game as bridgegame
import pb.alphabridge_pb2 as alphabridge_pb2
import pb.alphabridge_pb2_grpc as alphabridge_pb2_grpc
import grpc

import sys

flags.DEFINE_string("rb", "", "replay buffer address")

flags.DEFINE_integer("batch_size", 8, "sample batch size")

FLAGS = flags.FLAGS


class ReplayBuffer():
  def __init__(self, address, sample_batch_size):
    self.address = address
    self.sample_batch_size = sample_batch_size
    self.replay_buffer = None

  def get_batch(self):
    if not self.replay_buffer:
      options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
      channel = grpc.insecure_channel(self.address, options=options)
      self.replay_buffer = alphabridge_pb2_grpc.ReplayBufferStub(channel)

    req = alphabridge_pb2.SampleBatchRequest(batch_size=self.sample_batch_size)
    batch = self.replay_buffer.SampleBatch(req, wait_for_ready=True)
    for position in batch.position:
      yield position


def main(_):
    replay_buffer = ReplayBuffer(FLAGS.rb, FLAGS.batch_size)
    game = bridgegame.Game()
    for i, played_board_position in enumerate(replay_buffer.get_batch()):
        print("*" * 10 + f" Played Board {i} " + "*" * 10)
        for j, played_game in enumerate(played_board_position.board.tables):
            print(f"Table {j}")
            print(game.deal_from_played_game(played_game))
            print(played_game.result)


if __name__ == "__main__":
    app.run(main)
