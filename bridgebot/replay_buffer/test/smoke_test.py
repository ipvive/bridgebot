"""Usage:
$ ../../bazel-bin/bridgebot/replay_buffer/linux_amd64_stripped/replay_buffer &
$ python3 replay_buffer_smoke_test

"""

import tensorflow as tf
import subprocess
import grpc
import pb.alphabridge_pb2 as pb
import pb.alphabridge_pb2_grpc as pb_grpc


class SmokeTest(tf.test.TestCase):
    def setUp(self):
        channel = grpc.insecure_channel("localhost:10000")
        self.stub = pb_grpc.ReplayBufferStub(channel)

    def testRoundTrip(self):
        action = pb.Action(token="ACT")
        table = pb.PlayedGame(actions=[action])
        board = pb.PlayedBoard(tables=[table])
        self.stub.Put(board)
        req = pb.SampleBatchRequest(batch_size=1)
        batch = self.stub.SampleBatch(req)
        print(batch)

if __name__ == "__main__":
    tf.test.main()
