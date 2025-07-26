from __future__ import division
import asyncio
import os
import random
import threading

import absl.app
import absl.flags as flags
import absl.logging as logging
import grpc
from google.cloud import bigtable

import bridge.game as bridgegame
from bridge import tokens
import pb.alphabridge_pb2 as alphabridge_pb2
import pb.alphabridge_pb2_grpc as alphabridge_pb2_grpc
import simulate


class BoardAnalyzer:
    def __init__(self):
        self.played_boards = []

    def analyze(self):
        total_comparison_score = sum(played_board.tables[0].result.comparison_score
            for played_board in self.played_boards)
        return total_comparison_score / len(self.played_boards)

    def Put(self, played_board, wait_for_ready=True):
        logging.info("Saving completed board for analysis")
        self.played_boards.append(played_board)
        logging.debug("%s", played_board)


async def run_evaluation(config, inference_pipes, game, rng, board_io):
    saved_games = board_io.get_saved_games()
    await simulate.simulate_one_board(config, inference_pipes, game, rng, board_io, saved_games)


FLAGS = flags.FLAGS

flags.DEFINE_string("comparison_pipe_address", "localhost:20001",
    "Address of comparison inference pipe service")


async def run_evaluate():
    config = alphabridge_pb2.SimulationConfig(
        num_tables=2,
        max_moves=150,
        num_simulations_per_move=FLAGS.num_simulations_per_move,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        ucb_pb_c_base=19652,
        ucb_pb_c_init=1.25,
        num_parallel_inferences=FLAGS.num_parallel_inferences)

    # TODO: remove after https://github.com/grpc/grpc/issues/24018 is resolved.
    os.environ["GRPC_DNS_RESOLVER"] = "native"
    os.environ["GRPC_VERBOSITY"] = "debug"

    logging.info("connecting to inference pipe at %s", FLAGS.inference_pipe_address)
    pipe_channel = grpc.aio.insecure_channel(FLAGS.inference_pipe_address)
    pipe = alphabridge_pb2_grpc.InferencePipeStub(pipe_channel)

    logging.info("connecting to comparison pipe at %s", FLAGS.comparison_pipe_address)
    comparison_pipe_channel = grpc.aio.insecure_channel(FLAGS.comparison_pipe_address)
    comparison_pipe = alphabridge_pb2_grpc.InferencePipeStub(comparison_pipe_channel)

    game = bridgegame.Game()
    rng = random.Random()

    client = bigtable.Client()
    instance = client.instance(FLAGS.bigtable_instance)
    table = instance.table(FLAGS.bigtable_table)

    row_keys = [f"{FLAGS.shard_id}:{i:05d}" for i in range(FLAGS.concurrency)]

    board_analyzer = BoardAnalyzer()
    saver = simulate.Saver(board_analyzer, table, FLAGS.bigtable_column_family,
            FLAGS.bigtable_column_template, config.num_tables, row_keys, clear_rows=False)

    logging.info("starting io thread")
    io_thread = threading.Thread(target=lambda: saver.run(), daemon=True)
    io_thread.start()

    logging.info("creating %d concurrent simulation tasks", len(row_keys))
    tasks = [asyncio.create_task(
        run_evaluation(config, (pipe, comparison_pipe), game, rng,
            simulate.BoardSaver(saver, row_key)))
        for row_key in row_keys]
    logging.info("gathering...")
    await asyncio.gather(*tasks)

    average_score = board_analyzer.analyze()
    print(f"Average comparison score for {FLAGS.shard_id} is {average_score}.")


def main(_):
    logging.set_verbosity(logging.DEBUG)
    asyncio.run(run_evaluate())


if __name__ == "__main__":
    absl.app.run(main)
