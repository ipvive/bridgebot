import io
import tensorflow as tf

from bridge import lin
import bridge.game as bridgegame
from pb import alphabridge_pb2

import pdb


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("input_files", None, "Comma-separated list of input .lin files.")


def contract_class(board):
    for table, deal in board.tables.items():
        out = deal.result.outcome()
        if out:
            making = out == "=" or int(out) > 0
            if deal.contract_level() == "7":
                yield board, making 
                break


def useful_penalty(board, game):
    played_games = [game.played_game_from_deal(table)
            for table in board.tables.values()]
    played_board = alphabridge_pb2.PlayedBoard(tables=played_games)
    game.score_played_board(played_board)
    for (table, deal), pg in zip(board.tables.items(), played_board.tables):
        out = deal.result.outcome()
        if out:
            making = out == "=" or int(out) > 0
            doubled = deal.contract_doubled() != "undoubled"
            profit = pg.result.comparison_score
            seat = deal.contract_seat()
            if seat == "North" or seat == "South":
                profit = -profit
            if doubled:
                yield board, making, profit, pg.result


def boards_gen(input_files, game, parser):
    for input_file in input_files:
        lindata = tf.data.TFRecordDataset(input_file)
        for lin in lindata:
            reader = io.StringIO(lin.numpy().decode('utf-8'))
            reader.name = reader.readline().strip()
            played_boards, error_counts = parser.parse(reader, game)
            for idx, board in played_boards.items():
                yield board


def main(_):
    parser = lin.Parser()
    game = bridgegame.Game()

    input_files = []
    for input_pattern in FLAGS.input_files.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    for board in boards_gen(input_files, game, parser):
        for board, making, profit, result in useful_penalty(board, game):
            results = ["_".join(t.result.tokens) for t in board.tables.values()]
            print(f"Doubled: {making, profit, results}")


if __name__ == '__main__':
    flags.mark_flag_as_required("input_files")
    tf.compat.v1.app.run()
