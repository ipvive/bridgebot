import io
from absl import flags, app

from bridge import lin
import bridge.game as bridgegame
from pb import alphabridge_pb2

import pdb


FLAGS = flags.FLAGS


flags.DEFINE_string("input_file", None, "input .lin file.")


def boards_gen(input_file, game, parser):
    with open(input_file, "r") as f:
            b = f.read()
            reader = io.StringIO(b)
            reader.name = reader.readline().strip()
            pdb.set_trace()
            played_boards, error_counts = parser.parse(reader, game)
            for idx, board in played_boards.items():
                yield board


def main(_):
    parser = lin.Parser()
    game = bridgegame.Game()

    for board in boards_gen(FLAGS.input_file, game, parser):
        pdb.set_trace()


if __name__ == '__main__':
    flags.mark_flag_as_required("input_file")
    app.run(main)
