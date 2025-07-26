import tensorflow as tf

from bridge import lin

flags = tf.compat.v1.flags


FLAGS = flags.FLAGS


flags.DEFINE_string("input_files", None, "Comma-separated list of input .lin files.")


class CountedDeal(object):
    def __init__(self):
        self.table_name = None
        self.board_name = None
        self.result = 'not_a_real_result'
        self.error = None


class Counter(object):
    def __init__(self):
        self.player_counts = {}

    def Deal(self):
        return CountedDeal()

    def accept_claim(self, deal, _):
        return deal

    def set_dealer(self, deal, _):
        return deal

    def set_players(self, deal, *players):
        for player in players:
            player = player.lower()
            if player in self.player_counts:
                self.player_counts[player] += 1
            else:
                self.player_counts[player] = 1
        return deal

    def set_result(self, deal, *_):
        return deal

    def give_card(self, deal, *_):
        return deal

    def make_call(self, deal, *_):
        return deal

    def make_bid(self, deal, *_):
        return deal

    def play_card(self, deal, *_):
        return deal

    def accept_claim(self, deal, *_):
        return deal

    def add_explanation(self, deal, *_):
        return deal

    def add_commentary(self, deal, *_):
        return deal


def main(_):
    parser = lin.Parser()
    game = Counter()

    input_files = []
    for input_pattern in FLAGS.input_files.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    for input_file in input_files:
#        tf.compat.v1.logging.info(input_file)
        with tf.io.gfile.GFile(input_file, "r") as reader:
            parser.parse(reader, game)

    for player, n in game.player_counts.items():
        print("{} {}".format(n, "".join(player.split("\n"))))


if __name__ == '__main__':
    flags.mark_flag_as_required("input_files")
    tf.compat.v1.app.run()
