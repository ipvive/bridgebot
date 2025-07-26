from absl.testing import absltest

import bridge.game as bridgegame
from bridge import fastgame
import ncard.game as ncardgame

import dims


class DimsTest(absltest.TestCase):
    def setUp(self):
        self.games = [bridgegame.Game(), fastgame.Game(), ncardgame.Game()]

    def test_action_extraction(self):
        n = bridgegame._actions.index["3_Clubs"]
        d = dims.extract_observables_from_action_id_and_seat(n, 3)
        self.assertEqual(d["next_bid"], (2, 0, 3))
        self.assertEqual(d["next_call"], (None, None))
        self.assertEqual(d["next_card"], (None, None, None))

        n = bridgegame._actions.index["double"]
        d = dims.extract_observables_from_action_id_and_seat(n, 3)
        self.assertEqual(d["next_bid"], (None, None, None))
        self.assertEqual(d["next_call"], (1, 3))
        self.assertEqual(d["next_card"], (None, None, None))

        n = bridgegame._actions.index["Club_Three"]
        d = dims.extract_observables_from_action_id_and_seat(n, 3)
        self.assertEqual(d["next_bid"], (None, None, None))
        self.assertEqual(d["next_call"], (None, None))
        self.assertEqual(d["next_card"], (0, 1, 3))

    def test_game_extraction(self):
        for game in self.games:
            print(game)
            deal = game.Deal()
            game.set_board_number(deal, 1)
            d = dims.extract_observables_from_deal(deal)


if __name__ == "__main__":
    absltest.main()
