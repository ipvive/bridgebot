from absl.testing import absltest

import bridge.game as bridgegame
from bridge import fastgame
import ncard.game as ncardgame

import dims

import pdb


class DimsTest(absltest.TestCase):
    def setUp(self):
        self.games = [bridgegame.Game(), fastgame.Game(), ncardgame.Game()]

    def test_action_extraction(self):
        g = ncardgame.Game(n=5)
        n = g.actions.index("3_Clubs")
        d = dims.extract_action_from_action_id_and_seat(g, n, 3)
        self.assertDictEqual(d["next_bid"], {'level': 2, 'strain': 0, 'seat': 3})
        self.assertDictEqual(d["next_call"], {'call': None, 'seat': None})
        self.assertDictEqual(d["next_card"], {'strain': None, 'rank': None, 'seat': None})

        n = g.actions.index("double")
        d = dims.extract_action_from_action_id_and_seat(g, n, 3)
        self.assertDictEqual(d["next_bid"], {'level': None, 'strain': None, 'seat': None})
        self.assertDictEqual(d["next_call"], {'call': 1, 'seat': 3})
        self.assertDictEqual(d["next_card"], {'strain': None, 'rank': None, 'seat': None})

        n = g.actions.index("Club_Jack")
        d = dims.extract_action_from_action_id_and_seat(g, n, 3)
        self.assertDictEqual(d["next_bid"], {'level': None, 'strain': None, 'seat': None})
        self.assertDictEqual(d["next_call"], {'call': None, 'seat': None})
        self.assertDictEqual(d["next_card"], {'strain': 0, 'rank': 1, 'seat': 3})

    def test_state_extraction(self):
        game = ncardgame.Game(7)
        deal = game.Deal()
        game.set_board_number(deal, 1)
        d = dims.extract_state_from_deal(game, deal)

    def test_score_extraction(self):
        game = ncardgame.Game(n=1)
        d1 = game.Deal()
        game.set_board_number(d1, 1)

        def _after(deal, actions):
            idxs = [game.actions.index(a) for a in actions]
            return game.execute_action_ids(deal, idxs)

        d1 = _after(d1, [
            "pass",
            "pass",
            "pass",
            "pass",
        ])
        d2 = game.Deal()
        game.set_board_number(d2, 1)
        d2 = _after(d2, [
            "1_Spades",
            "pass",
            "pass",
            "pass",
            "Club_Ace",
            "Diamond_Ace",
            "Heart_Ace",
            "Spade_Ace",
        ])
        scores = dims.extract_scores_from_final_states(game, d1, d2)
        self.assertDictEqual(scores, {
            "imps": {'imps': game.imps.index(-2)},
            "positive_score": {'winloss': game.winloss.index(0)},
            "better_score": {'winloss': game.winloss.index(-1)},
        })


if __name__ == "__main__":
    absltest.main()
