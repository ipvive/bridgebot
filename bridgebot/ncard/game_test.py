from bridgebot.ncard import game
from absl.testing import absltest

import pdb

class TestGame(absltest.TestCase):
    def test_traverse(self):
        g = game.Game(n=1)
        d = g.Deal()
        g.set_board_number(d, 1)
        d = g.execute_action_ids(d, [3])

        def _traverse(d):
            idxs = g.possible_action_indices(d)
            self.assertEqual(d.error, None)
            if (len(idxs) == 0) != d.is_final():
                pdb.set_trace()
            self.assertEqual(len(idxs) == 0, d.is_final())
            for idx in idxs:
                dc = d.copy_replay_state()
                _traverse(g.execute_action_ids(dc, [idx]))

        _traverse(d)

    def test_play_heart_ace(self):
        g = game.Game(n=1)
        deal = g.parse_shorthand_problem("HA", "1N-P-P-P", "?")
        idxs = g.possible_action_indices(deal)
        self.assertEqual(idxs, [76])



if __name__ == "__main__":
    absltest.main()
