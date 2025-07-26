from absl.testing import absltest
from unittest import mock

from bridgebot.ncard import kg
from bridgebot.ncard import game as bridgegame
from bridgebot.ncard.onecard import optimal_kg

import pdb

class OptimalTest(absltest.TestCase):
    def test_links(self):
        b = kg.Builder()
        optimal_kg.build_kg(b)
        rkg = kg.link(b)
        root = rkg.nodes['Optimal play.']
        leaf = root.deps['Pass.'].deps['Actor state.']
        self.assertTrue(callable(root.fn))
        self.assertListEqual(list(root.deps.keys()), optimal_kg._priorities)
        self.assertListEqual([d.name for d in root.ordered_recursive_deps],
                ['Actor state.'] + optimal_kg._priorities)

    def test_execution(self):
        casedata = [
                "holding: HA bidding: 1N-P-P-P play: ? == HA",
                "holding: HA bidding: 1N-? == X",
                "holding: HA bidding: 1H-? == X",
                "holding: HA bidding: 1H-X-? == XX",
                "holding: HA bidding: 1C-X-? == P",
                "holding: HA bidding: 1C-X-P-P; XX-P-? == P",
                "holding: HA bidding: ? == 1H",
                "holding: HA bidding: 1S-? == P",
        ]
        b = kg.Builder()
        optimal_kg.build_kg(b)
        lkg = kg.link(b)
        game = bridgegame.Game(n=1)
        for c in casedata:
            actor_state, expected_action = _parse_case_data(game, c)
            v = kg.execute(lkg, {"game": game, "actor state": actor_state},
                {"actions": ("Optimal play.", "actions")})
            self.assertListEqual(v["actions"], [expected_action], c)


def _parse_case_data(game, c):
    c, expected = c.split(" == ")
    _, c = c.split("holding: ")
    holding, c = c.split(" bidding: ")
    cc  = c.split(" play: ")
    if len(cc) > 1:
        bidding, play = cc
    else:
        bidding, play = c, None

    deal = game.parse_shorthand_problem(
            holding=holding, bidding=bidding, play=play)
    expected = game.parse_action(expected)
    return deal, expected


if __name__ == "__main__":
    absltest.main()
