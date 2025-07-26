import random
import pdb

from absl.testing import absltest
from absl import flags
from unittest import mock

import bridge.game as bridgegame

import simulate


class SimulateTest(absltest.TestCase):
  def test_played_game_to_from_sim(self):
    game = bridgegame.Game()
    rng = random.Random()
    deal = game.random_deal(rng)
    sim = simulate.SimulatedGame(game, deal, 0)
    sim.apply(0) 
    sim.apply(1) 
    sim.apply(2)
    sim.child_visit_fractions = [{0: 0.5, 1: 0.5}, {0: 0.25, 1: 0.75}, {0: 0.125, 2: 0.875}]
    played_game = simulate.played_game_from_sim(sim)
    recovered_sim = simulate.SimulatedGame.from_played_game(game, played_game, sim.table_idx)
    self.assertEqual(len(sim.child_visit_fractions), len(recovered_sim.child_visit_fractions))
    for a, b in zip(sim.child_visit_fractions, recovered_sim.child_visit_fractions):
      self.assertDictEqual(a, b)
    self.assertEqual(sim.current_deal().num_actions(),
      recovered_sim.current_deal().num_actions())


if __name__ == "__main__":
  absltest.main()
