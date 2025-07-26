"""Implements optimal play for 1-card bridge."""
from absl.testing import absltest

from ncard import rdims


class OptimalPlayer:
    """Implements:
        1. Play my card.
        2. Double notrump.
        3. Double or redouble with trump length.
        4. Pass partner's doubled or redoubled contract.
        5. Bid my suit.
        6. Pass.
    """
    pass  # TODO: finish after ready to test.


class OptimalPlayerTest:
    def test_spot_check:
        """Passing the spot check is sufficient for intended use."""
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
        pass  # TODO: finish after ready to test seed.kg, executor.


if __name__ == "__main__":
    pass  # TODO
