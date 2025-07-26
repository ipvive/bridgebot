import unittest

from absl.testing import absltest

import rgeom


class RDimsTest(absltest.TestCase):
    def test_constuction(self):
        seat_t = rgeom.Dim("seat", ["me", "lho", "partner", "rho"])
        suit_t = rgeom.Dim("suit", [x for x in "CDHS"])
        rank_t = rgeom.Dim("rank", [x for x in "TJQKA"])
        card_t = rgeom.Space("card", [suit_t, rank_t])
        card_t2 = suit_t * rank_t
        cards_t = rgeom.Tensor("cards", card_t)
        unit_interval_t = (0 <= rgeom.reals() <= 1)
        card_info_t = rgeom.Map("card_info", card_t, unit_interval_t)
        cards_info_t = rgeom.Map("cards_info", cards_t, unit_interval_t)

        partner = rgeom.constant(seat_t, "partner")
        diamond = rgeom.constant(suit_t, "D")
        queen = rgeom.constant(rank_t, "Q")
        DQ1 = diamond * queen
        DQ2 = rgeom.constant(card_t, ("D", "Q"))
        DQ3 = rgeom.constant(card_t, {"rank": "Q", "suit": "D"})
        DQ4 = card_t.constant("DQ")
        my_hand = rgeom.constant(cards_t, ["CT", "CJ", DQ3, "HK", ("S", "A")])

        #self.assertEqual(card_t, card_t2)
        #self.assertEqual(DQ1, DQ2)
        #self.assertEqual(DQ2, DQ3)
        #self.assertEqual(DQ2, DQ4)
        
#    def test_sum(self):
#        self.assertEqual(self.tensor.sum(), self.array.sum())
#
#    def test_argmax(self):


if __name__ == "__main__":
    absltest.main()
