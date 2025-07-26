import io
import random
import tensorflow as tf
import unittest

import bridgegame
import tokens


class TokenizeTest(tf.test.TestCase):
    def setUp(self):
        self.game = bridgegame.Game()
        self.tokenizer = tokens.Tokenizer()
        self.rng = random.Random(0)


    def test_invalid_player_names(self):
        deal = self.game.Deal()
        deal.players = {i:"NOT_A_VALID_PLAYER_NAME" for i in bridgegame._seats.tokens}
        deal.vulnerability = []
        deal.scoring = "IMPs"
        deal = self.game.set_dealer(deal, "South")
        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_call(deal, "pass")
        tokens = self.tokenizer.tokenize_view(deal, self.rng)
        ids = self.tokenizer.tokens_to_ids(tokens)

    def test_no_result(self):
        deal = self.game.Deal()
        deal.players = {i:i for i in bridgegame._seats.tokens}
        deal.vulnerability = []
        deal.scoring = "IMPs"
        deal = self.game.set_dealer(deal, "South")
        tokens = self.tokenizer.tokenize_view(deal, self.rng)
        ids = self.tokenizer.tokens_to_ids(tokens)

    def test_no_vulnerability(self):
        deal = self.game.Deal()
        deal.players = {i:i for i in bridgegame._seats.tokens}
        deal.scoring = "IMPs"
        deal = self.game.set_dealer(deal, "South")
        deal.result = bridgegame.Event(["3", "notrump", "East", "="])
        tokens = self.tokenizer.tokenize_view(deal, self.rng)
        ids = self.tokenizer.tokens_to_ids(tokens)

    def test_no_scoring(self):
        deal = self.game.Deal()
        deal.players = {i:i for i in bridgegame._seats.tokens}
        deal.vulnerability = []
        deal = self.game.set_dealer(deal, "South")
        deal.result = bridgegame.Event(["3", "notrump", "East", "="])
        tokens = self.tokenizer.tokenize_view(deal, self.rng)
        ids = self.tokenizer.tokens_to_ids(tokens)
        

if __name__ == "__main__":
    tf.test.main()
