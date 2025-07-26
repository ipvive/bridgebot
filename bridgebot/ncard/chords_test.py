import numpy as np
import math

from absl.testing import absltest
import jax.numpy as jnp

from bridgebot.ncard import chords
from bridgebot.ncard import game as bridgegame

import pdb

class ChordsTest(absltest.TestCase):
    def setUp(self):
        self.game = bridgegame.Game(n=2)
        self.tokenizer = bridgegame.Tokenizer(self.game)
        self.builder = chords.MachineBuilder()
        self.rng = np.random.default_rng()

    def testGenerate(self):
        a = chords.all_action_chords(self.game)
        start = self.builder.traverse(a)
        codec = self.builder.compile([start])[0]
        for _ in range(10):
            l = 1 - self.rng.random(size=len(codec.token))
            seq, accepted  = chords.generate(l, codec)
            print(seq)
            print(chords.log_likelihood(l, codec, tuple(seq))[0])

    def testBoolLogLikelihood(self):
        a = chords.all_bool_chords(self.game)
        start = self.builder.traverse(a)
        codec = self.builder.compile([start])[0]
        l = np.zeros(len(codec.token))
        ll = chords.log_likelihood(l, codec, ("[YES]", "[EOS]"))[0]
        print(codec)
        self.assertAlmostEqual(ll, math.log(0.5))
        ll = chords.log_likelihood(l, codec, ("[NO]", "[EOS]"))[0]
        self.assertAlmostEqual(ll, math.log(0.5))
        ll = chords.log_likelihood(l, codec, ("[YES]",))[0]
        self.assertAlmostEqual(ll, math.log(0.5))
        ll = chords.log_likelihood(l, codec, ("[NO]",))[0]
        self.assertAlmostEqual(ll, math.log(0.5))
        l[codec.token_index["[YES]"]] = 3.
        l[codec.token_index["[NO]"]] = -1
        ll = chords.log_likelihood(l, codec, ("[YES]",))[0]
        ll2 = chords.batch_bool_log_likelihood(
                l, np.array([codec.token_index["[YES]"]]),
                YES=codec.token_index["[YES]"],
                NO=codec.token_index["[NO]"])
        expected = math.log(math.exp(4.) / (1 + math.exp(4.)))
        self.assertAlmostEqual(ll, expected, 6)
        self.assertAlmostEqual(ll2, expected, 6)
        ll = chords.log_likelihood(l, codec, ("[NO]",))[0]
        ll2 = chords.batch_bool_log_likelihood(
                l, np.array([codec.token_index["[NO]"]]),
                YES=codec.token_index["[YES]"],
                NO=codec.token_index["[NO]"])
        expected = math.log(math.exp(-4.) / (1 + math.exp(-4.)))
        self.assertAlmostEqual(ll, expected, 6)
        self.assertAlmostEqual(ll2, expected, 6)

        for _ in range(10):
            l = 1 - self.rng.random(size=len(codec.token))
            seq, accepted  = chords.generate(l, codec)
            print(seq)
            print(chords.log_likelihood(l, codec, tuple(seq))[0])

    def testErrors(self):
        a = chords.all_outcome_chords(self.game)
        start = self.builder.traverse(a)
        codec = self.builder.compile([start])[0]
        id_codec = chords.id_codec(codec, self.tokenizer.all_tokens)
        print(id_codec)
        l = np.zeros(len(codec.token))
        ids = [15, 0, 0, 0, 0]
        token_seq = self.tokenizer.ids_to_tokens([ids])[0]
        print("HELLO", token_seq)
        ll, ok = chords.log_likelihood(l, codec, tuple(token_seq))
        ll2, ok2 = chords.log_likelihood(l, id_codec, tuple(ids))
        self.assertTrue(ok)
        self.assertTrue(ok2)
        self.assertEqual(ll, ll2)

if __name__ == "__main__":
    absltest.main()
