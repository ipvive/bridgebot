#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import logging

from absl import app,  flags
import jax
import jax.numpy as jnp
import haiku as hk
from jaxline import platform

from bridgebot.ncard import experiment
from bridgebot.ncard.config import c0f
from bridgebot.ncard import model
from bridgebot.ncard import game as bridgegame
from bridgebot.ncard import chords


class Analyzer:
    def __init__(self, game):
        self.game = game
        self.tokenizer = bridgegame.Tokenizer(game)
        experiment.checkpoint_dir = "data/c0f"
        config = c0f.get_config()
        self.e = experiment.Experiment(
                mode='infer', init_rng=jax.random.PRNGKey(config.random_seed),
                config=config.experiment_kwargs.config)
        self.rng = jax.random.PRNGKey(config.random_seed)

        checkpointer = experiment.NaiveDiskCheckpointer(config, None)
        state = checkpointer.get_experiment_state("latest")
        state.experiment_module = self.e
        checkpointer.restore("latest")
        self.e._initialize_train()

    def __call__(self, views, par_results):
        if isinstance(views, list):
            features = [model.make_inference_features(
                    self.game, self.tokenizer, view, par_result, 256, 5, 35 + 1,
                    self.game.possible_action_indices(view)) \
                        for view, par_result in zip(views, par_results)]
            features = {k: jnp.array([f[k] for f in features])
                        for k in features[0]}
        else:
            features = model.make_inference_features(
                    self.game, self.tokenizer, views, par_results, 256, 5, 35 + 1,
                    self.game.possible_action_indices(views))
            features = {k:jnp.array([v]) for k, v in features.items()}
        (logits, t), state = self.e.forward.apply(
                self.e._params, self.e._state, self.rng, features, False)
        value_gt = float(logits[0,0,2] - logits[0,0,3])
        value_geq = float(logits[0,1,2] - logits[0,1,3])
        outcome = chords.generate(logits[0,2], self.e.codecs["outcome"], return_ll=True)
        outcome = [(self.tokenizer.all_tokens[i],ll) for (i,ll) in outcome[0]]
        tokens = self.tokenizer.ids_to_tokens(features["input_view_ids"][0])
        return value_gt, value_geq, outcome, tokens


def main(argv):
    game = bridgegame.Game(n=2)
    a =  Analyzer(game)

    deal = game.table_view(game.random_deal(), 0)
    deal = bridgegame._parse_bidding(game, deal, "1S-P-P-P")
    deal = bridgegame._parse_play(game, deal, "CA SA CK DA DK HK SK HA")
    view = game.kibitzer_view(deal, 8)
    for i in range(deal.num_actions()):
        print(a(game.kibitzer_view(deal, i), deal.result))
    import timeit
    #print(timeit.timeit(lambda: a(deal, deal.result), number=1000))
    print(timeit.timeit(lambda: a([deal] * 10, [deal.result] * 10), number=100))
    print(timeit.timeit(lambda: a([deal] * 20, [deal.result] * 20), number=50))
    print(timeit.timeit(lambda: a([deal] * 50, [deal.result] * 50), number=20))
    print(timeit.timeit(lambda: a([deal] * 100, [deal.result] * 100), number=10))


if __name__ == "__main__":
    logging.getLogger('apache_beam').setLevel(logging.ERROR)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    app.run(main)
