from absl import logging
import numpy as np

from bridgebot.ncard import kg
from bridgebot.ncard import dims

import pdb

@kg.rule(name="Actor state.",
        inputs=["game", "actor state"],
        outputs=["contract", "next_to_act", "dealt_cards", "played_cards"])
def actor_state(n):
   deal = n.inputs["actor state"]
   answers = dims.extract_state_from_deal(n.inputs["game"], deal)
   for k in n.outputs.allowed_keys:
       n.outputs[k] = answers[k]


@kg.rule(name="Doubled or redoubled.", inputs=["contract"], outputs=["doubled or redoubled"])
def doubled_or_redoubled(n):
    contract = n.inputs["contract"]
    if contract and contract["call"] != 0:
        n.outputs["doubled or redoubled"] = True
    else:
        n.outputs["doubled or redoubled"] = False


@kg.rule(name="My cards.",
        inputs=["next_to_act", "dealt_cards", "played_cards"],
        outputs=["my cards"])
def my_cards(n):
    my_seat = n.inputs["next_to_act"]["seat"]
    dealt = n.inputs["dealt_cards"]
    played = n.inputs["played_cards"]
    n.outputs["my cards"] = dealt[my_seat,:,:] & ~played


@kg.rule(name="My suit.", inputs=[], outputs=["my suit"])
def my_suit(n):
    my_suit = n.inputs["my cards"].sum(axis=1).argmax()
    n.outputs["my suit"] = my_suit


@kg.rule(name="My card.", inputs=["my cards"], outputs=["my card"])
def my_card(n):
    ix = n.inputs["my cards"].argmax()
    ixs = np.unravel_index(ix, n.inputs["my cards"].shape)
    n.outputs["my card"] = {'suit': ixs[0], 'rank': ixs[1] + 13 - n.inputs["my cards"].shape[1]}


@kg.rule(name="Play my card.", inputs=["my card"], outputs=["action"])
def play_any_card(n):
    n.outputs["action"] = {"play": n.inputs["my card"]}


@kg.rule(name="Trump suit.", inputs=["contract"], outputs=["trump suit"])
def trump_suit(n):
    n.outputs["trump suit"] = n.inputs["contract"]["strain"]


@kg.rule(name="Double notrump.", inputs=["trump suit"], outputs=["action"])
def double_notrump(n):
    if n.inputs["trump suit"] == 4:  # "notrump":
        n.outputs["action"] = {"call": "double"}


@kg.rule(name="Trump length.", inputs=["trump suit", "my cards"], outputs=["trump length"])
def trump_length(n):
    t = n.inputs["trump suit"]
    if t is not None and t < 4:
        l = n.inputs["my cards"][t].sum()
    else: # no trump
        l = 0
    n.outputs["trump length"] = l


@kg.rule(name="Double or redouble with trump length.",
            inputs=["trump length"], outputs=["actions"])
def x_or_xx_with_trump(n):
    if n.inputs["trump length"] > 0:
        n.outputs["actions"] = [{ "call": "double"}, {"call": "redouble"}]


@kg.rule(name="Partner's contract.", inputs=["next_to_act", "contract"], outputs=["partner's contract"])
def partners_contract(n):
    my_seat = n.inputs["next_to_act"]["seat"]
    partners_seat = (my_seat + 2) % 4
    n.outputs["partner's contract"] = n.inputs["contract"]["seat"] == partners_seat


@kg.rule(name="Pass partner's doubled or redoubled contract.",
            inputs=["partner's contract", "doubled or redoubled"], outputs=["action"])
def pass_if_partner_was_doubled(n):
    if n.inputs["partner's contract"] and n.inputs["doubled or redoubled"]:
        n.outputs["action"] = {"call": "pass"}


@kg.rule(name="Bid my suit.", inputs=["my cards"], outputs=["action"])
def bid_my_suit(n):
    my_suit = n.inputs["my cards"].sum(axis=1).argmax()
    n.outputs["action"] = {"bid": {"level": 0, "strain": my_suit}}


@kg.rule(name="Pass.", inputs=[], outputs=["action"])
def otherwise_pass(n):
    n.outputs["action"] = {"call": "pass"}


@kg.rule(name="Optimal play.", inputs=["game", "actor state"], outputs=["actions"])
def optimal_play(n):
    game, gs = n.inputs['game'], n.inputs['actor state']
    legal_action_ids = game.possible_action_indices(gs)
    for dep in n.deps:
        if 'action' in dep.outputs:
            actions = [dep.outputs['action']]
        elif 'actions' in dep.outputs:
            actions = dep.outputs['actions']
        else:
            continue
        legal_attempted_actions = []
        for action in actions:
            a = game.Action(action)
            if int(a) in legal_action_ids:
                legal_attempted_actions.append(a)
        if legal_attempted_actions:
            n.outputs["actions"] = legal_attempted_actions
            return
    logging.error("Strategy did not attempt any legal actions")
    n.outputs["actions"] = []


_priorities = [
    "Play my card.",
    "Double notrump.",
    "Double or redouble with trump length.",
    "Pass partner's doubled or redoubled contract.",
    "Bid my suit.",
    "Pass.",
]


def build_kg(builder):
    b = builder
    b.add_rules('bridgebot.ncard.onecard.optimal_kg',
            ['Actor state.', 'Optimal play.'] + _priorities +
            ['My cards.', 'My card.', 'Trump suit.', 'Trump length.',
             'Doubled or redoubled.', "Partner's contract."])
    for p, pp in zip(_priorities, [None] + _priorities):
        b.add_relations([
            (p, 'requires', 'Actor state.'),
            ('Optimal play.', 'requires', p),
        ])
        if pp:
            b.add_relations([(p, 'has lower priority than', pp)])


if __name__ == '__main__':
    b = kg.Builder()
    build_kg(b)
    s = b.to_bytes()
    b2 = kg.Builder()
    b2.from_bytes(s)
    rkg = kg.link(b)
    print(rkg)
