from absl import app, flags
import copy
import functools
import numpy as np

from scaffold import dims
from bridge.fastgame import wrapper as bridgegame


#TODO make it possible to run FSA with different question names
question_names = ('last_bid', 'pass_position', 'next_to_act', 'stage')


def get_obervables(deal):
    d = dims.extract_observables_from_deal(deal, names=question_names)
    kept = tuple(d[n] for n in question_names)
    return kept


#TODO make this work with non-flattened obervables
def get_labels(obervables):
    labels = {}
    for n in question_names:
        desc = dims.questions[n]
        these_obervables = obervables[:len(desc.dims)]
        obervables = obervables[len(desc.dims):]
        for a, d in zip(these_obervables, desc.dims):
            labels[f"{n}.{d.name}"] = f"{d.labels[a] if a is not None else None}"
    return labels


def traverse(game):
    interior = frozenset()
    transitions = []
    boundary = {get_obervables(deal): deal
            for deal in game.distinct_boards()}
    while boundary:
        print(len(interior), len(boundary))
        interior = interior | frozenset(boundary.keys())
        new_boundary = {}
        for obervables, deal in boundary.items():
            for idx in range(len(bridgegame._actions.tokens)):
                new_deal = game.execute_action_index(deal.copy_replay_state(), idx)
                new_obervables = tuple(get_obervables(new_deal))
                if (new_obervables not in interior
                        and new_obervables not in new_boundary):
                    new_boundary[new_obervables] = new_deal
                transitions.append((obervables, idx, new_obervables))
        boundary = new_boundary
    return interior, transitions, len(bridgegame._actions.tokens)


def traverse_online(game):
    interior = frozenset()
    boundary = {get_obervables(deal): deal for deal in game.distinct_boards()}
    while boundary:
        print(len(interior), len(boundary))
        interior = interior | frozenset(boundary.keys())
        new_boundary = {}
        for obervables, deal in boundary.items():
            for idx in range(len(bridgegame._actions.tokens)):
                new_deal = game.execute_action_index(deal.copy_replay_state(), idx)
                new_obervables = tuple(get_obervables(new_deal))
                if (new_obervables not in interior
                        and new_obervables not in new_boundary):
                    new_boundary[new_obervables] = new_deal
                yield deal, obervables, idx, new_obervables
        boundary = new_boundary


class FSA:
    def __init__(self, interior, transitions, num_actions):
        self.state_labels = tuple(interior)
        self.state_index = {v: i for i, v in enumerate(self.state_labels)}
        self.transition_table = np.zeros((len(self.state_labels), num_actions), dtype=np.int)
        for s, a, e in transitions:
            self.transition_table[self.state_index[s], a] = self.state_index[e]

    def apply(self, state, idx):
        return self.state_labels[int(self.transition_table[
            self.state_index[tuple(state)], idx])]


def main(argv):
    interior,  transitions, num_actions = traverse(bridgegame.Game())
    fsa = FSA(interior, transitions, num_actions)    


if __name__ == "__main__":
    app.run(main)
