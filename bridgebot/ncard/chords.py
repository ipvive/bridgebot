from functools import partial
import copy
from dataclasses import dataclass, field

import jax.numpy as jnp
import jax
import numpy as np

from bridgebot.ncard import game as bridgegame

import pdb

class Sequence(list):
    pass


class Choice(list):
    pass


def all_card_chords(game):
    return Sequence([Choice(game.suits), Choice(game.ranks)])


def all_bid_chords(game):
    return Sequence([Choice(game.levels), Choice(game.strains)])


def all_doubled_chords(game):
    return Choice([None, "doubled", "redoubled"])


def all_action_chords(game):
    return Sequence([Choice(game.seats), Choice([
        Sequence(["bids", all_bid_chords(game)]),
        Sequence(["calls", Choice(game.calls)]),
        Sequence(["plays", all_card_chords(game)])
        ])])


def all_outcome_chords(game):
    return Choice(["passed out", Sequence([
        all_bid_chords(game),
        Choice(game.seats),
        all_doubled_chords(game),
        Choice(game.str_trick_differences),
        ])])


def all_imp_difference_chords(game):
    return Choice([game.imps])


def all_matchpoint_chords(game):
    return Choice(game.winloss)


def all_bool_chords(game):
    return Choice(["[YES]", "[NO]"])


@dataclass(frozen=True)
class Transition:
    tokenid: int
    nodeid: int


@dataclass(frozen=True)
class Node:
    accept: bool
    transitions: Sequence[Transition]

@dataclass(frozen=True)
class Codec:
    # reject=0, post-accept=2, start = 3
    depth: int
    token: np.array  # dtype=str, dims=[token]
    token_index: dict[str, int]
    mask: np.array  # dtype=int32, dims=[state,token]
    transition: np.array  # dtype=int32, dims=[state,token]
    accept: np.array  # dtype=int32, dim=[state]

    def __hash__(self):
        return hash((self.depth, self.token, self.mask, self.transition,
                     self.accept))


# NOTE: this fn may not be needed.
def generate(logits, codec, return_ll=False):
    state = 2
    chord = []
    for i in range(codec.depth):
        l = logits - 20 * (1 - codec.mask[state])
        ll = jax.nn.log_softmax(l)
        m = ll.argmax()
        if return_ll:
            chord.append((codec.token[m], float(ll[m])))
        else:
            chord.append(codec.token[m])
        state = codec.transition[state, m]
    return chord, codec.accept[state]

def id_codec(codec, all_tokens):
    token = np.array([all_tokens.index(t) for t in codec.token])
    return Codec(
            depth=codec.depth,
            token=token,
            token_index={s: i for i, s in enumerate(token)},
            mask=codec.mask,
            transition=codec.transition,
            accept=codec.accept)

@partial(jax.jit, static_argnames=['YES', 'NO'])
def batch_bool_log_likelihood(logits, chords, YES=2, NO=3):
    flat_logits = logits.reshape((-1, logits.shape[-1]))
    flat_chords = chords.reshape((-1, chords.shape[-1]))
    labels = flat_chords[:,0] == YES
    diffs = flat_logits[:, YES] - flat_logits[:, NO]
    diffs = jnp.where(labels, diffs, -diffs)
    x = jnp.exp(diffs)
    ll = jnp.log(x / (1 + x))
    return jnp.reshape(ll, logits.shape[:-1])


@partial(jax.jit, static_argnames=['YES', 'NO'])
def batch_yes_log_likelihood(logits, YES=2, NO=3):
    flat_logits = logits.reshape((-1, logits.shape[-1]))
    diffs = flat_logits[:, YES] - flat_logits[:, NO]
    x = jnp.exp(diffs)
    ll = jnp.log(x / (1 + x))
    return jnp.reshape(ll, logits.shape[:-1])


def batch_log_likelihood(logits, codec, chord):
    if len(chord.shape) == 1:
        res = log_likelihood(logits, codec, tuple(chord.tolist()))
        if not res[1]:
            ll, ok, history = slow_log_likelihood(logits, codec, chord)
            if ok:
                print(f"log lokelihod failure: {chord}")
            else:
                print(f"slow log lokelihod failure: {codec}, {chord}, {history}")
                pdb.set_trace()
        return res[0]
    return jnp.array([batch_log_likelihood(logits[i], codec, chord[i])
                      for i in range(chord.shape[0])])

@partial(jax.jit, static_argnames=['codec', 'chord'])
def log_likelihood(logits, codec, chord):
    state = 2
    llsum = 0.
    for i, c in enumerate(chord):
        m = codec.token_index[c]
        # TODO: the following crashes if logits have diffferent dim than mask.
        # TODO: it will be icorrect if dims are permuted.
        try:
            l = logits - 20 * (1 - codec.mask[state])
        except:
            print("suspected N-card mismatch")
            pdb.set_trace()
        ll = jax.nn.log_softmax(l)
        llsum += ll[m]
        state = codec.transition[state, m]
    return llsum, codec.accept[state]


def slow_log_likelihood(logits, codec, chord):
    state = 2
    history = []
    llsum = 0.
    for i, c in enumerate(chord):
        m = codec.token_index[c]
        history.append((state, c, m,
                        codec.mask[state], codec.transition[state]))
        l = logits - 20 * (1 - codec.mask[state])
        ll = jax.nn.log_softmax(l)
        llsum += ll[m]
        state = codec.transition[state, m]
    return llsum, codec.accept[state], states


@dataclass
class MachineBuilder:
    tokens: list[object] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    def __init__(self, tokens=None):
        if tokens is None:
            tokens = ["[EOS]"]
        self.tokens = tokens
        self.nodes = []
        self.token_index = {s: i for i, s in enumerate(self.tokens)}
        self.node_index = {}

    def traverse(self, chords, after: int = None):
        """returns: nodeid"""
        if after is None:
            node = Node(accept=True, transitions=tuple())
            after = self.insert(node)
        if chords is None:
            return after 
        if isinstance(chords, int) or isinstance(chords, str):
            tokenid = self.token_index.get(chords, None)
            if tokenid is None:
                tokenid = self.token_index[chords] = len(self.tokens)
                self.tokens.append(chords)
            transitions = (Transition(tokenid=tokenid, nodeid=after),)
            node = Node(accept=False, transitions=transitions)
        elif isinstance(chords, Sequence):
            for child in reversed(chords):
                after = self.traverse(child, after)
            return after
        elif isinstance(chords, Choice):
            optionids = []
            for child in chords:
                optionids.append(self.traverse(child, after))
                transitions = []
                accept = False
                for optionid in optionids:
                    option = self.nodes[optionid]
                    transitions.extend(option.transitions)
                    if option.accept:
                        accept = True
                transitions = sorted(transitions, key=lambda t: t.tokenid)
                # TODO: check for duplicate transitions
                node = Node(accept=accept, transitions=tuple(transitions))
        return self.insert(node)

    def insert(self, node):
        nodeid = self.node_index.get(node, None)
        if nodeid is None:
            nodeid = self.node_index[node] = len(self.nodes)
            self.nodes.append(node)
        return nodeid


    def compile(self, start_nodeids: list[int]) -> list[Codec]:
        token_index = {s: i for i, s in enumerate(self.tokens)}
        return [self.compile1(ix, self.tokens, token_index)
                for ix in start_nodeids]

    def compile1(self, nodeid, token, token_index) -> Codec:
        mask, transition, accept = [], [], []
 
        # the first special state (id=0) is reject.
        reject_id = 0
        width = len(token)
        mask.append([1] * width)
        transition.append([reject_id] * width)
        accept.append(0)
 
        # the second special state (id=1) is post-accept.
        accept_id = 1
        mask.append([1] + [0] * (width - 1))
        transition.append([accept_id] + [reject_id] * (width - 1))
        accept.append(1)

        self.walk(mask, transition, accept, {},
                  nodeid, width, reject_id, accept_id)

        return Codec(depth=self.depth(set(), nodeid),
                     token=np.array([str(t) for t in token]),
                     token_index=token_index,
                     mask=np.array(mask),
                     transition=np.array(transition),
                     accept=np.array(accept))
     
    def walk(self, mask: list[list[int]],
             transition: list[list[int]],  # in/out
             accept: list[int],  # in/out  # in/out
             node_id_to_index: dict[int, int],  # in/out
             input_id: int,
             width: int, reject_id: int, accept_id: int) -> int:
        this_id = len(transition)
        if input_id in node_id_to_index:
            return node_id_to_index[input_id]
        else:
            node_id_to_index[input_id] = this_id
        node = self.nodes[input_id]
        mask.append([0] * width)
        transition.append([reject_id] * width)
        accept.append(node.accept)
        if node.accept:
            transition[this_id][0] = accept_id
            mask[this_id][0] = 1
        for t in node.transitions:
            that_id = self.walk(mask, transition, accept, node_id_to_index,
                                t.nodeid, width, reject_id, accept_id)
            transition[this_id][t.tokenid] = that_id
            mask[this_id][t.tokenid] = 1
        return this_id

    def depth(self, visited: set, input_id: int) -> int:
        if input_id in visited:
            return 0
        visited.add(input_id)
        node = self.nodes[input_id]
        return 1 + max([0] + [self.depth(visited, c.nodeid)
                              for c in node.transitions])

if __name__ == "__main__":
    game = bridgegame.Game(n=2)
    tokenizer = bridgegame.Tokenizer(game)
    rng = np.random.default_rng()
    l = np.zeros(5)
    l[1] = 3.
    l[2] = -1
    labels = np.array([2,0,0,0])
    ll2 = batch_bool_log_likelihood(l, labels, YES=1, NO=2)
