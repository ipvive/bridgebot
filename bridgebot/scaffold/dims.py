from collections import namedtuple
import numpy as np

from bridge import game as bridgegame

import pdb

Dim = namedtuple('Dim', ['name', 'labels'])
CategoricalDim = namedtuple('CategoricalDim', ['name', 'labels'])
RangeDim = namedtuple('RangeDim', ['name', 'labels', 'axis', 'max_value'])
CircleDim = namedtuple('CircleDim', ['name', 'labels', 'axis', 'period'])


dims = {
        'level': RangeDim('level', bridgegame._levels.tokens, axis=0, max_value=6),
        'strain': CircleDim('strain', bridgegame._strains.tokens, axis=1, period=5),
        'suit': CircleDim('suit', bridgegame._suits.tokens, axis=1, period=5), # reuse 1.
        'seat': CircleDim('seat', bridgegame._seats.tokens, axis=2, period=4),
        'call': CategoricalDim('call', bridgegame._calls.tokens),
        'position': CircleDim('position', ('1st', '2nd', '3rd', '4th'), axis=3, period=4),
        'rank': RangeDim('rank', bridgegame._ranks.tokens, axis=4, max_value=12),
        'stage': CategoricalDim('stage', ['bidding', 'play', 'final', 'error']),
        'tricks': RangeDim('tricks', list(range(14)), axis=5, max_value=13),
}


def _num_axes():
    num_axes = -1
    for d in dims.values():
        if isinstance(d, RangeDim) or isinstance(d, CircleDim):
            num_axes = max(num_axes, d.axis)
    return num_axes + 1


num_axes = _num_axes()


Observable = namedtuple('Observable', ['dims'])

class AbsoluteObservable(Observable):
    def parse(self, data):
        shape = tuple(len(d.labels) for d in self.dims)
        if data is not None:
            try:
                return np.array(data).reshape(shape)
            except:
                pdb.set_trace()
        else:
            return np.full(shape, np.nan)   

    def unparse(self, tensor):
        pass #TODO


class FuzzyObservable(Observable):
    def parse(self, data):
        pass #TODO

    def unparse(self, tensor):
        pass #TODO


class OneHotObservable(Observable):
    """Returns the positions of the observable, or None."""
    def parse(self, data):
        if data is not None:
            if not (isinstance(data, list) or isinstance(data, tuple)):
                data = (data,) 
            assert len(self.dims) == len(data)
            ixs = tuple(dim.labels.index(datum) if datum in dim.labels else datum
                    for dim, datum in zip(self.dims, data))
            return ixs
        else:
            return (None,) * len(self.dims)

    def unparse(self, ixs):
        assert len(self.dims) == len(ixs)
        return tuple(dim.labels[idx] if idx is not None else None for idx in ixs)


questions = {
        'next_bid': OneHotObservable((dims['level'], dims['strain'], dims['seat'])),
        'next_call': OneHotObservable((dims['call'], dims['seat'])),
        'next_card': OneHotObservable((dims['strain'], dims['rank'], dims['seat'])),
        'stage': OneHotObservable((dims['stage'],)),
        'last_bid': OneHotObservable(
            (dims['level'], dims['strain'], dims['seat'], dims['call'])),
        'next_to_act': OneHotObservable((dims['seat'],)),
        'vulnerability': AbsoluteObservable((dims['seat'],)),
        'pass_position': OneHotObservable((dims['position'],)),
        'first_mention': AbsoluteObservable((dims['strain'], dims['seat'])),
        'contract': OneHotObservable(
            (dims['level'], dims['strain'], dims['seat'], dims['call'])),
        'tricks_taken_ns': OneHotObservable((dims['tricks'],)),
        'tricks_taken_ew': OneHotObservable((dims['tricks'],)),
        'trick_suit': OneHotObservable((dims['suit'],)),
        'trick_winner': OneHotObservable(
            (dims['seat'], dims['suit'], dims['rank'])),
        'dealt_cards': AbsoluteObservable(
            (dims['seat'], dims['suit'], dims['rank'])),
        'played_cards': AbsoluteObservable((dims['suit'], dims['rank'])),
        #TODO 'no_cards': AbsoluteObservable((dims['seat'], dims['suit'])),
}


def extract_observables_from_action_id_and_seat(n, s):
    return {
        'next_bid': (n // 5, n % 5, s) if n < 35 else (None, None, None),
        'next_call': (n - 35, s) if n >= 35 and n < 38 else (None, None),
        'next_card': ((n - 38) // 13, (n - 38) % 13, s)
            if n >= 38 else (None, None, None),
    }


def extract_observables_from_deal(x, names=None):
    extractors = {
            'stage': lambda x: 'error' if x.error else (
                'final' if x.is_final() else (
                'play' if x.contract_level() else 'bidding')),
            'last_bid': lambda x: (x.last_bid_level(), x.last_bid_strain(),
                x.last_bid_seat(), x.last_double_as_call()),
            'next_to_act': lambda x: x.next_to_act(),
            'vulnerability': lambda x: tuple(s in x.vulnerability 
                                            for s in bridgegame._seats.tokens),
            'pass_position': lambda x: x.pass_position(),
            'first_mention': lambda x: x.first_seat_of_partnership_to_mention_suit(),
            'contract': lambda x: (x.contract_level(), x.contract_strain(),
                x.contract_seat(), x.contract_doubled_as_call()),
            'tricks_taken_ns': lambda x: x.trick_counts()['North'] + x.trick_counts()['South'],
            'tricks_taken_ew': lambda x: x.trick_counts()['East'] + x.trick_counts()['West'],
            'trick_suit': lambda x: x.trick_suit(),
            'trick_winner': lambda x: (x.trick_winning_seat(), x.trick_winning_suit(),
                x.trick_winning_rank(),), 
            'dealt_cards': lambda x: x.dealt_cards,
            'played_cards': lambda x: x.played_cards,
        #TODO 'no_cards': x.no_cards != -1,
    }
    if names is None:
        names = extractors.keys()
    data = {key: extractors[key](x) for key in names}
    return {key: questions[key].parse(datum) for key, datum in data.items()}

