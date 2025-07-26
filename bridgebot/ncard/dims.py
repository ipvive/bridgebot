from collections import namedtuple
import numpy as np

import bridgebot.ncard.game as ncardgame 
from bridgebot.ncard.rgeom import Dim, Axis, OneHotObservable, AbsoluteObservable

import pdb


def get_dims(game):
    return {
        'level': Dim('level', game.levels),
        'strain': Dim('strain', game.strains),
        'suit': Dim('suit', game.suits),
        'seat': Dim('seat', game.seats),
        'side': Dim('side', game.sides),
        'position': Dim('position', game.positions),
        'rank': Dim('rank', game.ranks),
        'call': Dim('call', game.calls),
        'stage': Dim('stage', game.stages), 
        'tricks': Dim('tricks', game.tricks), # tuple(range(14))
        'imps': Dim('imps', game.imps), # tuple(range(-24, 25)))
        'winloss': Dim('winloss', game.winloss), # ('win', 'tie', 'lose')
    }


def get_axes(game):
    result = {}
    result['level'] = Axis(index=0, size=len(game.levels), embedding_type='range')
    result['strain'] = Axis(index=1, size=len(game.strains), embedding_type='circle')
    result['suit'] = result['strain']
    result['seat'] = Axis(index=2, size=len(game.seats), embedding_type='circle')
    result['side'] = result['seat']
    result['position'] = Axis(index=3, size=len(game.positions), embedding_type='circle')
    result['rank'] = Axis(index=4, size=len(game.ranks), embedding_type='range')
    result['call'] = Axis(index=None, size=len(game.calls), embedding_type='categorical')
    result['stage'] = Axis(index=None, size=len(game.stages), embedding_type='categorical')
    result['tricks'] = Axis(index=None, size=len(game.tricks), embedding_type='categorical')
    result['imps'] = Axis(index=None, size=len(game.imps), embedding_type='categorical')
    result['winloss'] = Axis(index=None, size=len(game.winloss), embedding_type='categorical')
    return result


def get_questions(game):
    dims = get_dims(game)
    return {
        # actions
        'next_bid': OneHotObservable((dims['level'], dims['strain'], dims['seat'])),
        'next_call': OneHotObservable((dims['call'], dims['seat'])),
        'next_card': OneHotObservable((dims['strain'], dims['rank'], dims['seat'])),
         # value
        'imps': OneHotObservable((dims['imps'],)),
        'positive_score': OneHotObservable((dims['winloss'],)),
        'better_score': OneHotObservable((dims['winloss'],)),
         # game state
        'next_to_act': OneHotObservable((dims['seat'],)),
        'stage': OneHotObservable((dims['stage'],)),
        'vulnerability': AbsoluteObservable((dims['side'],)),
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
        'no_cards': AbsoluteObservable((dims['seat'], dims['suit'])),
}


def extract_action_from_action_id_and_seat(game, n, s, names=None):
    n_strains = len(game.strains)
    n_levels = 7
    n_calls = len(game.calls)
    n_ranks = 13
    n_bids = n_strains * n_levels
    n_play = n_bids + n_calls
    data = {
        'next_bid': (n // n_strains, n % n_strains, s) 
            if n < n_bids else (None, None, None),
        'next_call': (n - n_bids, s) if n >= n_bids and n < n_play else (None, None),
        'next_card': ((n - n_play) // n_ranks, (n - n_play) % n_ranks, s)
            if n >= n_play else (None, None, None),
    }
    if names is None:
        names = data.keys()
    questions = get_questions(game)
    return {key: questions[key].parse(data[key]) for key in names}


def extract_scores_from_final_states(game, state, other_state, names=None):
    game.compute_score(state)
    game.compute_score(other_state)
    table_score = game.table_score(state._impl.result,
            state._impl.vulnerability)
    other_table_score = game.table_score(other_state._impl.result,
            other_state._impl.vulnerability)
    table_score = table_score[0] or -table_score[1]
    other_table_score = other_table_score[0] or -other_table_score[1]
    imps = game.comparison_score(table_score - other_table_score, 'IMPs')
    if imps[0] is not None:
        imps = imps[0]
    else:
        imps = -imps[1]

    def _side_of_zero(z):
        if z > 0: return 1
        if z == 0: return 0
        if z < 0:  return -1

    scores = {
        'imps': imps,  
        'positive_score': _side_of_zero(table_score),
        'better_score': _side_of_zero(table_score - other_table_score),
    }
    if names is None:
        names = scores.keys()
    questions = get_questions(game)
    return {key: questions[key].parse(scores[key]) for key in names}


def extract_state_from_deal(game, x, names=None):
    extractors = {
            'stage': lambda x: 'error' if x.error else (
                'final' if x.is_final() else (
                'play' if x.contract_level() else 'bidding')),
            'next_to_act': lambda x: x.next_to_act(),
            'vulnerability': lambda x: tuple(s in x.vulnerability for s in game.seats[:2]),
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
            'no_cards': lambda x: x.no_cards,
    }
    if names is None:
        names = extractors.keys()
    data = {key: extractors[key](x) for key in names}
    questions = get_questions(game)
    return {key: questions[key].parse(datum) for key, datum in data.items()}

