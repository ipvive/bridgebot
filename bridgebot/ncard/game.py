"""Defines Ncard(N) game logic"""
import copy
import random

import numpy as np
from absl import logging

from bridgebot.bridge import fastgame

import pdb

from bridgebot.bridge.fastgame import Event

class Action(fastgame.Action):
    @classmethod
    def from_dict(cls, d):
        if 'play' in d:
            return Action.from_suit_and_rank_ids(d['play']['suit'], d['play']['rank'])
        elif 'call' in d:
            return Action.from_call(d['call'])
        elif 'bid' in d:
            return Action.from_level_and_strain_ids(d['bid']['level'], d['bid']['strain'])
        return cls(d)
    def chord(self):
        kind = self.kind()
        if kind == "bids":
            return [kind] + list(self.level_and_strain())
        elif kind == "calls":
            return [kind] + [self.call()]
        else:
            return [kind] + list(self.suit_and_rank())

class Game:
    def __init__(self, n=13):
        if n < 1 or n > 13:
            raise ValueError(n)
        self.num_ranks = n
        self.num_tricks = n
        self.book = n // 2
        self.num_levels = self.num_tricks - self.book

        self.levels = tuple(fastgame._levels.tokens[:self.num_levels])
        self.level_letters = tuple(str(i) for i in range(1, 1 + self.num_levels))
        self.strains = tuple(fastgame._strains.tokens)
        self.strain_letters = tuple(s[0].upper() for s in fastgame._strains.tokens)
        self.suits = tuple(fastgame._suits.tokens)
        self.suit_letters = tuple(s[0].upper() for s in fastgame._suits.tokens)
        self.seats = tuple(fastgame._seats.tokens)
        self.sides = ("North-South", "East-West")
        self.ranks = tuple(fastgame._ranks.tokens[13 - self.num_ranks:])
        self.rank_letters = [str(i + 2) if i + 2 < 10 else r[0]
                for i, r in enumerate(fastgame._ranks.tokens)]
        self.rank_letters = tuple(self.rank_letters[13 - self.num_ranks:])
        self.calls = tuple(fastgame._calls.tokens)
        self.call_letters = ('P', 'X', 'XX')
        self.positions = ('1st', '2nd', '3rd', '4th')
        self.stages = ('bidding', 'play', 'final', 'error')
        self.tricks = tuple(range(0, self.num_tricks + 1))
        self.trick_differences = tuple(range(
            0 - self.num_levels - self.book,  # bid self.num_levels + 1, take 0
            1 + self.num_tricks - self.book,  # inclusive: bid 1, take num_tricks
            ))
        self.str_trick_differences = fastgame._outcomes.tokens
        self.imps = tuple(range(-24, 25))
        self.winloss = (1, -1, 0)
        self.actions = (
                [f"{l}_{s}" for l in self.levels for s in self.strains] +
                ["[PAD]"] * (7 - self.num_levels) * len(self.strains) +
                list(self.calls))
        for s in self.suits:
            self.actions += (
                [f"{s}_{r}" for r in self.ranks] +
                ["[PAD]"] * (13 - self.num_ranks))

        self._impl = fastgame.Game(num_ranks=self.num_ranks)

    def Deal(self):
        return Deal(game=self, impl=self._impl.Deal(), num_ranks=self.num_ranks)

    def played_game_from_deal(self, deal: Deal):
        return self._impl.played_game_from_deal(deal._impl)

    def deal_from_played_game(self, played_game):
        deal = self.Deal()
        #deal.players = {k: v.player_name
        #        for k, v in played_game.player.items()}
        for seat, hand in played_game.board.dealt_cards.items():
            seat_id = self.seats.index(seat)
            for c in hand.cards:
                deal._impl._state.give_card(seat_id, c.suit_id, c.rank_id)
        deal.set_vulnerability(played_game.board.vulnerable_seat)
        deal.board_name = played_game.board.board_sequence_name
        deal.scoring = played_game.board.scoring
        deal = self.set_dealer(deal, played_game.board.dealer)
        annotation_index = 0

        self.execute_action_ids(deal, [
            action.action_id for action in played_game.actions])

        deal.table_name = played_game.table_name
        if len(played_game.result.summary_token) == 5:
            deal = self.set_result(deal, *played_game.result.summary_token)
        return deal
    
    def random_deal(self):
        deal = self.Deal()
        dealer = self.seats[random.randrange(4)]
        deal = self.set_dealer(deal, dealer)
        vulnerability_mask = random.randrange(4)
        vulnerability = []
        if vulnerability_mask & 1:
            vulnerability.extend(["North", "South"])
        if vulnerability_mask & 2:
            vulnerability.extend(["East", "West"])
        deal._impl.vulnerability = vulnerability
        deal._impl.players = {
                "South": "Rodwell",
                "West": "Platnick",
                "North": "Meckstroth",
                "East": "Diamond"}
        cards = [(suit, rank) for suit in range(4) for rank in range(self.num_ranks)]
        random.shuffle(cards)
        for i, card in enumerate(cards):
            self.give_card(deal, i % 4, card[0], card[1])
        return deal

    def Action(self, d):
        return Action.from_dict(d)

    def parse_action(self, data):
        if data in self.call_letters:
            return Action.from_call_id(self.call_letters.index(data))
        elif len(data) == 2:
            if (data[0] in self.level_letters and
                    data[1] in self.strain_letters):
                l = self.level_letters.index(data[0])
                s = self.strain_letters.index(data[1])
                return Action.from_level_and_strain_ids(l, s)
            if (data[0] in self.suit_letters and
                    data[1] in self.rank_letters):
                s = self.suit_letters.index(data[0])
                r = self.rank_letters.index(data[1])
                r += 13 - self.num_ranks
                return Action.from_suit_and_rank_ids(s, r)
            else:
                raise ValueError(f"invalid action {data}")
        else:
            raise ValueError(f"invalid action {data}")
#        elif isinstance(data, dict):
#            if 'play' in data:
#                s = data['play']['suit']
#                r = data['play']['rank']
#                r += 13 - game.num_ranks
#                self.id = 38 + 13 * s + r
#            elif 'bid' in data:
#                l = data['bid']['level']
#                s = data['bid']['strain']
#                self.id = 5 * l + s
#            elif 'call' in data:
#                self.id = 35 + game.calls.index(data['call'])
#            else:
#   .set_trace()

    def set_board_number(self, deal, *args, **kwargs):
        deal._impl = self._impl.set_board_number(deal._impl, *args, **kwargs)
        return deal

    def set_dealer(self, deal, seat):
        self._impl.set_dealer(deal._impl, seat)
        return deal
    
    def give_card(self, deal, seat, suit, rank):
        return self.give_cards(deal, [(seat, suit, rank)])

    def give_cards(self, deal, cards_list):
        card_ixs_list = []
        for seat, suit, rank in cards_list:
            if isinstance(seat, str):
                if seat in self.seats:
                    seat = self.seats.index(seat)
                else:
                    seat = self.seat_letters.index(seat)
            if isinstance(suit, str):
                if suit in self.suits:
                    suit = self.suits.index(suit)
                else:
                    suit = self.suit_letters.index(suit)
            if isinstance(rank, str):
                if rank in self.ranks:
                    rank = self.ranks.index(rank)
                else:
                    rank = self.rank_letters.index(rank)
            rank += 13 - self.num_ranks
            card_ixs_list.append((seat, suit, rank))
        deal._impl = self._impl.give_cards(deal._impl, card_ixs_list)
        # additional check is needed.
        if not deal.error and np.any(deal.dealt_cards.sum(axis=(1,2)) > self.num_ranks):
            self._impl._set_error(deal._impl, f"more than {self.num_ranks} cards in a hand")

        return deal

    def possible_action_indices(self, deal):
        impl_possible = self._impl.possible_action_indices(deal._impl)
        return [idx for idx in impl_possible if self.is_valid_action_idx(idx)]
    
    def execute_action_ids(self, deal, action_ids):
        for idx in action_ids:
            if not self.is_valid_action_idx(idx):
                pdb.set_trace()
        deal._impl = self._impl.execute_action_ids(deal._impl, action_ids)
        if deal._impl._state.tricks_taken.sum() == self.num_tricks:
            deal._impl._state.stage = deal._impl._state.STAGE_SCORING
            deal._impl._state.next_to_act = None
            self.compute_score(deal)
        return deal

    def compute_score(self, deal):
        return self._impl.compute_score(deal._impl)

    def table_score(self, result_event, vulnerability_list):
        return self._impl.table_score(result_event, vulnerability_list)

    def comparison_score(self, diff, scoring):
        return self._impl.comparison_score(diff, scoring)

    def is_valid_action_idx(self, idx):
        if idx < 35:
            level = idx // 5
            return level < self.num_levels
        elif idx < 38:
            return True
        else:
            rank = (idx - 38) % 13
            return rank >= 13 - self.num_ranks

    def parse_shorthand_problem(self, holding, bidding, play, board=1):
        deal = self.Deal()
        deal = self.set_board_number(deal, board)
        deal = _parse_bidding(self, deal, bidding)
        deal = _parse_play(self, deal, play)
        deal = _parse_cards(self, deal, holding)
        return deal

    def table_view(self, deal, action_count):
        view = self.Deal()
        view._impl = self._impl.table_view(deal._impl, action_count)
        return view

    def actor_view(self, deal, action_count):
        view = self.Deal()
        view._impl = self._impl.actor_view(deal._impl, action_count)
        return view

    def kibitzer_view(self, deal, action_count):
        view = self.Deal()
        view._impl = self._impl.kibitzer_view(deal._impl, action_count)
        return view

    def is_dealt_card(self, deal, suit, rank):
        if isinstance(suit, str):
            if suit in self.suits:
                suit = self.suits.index(suit)
            else:
                suit = self.suit_letters.index(suit)
        if isinstance(rank, str):
            if rank in self.ranks:
                rank = self.ranks.index(rank)
            elif rank in self.rank_letters:
                rank = self.rank_letters.index(rank)
            else:
                pdb.set_trace()
        return deal.dealt_cards[:, suit, rank].sum() > 0

    def score_played_board(self, b):
        self._impl.score_played_board(b)


class Deal:
    def __init__(self, game, impl, num_ranks):
        self.game = game
        self._impl = impl
        self._num_ranks = num_ranks

    def __str__(self):
        return f"Deal<{self._num_ranks}>({self._impl})"

    def copy_replay_state(self):
        new_deal = copy.copy(self)
        new_deal._impl = self._impl.copy_replay_state()
        return new_deal

    def num_actions(self):
        return self._impl.num_actions()
    
    def action_ix_history(self):
        return self._impl.action_ix_history()

    @property
    def error(self):
        return self._impl.error

    def is_final(self):
        return self._impl.is_final()

    def last_bid_level(self):
        return self._impl.last_bid_level()

    def last_bid_strain(self):
        return self._impl.last_bid_strain()

    def last_bid_seat(self):
        return self._impl.last_bid_seat()

    def last_double_as_call(self):
        return self._impl.last_double_as_call()

    def next_to_act(self):
        return self._impl.next_to_act()

    def next_to_act_index(self):
        return self._impl.next_to_act_index()

    @property
    def vulnerability(self):
        return self._impl.vulnerability

    def set_vulnerability(self, v):
        self._impl.vulnerability = v

    def pass_position(self):
        return self._impl.pass_position()

    def first_seat_of_partnership_to_mention_suit(self):
        return self._impl.first_seat_of_partnership_to_mention_suit()

    def contract_level(self):
        return self._impl.last_bid_level()

    def contract_strain(self):
        return self._impl.last_bid_strain()

    def contract_seat(self):
        return self._impl.last_bid_seat()

    def contract_doubled_as_call(self):
        return self._impl.last_double_as_call()

    def trick_counts(self):
        return self._impl.trick_counts()

    def trick_suit(self):
        return self._impl.trick_suit()

    def trick_winning_seat(self):
        return self._impl.trick_winning_seat()

    def trick_winning_suit(self):
        return self._impl.trick_winning_suit()

    def trick_winning_rank(self):
        return self._impl.trick_winning_rank()

    @property
    def dealt_cards(self):
        return self._impl.dealt_cards[:,:,13 - self._num_ranks:]

    @property
    def played_cards(self):
        return self._impl.played_cards[:,13 - self._num_ranks:]

    @property
    def no_cards(self):
        # TODO: fix this mess.
        return self._impl._state.max_length < 13

    @property
    def events(self):
        return self._impl.events

    @property
    def players(self):
        return self._impl.players

    @property
    def dealer(self):
        return self._impl.dealer()

    def action(self, n):
        return self._impl.action(n)

    @property
    def result(self):
        return self._impl.result


def _parse_cards(game, deal, cards):
    card_strings = cards.split(" ")
    seat = deal.next_to_act()
    cards = []
    for c in card_strings:
        s = game.suit_letters.index(c[0])
        r = game.rank_letters.index(c[1])
        cards.append((seat, game.suits[s], game.ranks[r]))
    deal = game.give_cards(deal, cards)
    return deal


def _parse_bidding(game, deal, bidding):
    rounds = bidding.split("; ")
    for i, r in enumerate(rounds):
        bids = r.split("-")
        if i < len(rounds) - 1:
            assert len(bids) == 4
        elif bids[-1] == '?':
            bids = bids[:-1]
        for bid in bids:
            action = game.parse_action(bid)
            deal = game.execute_action_ids(deal, [action])
    return deal

def _parse_play(game, deal, play):
    if not play:
        return deal
    for p in play.split(" "):
        if p == '?':
            continue
        action = game.parse_action(p)
        deal = game.execute_action_ids(deal, [action])
    return deal


class Tokenizer:
    def __init__(self, game):
        self.game = game
        self.all_tokens = ["[EOS]", "[PAD]", "[YES]", "[NO]",
                           "value_gt", "value_geq", "outcome",
                           "is vulnerable", "is dealer", "was dealt",
                           "bids", "calls", "plays", "doubled", "redoubled",
                           "passed out"] + \
                list(game.seats) + list(game.calls) + list(game.strains) + \
                list(game.levels) +  list(game.ranks) + list(game.suits) + \
                list(game.str_trick_differences)

    def tokenize_view(self, view: Deal) -> list[list]:
        chords = self.tokenize_board_info(view)
        chords.extend(self.tokenize_cards(view))
        chords.extend(self.tokenize_actions(view))
        return self.tokens_to_ids(chords)

    def tokenize_board_info(self, view: Deal) -> list[list]:
        chords = []
        for seat in view.vulnerability:
            chords.append([seat, "is vulnerable"])
        chords.append([view.dealer, "is dealer"])
        return chords

    def tokenize_cards(self, view:Deal) -> list[list]:
        chords = []
        for i, seat in enumerate(self.game.seats):
            for j, suit in enumerate(self.game.suits):
                for k, rank in enumerate(self.game.ranks):
                    if view.dealt_cards[i,j,k]:
                        chords.append([seat, "was dealt", suit, rank])
        return chords

    def tokenize_actions(self, view):
        chords = []
        for i in range(view.num_actions()):
            chords.append(view.action(i))
        return chords

    def tokenize_result(self, result):  # TODO: maybe append table score
        if result is None:
            pdb.set_trace()
        if result.cls == 'passed out':
            return [[result.cls]]
        else:
            utokens = [result.level, result.suit_or_strain, result.seat]
            if result.double != 'undoubled':
                utokens.append(result.double)
            utokens.append(result.trick_diff)
            return [utokens]

    def tokens_to_ids(self, chords):
        return [[self.all_tokens.index(t) for t in c] for c in chords]

    def ids_to_tokens(self, ids):
        tokens = []
        for v in ids:
            t = [self.all_tokens[ix] for ix in v if ix > 0]
            if t:
                tokens.append(t)
            else:
                break
        return tokens

    def tokenize_action_ids(self, action_ids):
        try:
            return self.tokens_to_ids([
                    Action(id_).chord() for id_ in action_ids])
        except ValueError:
            pdb.set_trace()
