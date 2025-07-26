"""Director ("referee") role for bridge robot as a finite state machine."""
import copy
import logging
import numpy as np

from bridgebot.bridge import players
from bridgebot.pb import alphabridge_pb2

import pdb

class Game(object):
    def Deal(self):
        return Deal()

    def distinct_boards(self):
        return [self.set_board_number(Deal(), n) for n in range(1, 17)]

    def set_board_number(self, deal, n):
        deal = self.set_dealer(deal,
            ["North", "East", "South", "West"][(n - 1) % 4])
        deal.vulnerability = [
            [],
            ["North", "South"],
            ["East", "West"],
            ["North", "South", "East", "West"]
        ][((n - 1) + (n - 1) // 4) % 4]
        return deal

    def random_deal(self, rng):
        deal = Deal()
        print(deal.players)
        dealer = _seats.tokens[rng.randrange(4)]
        deal = self.set_dealer(deal, dealer)
        vulnerability_mask = rng.randrange(4)
        deal.vulnerability = []
        deal.scoring = "IMPs"
        if vulnerability_mask & 1:
            deal.vulnerability.extend(["North", "South"])
        if vulnerability_mask & 2:
            deal.vulnerability.extend(["East", "West"])
        deal.players = {
                "South": "Rodwell",
                "West": "Platnick",
                "North": "Meckstroth",
                "East": "Diamond"}
        cards = [(suit, rank) for suit in range(4) for rank in range(13)]
        rng.shuffle(cards)
        for i, card in enumerate(cards):
            self._give_card(deal, i % 4, card[0], card[1])
        return deal

    def deal_from_played_game(self, played_game):
        deal = self.Deal()
        deal.players = {k: v.player_name
                for k, v in played_game.player.items()}
        for seat, hand in played_game.board.dealt_cards.items():
            for t in hand.card_token:
                suit, rank = t.split("_")
                deal = self.give_card(deal, seat, suit, rank)
        deal.vulnerability = played_game.board.vulnerable_seat
        deal.board_name = played_game.board.board_sequence_name
        deal.scoring = played_game.board.scoring
        deal = self.set_dealer(deal, played_game.board.dealer)
        annotation_index = 0

        def add_commentary(deal, ann):
            if ann.explanation:
                return self.add_explanation(deal, ann.explanation)
            if ann.kibitzer_comment:
                return self.add_commentary(deal, ann.kibitzer_comment)

        for i, action in enumerate(played_game.actions):
            while (annotation_index < len(played_game.annotations) and
                played_game.annotations[annotation_index].action_index <= i):
                deal = add_commentary(deal, played_game.annotations[annotation_index])
                annotation_index += 1

            if action.token in _calls.tokens:
                deal = self.make_call(deal, action.token)
            elif action.token in _bids.tokens:
                deal = self.make_bid(deal, *action.token.split("_"))
            elif action.token in _cards.tokens:
                deal = self.play_card(deal, *action.token.split("_"))

        while annotation_index < len(played_game.annotations):
            deal = add_commentary(deal, played_game.annotations[annotation_index])
            annotation_index += 1

        deal.table_name = played_game.table_name
        if len(played_game.result.summary_token) == 5:
            deal = self.set_result(deal, *played_game.result.summary_token)
        return deal

    def played_game_from_deal(self, deal):
        """Doesn't set result.comparison_score."""
        player_ids = {k: alphabridge_pb2.PlayerId(player_name=v)
                for k, v in deal.players.items()}
        dealt_cards = {"South": [], "West": [], "North": [], "East": []}
        for i, seat in enumerate(_seats.tokens):
            for j, suit in enumerate(_suits.tokens):
                for k, rank in enumerate(_ranks.tokens):
                    if deal.dealt_cards[i, j, k]:
                        dealt_cards[seat].append("{}_{}".format(suit, rank))
        board = alphabridge_pb2.Board(
                vulnerable_seat=deal.vulnerability,
                board_sequence_name=deal.board_name,
                scoring=deal.scoring,
                dealer=deal.dealer(),
                dealt_cards={k: alphabridge_pb2.Hand(card_token=v)
                    for k, v in dealt_cards.items()})
        actions = []
        annotations = []
        for event in deal.events:
            if event.is_bid() or event.is_play():
                actions.append(alphabridge_pb2.Action(
                        token="{}_{}".format(event.tokens[2], event.tokens[3])))
            elif event.is_call():
                actions.append(alphabridge_pb2.Action(
                        token=event.tokens[2]))
            if event.explanation:
                annotations.append(alphabridge_pb2.Annotation(
                    action_index=len(actions), explanation=event.explanation))
            for comment in event.commentary:
                annotations.append(alphabridge_pb2.Annotation(
                    action_index=len(actions), kibitzer_comment=comment))

        if deal.result:
            result = alphabridge_pb2.Result(summary_token=deal.result.tokens)
        else:
            result = alphabridge_pb2.Result()
        played_game = alphabridge_pb2.PlayedGame(
                player=player_ids,
                board=board,
                actions=actions,
                annotations=annotations,
                result=result,
                table_name=deal.table_name)
        return played_game

    def set_dealer(self, deal, seat):
        if deal.events:
            deal.error = "dealer already set"
        else:
            seat_ix = _seats.index[seat]
            deal.events = [_make_deal_event(seat_ix)]
        return deal

    def set_players(self, deal, south, west, north, east):
        deal.players = {
                "South": south.lower(),
                "West": west.lower(),
                "North": north.lower(),
                "East": east.lower()}
        return deal

    def give_card(self, deal, seat, suit, rank):
        seat_ix = _seats.index[seat]
        suit_ix = _suits.index[suit]
        rank_ix = _ranks.index[rank]
        return self._give_card(deal, seat_ix, suit_ix, rank_ix)

    def _give_card(self, deal, seat_ix, suit_ix, rank_ix):
        if deal.error:
            return deal
        if deal.dealt_cards[:,suit_ix,rank_ix].sum() > 0:
            deal.error = "Duplicate card"
        elif deal.played_cards[suit_ix, rank_ix] > 0:
            deal.error = "Card already played"
        elif deal.dealt_cards[seat_ix,:,:].sum() >= 13:
            deal.error = "14 cards in hand"
        elif deal.no_cards[seat_ix,suit_ix] != -1:
            deal.error = "Revoke"
        else:
            deal.dealt_cards[seat_ix,suit_ix,rank_ix] = 1
        return deal

    def make_bid(self, deal, level, strain):
        if deal.error:
            return deal
        if not deal.next_to_act():
            deal.error = "Deal finished"
            return deal
        seat_ix = _seats.index[deal.next_to_act()]
        level_ix = _levels.index[level]
        strain_ix = _strains.index[strain]
        if deal.contract_level():
            deal.error = "Bidding finished"
            return deal
        last_bid = deal.last_bid()
        if last_bid:
            if level_ix < _levels.index[last_bid.level()] or (
                    level == last_bid.level() and
                    strain_ix <= _strains.index[last_bid.strain()]):
                deal.error = "Insufficient bid"
                return deal
        deal.events.append(_make_bid_event(seat_ix, level_ix, strain_ix))
        partner_seat_ix = (seat_ix + 2) % 4
        if not deal.first_mention[partner_seat_ix, strain_ix]:
            deal.first_mention[seat_ix, strain_ix] = 1
        return deal

    def make_call(self, deal, call):
        if deal.contract_index != -1:
            deal.error = "Call after bidding finished"
        if deal.error:
            return deal
        bid_event, double_event, pass_count = None, None, 0
        for i in range(len(deal.events)-1, 0, -1):
            event = deal.events[i]
            if event.is_bid():
                bid_event = event
                break
            elif event.is_pass():
                if not double_event:
                    pass_count += 1
            elif event.is_redouble() or event.is_double():
                if not double_event:
                    double_event = event
            else:
                deal.error = "pass after bidding has ended"
                return deal

        seat = deal.next_to_act()
        if not seat:
            deal.error = "dealer not set, or deal finished"
            return deal
        else:
            seat_ix = _seats.index[seat]

        if call == "pass":
            if pass_count == 3:
                deal.events.append(_make_call_event(seat_ix, call))
                deal.events.append(_make_passed_out_event())
                deal._is_final = True
                return deal
            elif bid_event and pass_count == 2:
                deal.events.append(_make_call_event(seat_ix, call))
                seat_ix = _seats.index[bid_event.seat()]
                strain_ix = _strains.index[bid_event.strain()]
                if deal.first_mention[seat_ix,strain_ix]:
                    seat = _seats.tokens[seat_ix]
                else:
                    partner_seat_ix = (seat_ix + 2) % 4
                    seat = _seats.tokens[partner_seat_ix]
                deal.events.append(
                        _make_contract_event(seat, bid_event, double_event))
                deal.contract_index = len(deal.events) - 1
                return deal
        elif call == "double":
            if not bid_event:
                deal.error = "double before first bid"
            elif double_event:
                deal.error = "contract already doubled"
            else:
                bid_seat_ix = _seats.index[bid_event.seat()]
                if bid_seat_ix % 2 == seat_ix % 2:
                    deal.error = "double of own sides' contract"
        elif call == "redouble":
            if not bid_event:
                deal.error = "redouble before first bid"
            elif not double_event:
                deal.error = "redouble of non-doubled contract"
            elif double_event.is_redouble():
                deal.error = "redouble of redoubled contract"
            else:
                bid_seat_ix = _seats.index[bid_event.seat()]
                if bid_seat_ix % 2 != seat_ix % 2:
                    deal.error = "double of other sides' contract"
        else:
            deal.error = "unrecognized call"
        if deal.error:
            return deal
        deal.events.append(_make_call_event(seat_ix, call))
        return deal

    def play_card(self, deal, suit, rank):
        if deal.error:
            return deal
        if not deal.next_to_act():
            deal.error = "Card played, but dealer not set."
            return deal
        seat_ix = _seats.index[deal.next_to_act()]
        suit_ix = _suits.index[suit]
        rank_ix = _ranks.index[rank]
        if deal.contract_index == -1:
            deal.error = "Card played before bidding finished"
        elif deal.played_cards[suit_ix, rank_ix]:
            deal.error = "Card already played"
        elif not deal.dealt_cards[seat_ix, suit_ix, rank_ix]:
            deal = self._give_card(deal, seat_ix, suit_ix, rank_ix)
        if deal.error:
            return deal
        lead_event = deal.last_lead()
        if lead_event and suit != lead_event.suit():
            lead_suit_ix = _suits.index[lead_event.suit()]
            remaining = deal.dealt_cards[seat_ix,lead_suit_ix,:] & ~deal.played_cards[lead_suit_ix]
            if remaining.sum() > 0:
                deal.error = "Revoke"
                return deal
            deal.no_cards[seat_ix, lead_suit_ix] = len(deal.events)
        deal.played_cards[suit_ix, rank_ix] = 1
        deal.events.append(_make_play_event(seat_ix, suit_ix, rank_ix))
        self._maybe_take_trick(deal)
        return deal

    def _maybe_take_trick(self, deal):
        if (len(deal.events) - deal.contract_index) % 5 != 0:
            return
        winning_event = deal.trick_winner()
        deal.events.append(_make_trick_event(winning_event.seat()))
        if deal.played_cards.sum() == 52:
            self._finalize(deal)

    def _finalize(self, deal):
        deal._is_final = True
        counts = deal.trick_counts()
        declarer = deal.contract_seat()
        declarer_ix = _seats.index[declarer]
        total_tricks = 0
        for seat, n in counts.items():
            if _seats.index[seat] % 2 == declarer_ix % 2:
                total_tricks += n
        self._accept_claim(deal, total_tricks)


    def add_explanation(self, deal, explanation):
        if len(deal.events) == 0:
            deal.error = "explanation with no events"
            return deal
        if deal.events[-1].explanation is not None:
            deal.error = "explanation already set"
            return deal
        deal.events[-1].explanation = explanation
        return deal

    def add_commentary(self, deal, comment):
        if len(deal.events) == 0:
            deal.error = "comment with no events"
            return
        deal.events[-1].commentary.append(comment)
        return deal

    def set_result(self, deal, level, strain, player, double, outcome):
        """ Sets deal.result."""
        if (level not in _levels.index or
            strain not in _strains.index or
            player not in _seats.index or
            double not in _extra.index or
            outcome not in _outcomes.index):
            level, strain, player, double, outcome = ["passed_out"]*5

        deal.result = Event([level, strain, player, double, outcome])
        return deal

    def accept_claim(self, deal, total_tricks):
        self._accept_claim(deal, total_tricks)
        return deal

    def _accept_claim(self, deal, total_tricks):
        if not deal.contract_level():
            deal.error = "claim before bidding finished"
            return
        tricks_contracted = int(deal.contract_level()) + 6
        diff = int(total_tricks) - tricks_contracted
        if diff == 0:
            outcome = "="
        elif diff < 0:
            outcome = str(diff)
        else:
            outcome = "+" + str(diff)
        result = Event([
            deal.contract_level(),
            deal.contract_strain(),
            deal.contract_seat(),
            deal.contract_doubled(),
            outcome])

        if deal.result:
            if deal.result.tokens != result.tokens:
                deal.error = "claim/result mismatch"
        else:
            deal.result = result
        deal._is_final = True

    def execute_action(self, deal, action):
        if action.is_bid():
            return self.make_bid(deal, action.level(), action.strain())
        elif action.is_call():
            return self.make_call(deal, action.tokens[2])
        elif action.is_play():
            return self.play_card(deal, action.suit(), action.rank())

    def execute_action_index(self, deal, index):
        token = _actions.rindex[index]
        if token in _cards.index:
            return self.play_card(deal, *token.split("_"))
        elif token in _bids.index:
            return self.make_bid(deal, *token.split("_"))
        else:
            return self.make_call(deal, token)


    def possible_actions(self, deal):
        actions = []
        n = len(deal.events)
        actor_ix = _seats.index[deal.next_to_act()]
        if not deal.contract_level():
            for call in _calls.tokens:
                hdeal = self.make_call(deal.copy_replay_state(), call)
                if not hdeal.error:
                    actions.append(hdeal.events[n])
            higher = deal.last_bid()
            for i, level in enumerate(_levels.tokens):
                for j, strain in enumerate(_strains.tokens):
                    if not higher:
                        actions.append(_make_bid_event(actor_ix, i, j))
                    elif level == higher.level() and strain == higher.strain():
                        higher = None
        else:
            if deal.dealt_cards[actor_ix, :, :].sum() == 13:
                cards_left = (deal.dealt_cards[actor_ix, :, :] & ~deal.played_cards[:, :])
                follow_suit_ix = None
                lead_event = deal.last_lead()
                if lead_event:
                    follow_suit_ix = _suits.index[lead_event.suit()]
                if follow_suit_ix is not None and cards_left[follow_suit_ix].sum() == 0:
                    follow_suit_ix = None
                for suit_ix in range(4):
                    if follow_suit_ix is not None and suit_ix != follow_suit_ix:
                        continue
                    for rank_ix in range(13):
                        if cards_left[suit_ix][rank_ix]:
                            actions.append(_make_play_event(actor_ix, suit_ix, rank_ix))
            else:
                for suit in _suits.tokens:
                    for rank in _ranks.tokens:
                        hdeal = self.play_card(deal.copy_replay_state(), suit, rank)
                        if not hdeal.error:
                            actions.append(hdeal.events[n])
        return actions

    def possible_action_indices(self, deal):
        possible_events = self.possible_actions(deal)
        possible_tokens = []
        for ev in possible_events:
            if ev.is_bid() or ev.is_play():
                possible_tokens.append("{}_{}".format(ev.tokens[2], ev.tokens[3]))
            elif ev.is_call():
                possible_tokens.append(ev.tokens[2])
        return [_actions.index[t] for t in possible_tokens]

    def kibitzer_view(self, deal, action_index):
        view = self._replay(deal, action_index)
        view.dealt_cards = np.copy(deal.dealt_cards)
        return view

    def table_view(self, deal, action_index):
        view = self._replay(deal, action_index)
        return view

    def actor_view(self, deal, action_index):
        view = self._replay(deal, action_index)
        actor = view.next_to_act()
        if actor is None:
            view.dealt_cards = np.copy(deal.dealt_cards)
            return view
        actor_ix = _seats.index[actor]
        if view.contract_index != -1:
            declarer = view.events[view.contract_index].seat()
            declarer_ix = _seats.index[declarer]
            if actor_ix % 2 == declarer_ix % 2:
                actor_ix = declarer_ix
        view.dealt_cards[actor_ix,:,:] = deal.dealt_cards[actor_ix,:,:]
        return view

    def _replay(self, deal, action_index):
        """Creates a public-information view, replaying each call and play."""
        # TODO(njt): move this function to _test, before optimizing.
        view = self.Deal()
        view.board_name = deal.board_name
        view.table_name = deal.table_name
        view.players = deal.players
        view.vulnerability = deal.vulnerability
        view.scoring = deal.scoring
        view.result = deal.result
        view = self.set_dealer(view, deal.dealer())
        num_actions = 0
        dummy_is_shown = False
        for event in deal.events[1:]:
            if num_actions >= action_index:
                return view
            if event.is_bid():
                view = self.make_bid(view, event.level(), event.strain())
                num_actions += 1
            elif event.is_call():
                view = self.make_call(view, event.tokens[2])
                num_actions += 1
            elif event.is_play():
                if not dummy_is_shown:
                    dummy_is_shown = True
                    declarer = view.events[view.contract_index].seat()
                    dummy_ix = (_seats.index[declarer] + 2) % 4
                    view.dealt_cards[dummy_ix,:,:] = deal.dealt_cards[dummy_ix,:,:]
                view = self.play_card(view, event.suit(), event.rank())
                num_actions += 1
        return view

    def table_score(self, result_event, vulnerability_list):
        """table_score computes the score at a table.

        Bridge score explained by `https://www.acbl.org/learn_page/how-to-play-bridge/how-to-keep-score/duplicate/`

        Args:
          result_event: the result of play, from deal.result.
          vulnerability_list: either [], ["North", "South"], ...

        Returns:
          (score for North-South, score for East-West).
        """
        (level, strain, seat, double, outcome) = result_event.tokens
        if level == "passed_out":
            return (0, 0)
        vulnerable = seat in vulnerability_list
        declarer_score, defender_score = (None, None)
        if outcome == "=":
            trick_diff = 0
        else:
            trick_diff = int(outcome)

        if trick_diff < 0:
            if double == "undoubled":
                if vulnerable:
                    defender_score = 100 * abs(trick_diff)
                else:
                    defender_score = 50 * abs(trick_diff)
            else:
                if vulnerable:
                    if trick_diff >= -3:
                        defender_score = (200, 500, 800)[abs(trick_diff) - 1]
                    else:
                        defender_score = 800 + 300 * (abs(trick_diff) - 3)
                else:
                    if trick_diff >= -3:
                        defender_score = (100, 300, 500)[abs(trick_diff) - 1]
                    else:
                        defender_score = 500 + 300 * (abs(trick_diff) - 3)
                if double == "redoubled":
                    defender_score *= 2
        else:
            if strain == "notrump":
                below_line_score = 40 + 30 * (int(level) - 1)
                above_line_score = 30 * trick_diff
            elif strain in ["Spades", "Hearts"]:
                below_line_score = 30 * (int(level))
                above_line_score = 30 * trick_diff
            elif strain in ["Diamonds", "Clubs"]:
                below_line_score = 20 * (int(level))
                above_line_score = 20 * trick_diff

            if double != "undoubled":
                below_line_score *= 2
                if vulnerable:
                    above_line_score = 50 + 200 * trick_diff
                else:
                    above_line_score = 50 + 100 * trick_diff
                if double == "redoubled":
                    below_line_score *= 2
                    above_line_score *= 2

            if below_line_score >= 100:
                if vulnerable:
                    bonus = 500
                else:
                    bonus = 300
            else:
                bonus = 50

            if level == "6":
                if vulnerable:
                    bonus += 750
                else:
                    bonus += 500
            elif level == "7":
                if vulnerable:
                    bonus += 1500
                else:
                    bonus += 1000

            declarer_score = below_line_score + above_line_score + bonus

        if seat in ["North", "South"]:
            return declarer_score, defender_score
        else:
            return defender_score, declarer_score

    def comparison_score(self, diff, scoring):
        if scoring == "Matchpoints":
            if diff > 0:
                return 1, None
            elif diff < 0:
                return -1, None
            else:
                return 0, None
        elif scoring == "total_points":
            return diff, None
        elif scoring == "IMPs":
            for i, cutoff in enumerate(_IMP_table):
                if abs(diff) < cutoff:
                    if diff > 0:
                        return i, None
                    else:
                        return -i, None
            return None, "score diff impossibly large"
        else:
            return None, "unknown scoring"

    def score_played_board(self, played_board):
        for game in played_board.tables:
            if game.result.summary_token:
                scores = self.table_score(Event(game.result.summary_token),
                        game.board.vulnerable_seat)
                game.result.table_score = scores[0] if scores[0] else -scores[1]
            else:
                game.result.table_score = 1 # No score

        for game in played_board.tables:
            comparison_score = 0
            for other_game in played_board.tables:
                if other_game != game and \
                        game.result.table_score != 1 and \
                        other_game.result.table_score != 1:
                    score, err = self.comparison_score(
                            game.result.table_score - other_game.result.table_score,
                            game.board.scoring)
                    comparison_score += score
            game.result.comparison_score = comparison_score

    def same_side(self, a, b):
        d = {"North": 0, "South": 0, "East": 1, "West": 1}
        return d[a] == d[b]

    def same_side_for_index(self, ia, ib):
        return (ia - ib) % 2 == 0

    def all_same_side_for_index(self, ia, ib, ic):
        return (all(x is not None for x in [ia, ib, ic]) and
                (ia - ib) % 2 == 0 and (ia - ic) % 2 == 0)


def dealcards(m):
    return " ".join(f"{p}: {suitcards(m[k])}" for k, p in enumerate("SWNE"))


def suitcards(mat):
    s = ""
    for i, n in enumerate("CDHS"):
        s += n
        for j, m in enumerate("23456789TJQKA"):
            if mat[i][j]:
                s += m
        s += " "
    return s


class Deal(object):
    """Represents a position in the bidding and play of the cards."""
    def __init__(self):
        self.board_name = None
        self.table_name =  None
        self.players =  None
        self.vulnerability =  None
        self.scoring =  None

        self.dealt_cards =  np.zeros((4,4,13), np.int8)  # seat, suit, rank. 1=has.
        self.played_cards = np.zeros((4, 13), np.int8)  # suit, rank. 1=played.
        self.events = []
        self.no_cards = np.full((4,4), -1, np.int16)  # seat, suit. n=event index, -1=None.
        self.first_mention = np.zeros((4,5), np.int8)  # seat, strain. 1=first
        self.contract_index = -1
        self._is_final = False

        self.result = None
        self.error = None

    def __repr__(self):
        buf = ""
        buf += "board {}{}\n".format(self.table_name, self.board_name)
        buf += "players {}\n".format(self.players)
        buf += "vulnerability {}\n".format(self.vulnerability)
        buf += "scoring {}\n".format(self.scoring)
        buf += "dealt_cards: {}\n".format(dealcards(self.dealt_cards))
        buf += "played_cards: {}\n".format(suitcards(self.played_cards))
        buf += "no_cards:\n{}\n".format(self.no_cards)
        buf += "first_mention:\n{}\n".format(self.first_mention)
        buf += "events:\n"
        for event in self.events:
            buf += "  {}\n".format(event)
        return buf

    def __str__(self):
        return self.__repr__()

    def copy_replay_state(self):
        return copy.deepcopy(self)

    def has_error(self):
        return self.error is not None

    def next_to_act(self):
        if not self.events:
            return None
        else:
            event = self.events[-1]
            if event.is_call() or event.is_play() or event.is_contract():
                seat_ix = _seats.index[event.seat()]
                next_seat_ix = (seat_ix + 1) % 4
                return _seats.rindex[next_seat_ix]
            else:
                return self.events[-1].seat()

    def next_to_act_index(self):
        return _seats.index[self.next_to_act()]

    def next_action_verb_index(self):
        bid_or_play = "bids" if self.contract_index == -1 else "plays"
        token = "{}_{}".format(self.next_to_act(), bid_or_play)
        return _action_verbs.index[token]

    def _reversed_bidding_events(self):
        if self.contract_index == -1:
            return reversed(self.events)
        else:
            return reversed(self.events[:self.contract_index])
    
    def last_bid(self):
        for event in self._reversed_bidding_events():
            if event.is_bid():
                return event
        return None

    def last_bid_level(self):
        b = self.last_bid()
        if b:
            return b.level()
        else:
            return None

    def last_bid_strain(self):
        b = self.last_bid()
        if b:
            return b.strain()
        else:
            return None

    def last_bid_seat(self):
        b = self.last_bid()
        if b:
            return b.seat()
        else:
            return None

    def last_double(self):
        for event in self._reversed_bidding_events():
            if event.is_double() or event.is_redouble():
                return event
            elif not event.is_pass():
                return None

    def last_double_as_call(self):
        event = self.last_double()
        if event:
            return event.tokens[2]
        return 'pass'

    def trailing_pass_count(self):
        count = 0
        for event in reversed(self.events):
            if event.is_pass():
                count += 1
            else:
                break
        return count

    def pass_position(self):
        if self.contract_index == -1:
            return self.trailing_pass_count()

    def last_lead(self):
        lead_event = None
        for event in reversed(self.events):
            if event.is_play():
                lead_event = event
            elif event.is_trick() or event.is_contract():
                break
        return lead_event

    def dealer(self):
        if not self.events:
            return None
        else:
            return self.events[0].tokens[0]

    def num_actions(self):
        count = 0
        for event in self.events:
            if event.is_bid() or event.is_call() or event.is_play():
                count += 1
        return count

    def action(self, action_num):
        count = 0
        for event in self.events:
            if event.is_bid() or event.is_call() or event.is_play():
                if count == action_num:
                    return event
                count += 1

    def contract_strain(self):
        if self.contract_index == -1:
            return None
        else:
            return self.events[self.contract_index].strain()

    def contract_level(self):
        if self.contract_index == -1:
            return None
        else:
            return self.events[self.contract_index].level()

    def contract_seat(self):
        if self.contract_index == -1:
            return None
        else:
            return self.events[self.contract_index].seat()

    def contract_seat_index(self):
        if self.contract_index == -1:
            return None
        else:
            return _seats.index[self.events[self.contract_index].seat()]

    def contract_doubled(self):
        if self.contract_index == -1:
            return None
        else:
            return self.events[self.contract_index].doubled()

    def contract_doubled_as_call(self):
        translation = {
                None: None,
                'undoubled': 'pass',
                'doubled': 'double',
                'redoubled': 'redouble',
        }
        return translation[self.contract_doubled()]

    def trick_counts(self):
        counts = {"North": 0, "South": 0, "East": 0, "West": 0}
        for event in self.events:
            if event.is_trick():
                counts[event.seat()] += 1
        return counts

    def is_final(self):
        return self._is_final

    def are_ns_vulnerable(self):
        return "North" in self.vulnerability

    def are_ew_vulnerable(self):
        return "East" in self.vulnerability

    def first_seat_of_partnership_to_mention_suit(self):
        return self.first_mention

    def trick_suit(self):
        event = self.last_lead()
        if event:
            return event.suit()

    def trick_winner(self):
        lead_event = self.last_lead()
        if not lead_event:
            return lead_event
        lead_suit_ix = _suits.index[lead_event.suit()]
        contract_strain_ix = _strains.index[self.events[self.contract_index].strain()]
        winning_event = lead_event
        for event in reversed(self.events):
            if event == lead_event:
                break
            elif event.suit() == winning_event.suit():
                if _ranks.index[event.rank()] > _ranks.index[winning_event.rank()]:
                    winning_event = event
            elif _suits.index[event.suit()] == contract_strain_ix:
                winning_event = event
        return winning_event

    def trick_winning_seat(self):
        ev = self.trick_winner()
        if ev:
            return ev.seat()

    def trick_winning_suit(self):
        ev = self.trick_winner()
        if ev:
            return ev.suit()

    def trick_winning_rank(self):
        ev = self.trick_winner()
        if ev:
            return ev.rank()


class Event(object):
    """Wrapper around an array of tokens."""
    def __init__(self, tokens, commentary=None, explanation=None):
        self.tokens = tokens
        self.commentary = commentary or []
        self.explanation = explanation

    def __str__(self):
        return str(self.tokens)

    def is_bid(self):
        return (len(self.tokens) == 4 and
                self.tokens[0] in _seats.index and
                self.tokens[1] == "bids" and
                self.tokens[2] in _levels.index and
                self.tokens[3] in _strains.index)

    def is_pass(self):
        return (len(self.tokens) == 4 and
                self.tokens[0] in _seats.index and
                self.tokens[1] == "bids" and
                self.tokens[2] == "pass")

    def is_double(self):
        return (len(self.tokens) == 4 and
                self.tokens[0] in _seats.index and
                self.tokens[1] == "bids" and
                self.tokens[2] == "double")

    def is_redouble(self):
        return (len(self.tokens) == 4 and
                self.tokens[0] in _seats.index and
                self.tokens[1] == "bids" and
                self.tokens[2] == "redouble")

    def is_call(self):
        return self.is_bid() or self.is_pass() or self.is_double() or self.is_redouble()

    def is_play(self):
        return (len(self.tokens) == 4 and
                self.tokens[0] in _seats.index and
                self.tokens[1] == "plays" and
                self.tokens[2] in _suits.index and
                self.tokens[3] in _ranks.index)

    def is_contract(self):
        return (len(self.tokens) == 5 and
                self.tokens[0] in _seats.index and
                self.tokens[1] == "declares" and
                self.tokens[2] in _levels.index and
                self.tokens[3] in _strains.index)

    def is_result(self):
        return (len(self.tokens) == 5 and
                self.tokens[0] in _levels.index and
                self.tokens[1] in _strains.index and
                self.tokens[2] in _seats.index and
                self.tokens[4] in _outcomes.index)

    def is_trick(self):
        return (len(self.tokens) == 2 and
                self.tokens[0] in _seats.index and
                self.tokens[1] == "takes_trick")

    def seat(self):
        if len(self.tokens) > 0 and self.tokens[0] in _seats.index:
            return self.tokens[0]

    def strain(self):
        if len(self.tokens) > 3 and self.tokens[3] in _strains.index:
            return self.tokens[3]

    def level(self):
        if len(self.tokens) > 2 and self.tokens[2] in _levels.index:
            return self.tokens[2]

    def doubled(self):
        if len(self.tokens) > 4:
            return self.tokens[4]

    def suit(self):
        if self.is_play():
            return self.tokens[2]

    def rank(self):
        if self.is_play():
            return self.tokens[3]

    def outcome(self):
        if self.is_result():
            return self.tokens[4]
            


class Tokens(object):
    def __init__(self, tokens):
        self.tokens = tokens
        self.index = {v:i for i, v in enumerate(tokens)}
        self.rindex = {i:v for i, v in enumerate(tokens)}


_extra = Tokens([
    "[PAD]", "[MASK]", "[ACTION]", "[RESULT]", "[SCORE]", "[TO_ACT]",
    "[CONTRACT_SEAT]", "deals", "undoubled", "doubled", "redoubled",
    "passed_out", "declares", "takes_trick", "gets", "sits", "vulnerable"])


_scorings = Tokens(["Matchpoints", "IMPs", "total_points"])


_seats = Tokens(["South", "West", "North", "East"])


_suits = Tokens(["Club", "Diamond", "Heart", "Spade"])


_ranks = Tokens(["Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
    "Nine", "Ten", "Jack", "Queen", "King", "Ace"])


_strains = Tokens(["Clubs", "Diamonds", "Hearts", "Spades", "notrump"])


_levels = Tokens(["1", "2", "3", "4", "5", "6", "7"])

# NOTE: all outcomes are included in _scores.
#       consequently, we don't include them in all_tokens.
_outcomes = Tokens([str(x) for x in range(-13,0)] + ["="] +
                   [ "+" + str(x) for x in range(1,6)])

_scores = Tokens([str(x) for x in range(-24,0)] + ["="] +
                   [ "+" + str(x) for x in range(1,25)])

_cards = Tokens(["{}_{}".format(s, r)
                 for s in _suits.tokens for r in _ranks.tokens])

_bids = Tokens(["{}_{}".format(l, s)
                for l in _levels.tokens for s in _strains.tokens])

_calls = Tokens(["pass", "double", "redouble"])

_action_verbs = Tokens(["{}_bids".format(s) for s in _seats.tokens] +
                       ["{}_plays".format(s) for s in _seats.tokens])

_actions = Tokens(_bids.tokens + _calls.tokens + _cards.tokens)

# TODO: move these into _extra
_new_tokens = Tokens(["[HIDDEN]"])

all_tokens = Tokens(
    _extra.tokens +
    _actions.tokens + _scores.tokens + _action_verbs.tokens +
    _scorings.tokens + _seats.tokens +
    [t[1].lower() for t in players.players_by_frequency] +
    _new_tokens.tokens)

num_actions = len(_actions.tokens)
num_scores = len(_scores.tokens)
num_action_verbs = len(_action_verbs.tokens)
first_action_id = all_tokens.index[_actions.tokens[0]]
first_score_id = all_tokens.index[_scores.tokens[0]]
first_action_verb_id = all_tokens.index[_action_verbs.tokens[0]]


_IMP_table = [ 20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600, 750,
        900, 1100, 1300, 1500, 1750, 2000, 2250, 2500, 3000, 3500, 5000, 1e99 ]


def _make_deal_event(seat_ix):
    seat = _seats.rindex[seat_ix]
    return Event([seat, "deals"])


def _make_call_event(seat_ix, call):
    seat = _seats.rindex[seat_ix]
    return Event([seat, "bids", call, "[PAD]"])


def _make_bid_event(seat_ix, level_ix, strain_ix):
    seat = _seats.rindex[seat_ix]
    level = _levels.rindex[level_ix]
    strain = _strains.rindex[strain_ix]
    return Event([seat, "bids", level, strain])


def _make_passed_out_event():
    return Event(["passed_out"])

def _make_contract_event(seat, bid_event, double_event):
    if double_event:
        if double_event.is_double():
            double_token = "doubled"
        else:
            double_token = "redoubled"
    else:
        double_token = "undoubled"
    return Event([seat, "declares", bid_event.level(), bid_event.strain(), double_token])

def _make_play_event(seat_ix, suit_ix, rank_ix):
    seat = _seats.rindex[seat_ix]
    suit = _suits.rindex[suit_ix]
    rank = _ranks.rindex[rank_ix]
    return Event([seat, "plays", suit, rank])

def _make_trick_event(seat):
    return Event([seat, "takes_trick"])

if __name__ == "__main__":
    print(len(all_tokens.tokens))
