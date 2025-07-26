from absl.testing import absltest
import copy
import io
import pickle
from absl import flags
import numpy.testing
import random
import time

import bridgebot.bridge.fastgame.wrapper as bridgegame
import bridgebot.bridge.game as origgame
import bridgebot.bridge.lin as lin
#import bridgebot.scaffold.fsa as fsa


import pdb

flags.DEFINE_string("games", None, "comma-sseparated list of games to test")

FLAGS = flags.FLAGS



class Reader(io.StringIO):
    def __init__(self, buffer=None):
        super().__init__(buffer)
        self.name = "test"


class StateTest(absltest.TestCase):
    def test_init(self):
        state = bridgegame.DealState(3, 0, 2, 4)
        self.assertEqual(state.next_to_act, 3)
        self.assertEqual(state.dealt_cards[3][3][12], 0)
        self.assertEqual(state.min_length[1][2], 2)
        self.assertEqual(state.max_length[0][3], 4)

    def test_accessors(self):
        state = bridgegame.DealState()
        state._vector[316] = 17
        self.assertEqual(state.stage, 17)
        state.stage = 23
        self.assertEqual(state._vector[316], 23)
        state.dealt_cards[1][2][3] = 5
        self.assertEqual(state._vector[3 + 2 * 13 + 1 * 4 * 13], 5)


class GameTest(absltest.TestCase):
    def setUp(self):
        self.lin = lin.Parser()
        self.games = {
            'orig': origgame.Game(),
            'debug': bridgegame.Game(mode=bridgegame.MODE_DEBUG),
            'fast': bridgegame.Game(mode=bridgegame.MODE_FAST),
        }
        self.game = self.games['fast']
        if FLAGS.games:
            self.games = {n:self.games[n] for n in FLAGS.games.split(",")}
        self.good_lin = [
                """vg|Gabi Pleven Teams,Round 5_11,I,1,32,Avesta,0,Struma,0|
                rs|,,,,,,,,,,,,,,,,2HN+1,1NSx=,3CN+2,3HW-4,3SE+1,4SE=,4SW-1,4SW-1,3NN+3,3NN=,3HW-1,2HW=,1NS=,2CW-2,4HE+1,4SE+1,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,|
                pn|Ferov,Dunev,Andonov,Kovandzhiy,Slavov,Alexandrov,Videnova,Georgiev|pg||
                qx|o9|st||md|3SK97H96DQJT98CA42,SAQJ5HKT4DA7CKJT5,ST86HAQ832DK52C73,S432HJ75D643CQ986|sv|e|mb|p|mb|p|mb|1D!|mb|d|mb|1H|mb|p|mb|1N|mb|d|mb|2H|mb|p|mb|p|mb|p|pc|c6|pc|c2|pc|cK|pc|c3|pg||
                pc|cJ|pc|c7|pc|c8|pc|cA|pg||
                pc|h6|pc|h4|pc|hQ|pc|h7|pg||
                pc|d2|pc|d3|pc|dQ|pc|dA|pg||
                pc|c5|pc|h2|pg||
                """]

    def assertEventsEqual(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for i in range(len(actual)):
            self.assertEqual(actual[i].tokens, expected[i].tokens)
            self.assertEqual(actual[i].commentary, expected[i].commentary)
            self.assertEqual(actual[i].explanation, expected[i].explanation)

    def assertAllEqual(self, actual, expected):
        numpy.testing.assert_array_equal(actual, expected)

    def assertStateEqual(self, actual, expected):
        self.assertEqual(actual.stage, expected.stage)
        self.assertEqual(actual.next_to_act, expected.next_to_act)
        self.assertEqual(actual.pass_position, expected.pass_position)
        self.assertEqual(actual.last_bid_seat, expected.last_bid_seat)
        self.assertEqual(actual.last_bid_level, expected.last_bid_level)
        self.assertEqual(actual.last_bid_strain, expected.last_bid_strain)
        self.assertEqual(actual.last_bid_double, expected.last_bid_double)
        self.assertEqual(actual.declarer, expected.declarer)
        self.assertEqual(actual.trick_suit, expected.trick_suit)
        self.assertEqual(actual.trick_position, expected.trick_position)
        self.assertEqual(actual.trick_winning_seat, expected.trick_winning_seat)
        self.assertEqual(actual.trick_winning_suit, expected.trick_winning_suit)
        self.assertEqual(actual.trick_winning_rank, expected.trick_winning_rank)

        self.assertAllEqual(actual.dealt_cards, expected.dealt_cards)
        self.assertAllEqual(actual.min_length, expected.min_length)
        self.assertAllEqual(actual.max_length, expected.max_length)
        self.assertAllEqual(actual.played_cards, expected.played_cards)
        self.assertAllEqual(actual.first_to_mention, expected.first_to_mention)
        self.assertAllEqual(actual.tricks_taken, expected.tricks_taken)

        self.assertAllEqual(actual._vector, expected._vector)

    def assertDealEqual(self, actual, expected):
        self.assertEqual(actual.board_name, expected.board_name)
        self.assertEqual(actual.table_name, expected.table_name)
        self.assertEqual(actual.players, expected.players)
        self.assertEqual(actual.vulnerability, expected.vulnerability)
        self.assertStateEqual(actual._state, expected._state)
        self.assertAllEqual(actual._history, expected._history)
        self.assertEqual(actual.scoring, expected.scoring)
        self.assertEqual(actual.result, expected.result)
        self.assertEqual(actual.error, expected.error)

    def test_duplicate_card(self):
        deal = self.game.Deal()
        deal = self.game.give_card(deal, "East", "Spade", "Two")
        self.assertEqual(deal.error, None)
        deal = self.game.give_card(deal, "North", "Spade", "Two")
        self.assertNotEqual(deal.error, None)

    def test_fourteen_cards(self):
        deal = self.game.Deal()
        for rank in ["Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
                "Nine", "Ten", "Jack", "Queen", "King", "Ace"]:
            deal = self.game.give_card(deal, "South", "Club", rank)
        self.assertEqual(deal.error, None)
        deal = self.game.give_card(deal, "South", "Spade", "Two")
        self.assertNotEqual(deal.error, None)

    def test_sufficient(self):
        deal = self.game.Deal()
        deal = self.game.set_dealer(deal, "South")
        deal = self.game.make_bid(deal, "2", "Hearts")
        fail = self.game.make_bid(copy.deepcopy(deal), "2", "Clubs")
        self.assertNotEqual(fail.error, None)
        fail = self.game.make_bid(copy.deepcopy(deal), "2", "Diamonds")
        self.assertNotEqual(fail.error, None)
        fail = self.game.make_bid(copy.deepcopy(deal), "2", "Hearts")
        self.assertNotEqual(fail.error, None)
        fail = self.game.make_bid(copy.deepcopy(deal), "1", "Hearts")
        self.assertNotEqual(fail.error, None)
        ok = self.game.make_bid(copy.deepcopy(deal), "2", "Spades")
        self.assertEqual(ok.error, None)
        ok = self.game.make_bid(copy.deepcopy(deal), "3", "Hearts")
        self.assertEqual(ok.error, None)
        ok = self.game.make_bid(copy.deepcopy(deal), "3", "Clubs")
        self.assertEqual(ok.error, None)

    def test_pass_out(self):
        deal = self.game.Deal()
        deal = self.game.set_dealer(deal, "South")
        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_call(deal, "pass")
        self.assertEqual(deal.error, None)
        self.assertEqual(deal.next_to_act(), None)
        self.assertEqual(deal.pass_position(), 0)
        deal = self.game.make_call(deal, "pass")
        self.assertNotEqual(deal.error, None)
        self.assertEqual(deal.next_to_act(), None)
        self.assertTrue(deal.result)
        self.assertEqual(deal.result.cls, 'passed out')

    def test_double(self):
        deal = self.game.Deal()
        deal = self.game.set_dealer(deal, "South")
        deal = self.game.make_bid(deal, "1", "Clubs")
        dealP = self.game.make_call(copy.deepcopy(deal), "pass")
        dealPP = self.game.make_call(copy.deepcopy(dealP), "pass")
        dealPPX = self.game.make_call(copy.deepcopy(dealPP), "double")
        dealPPXP = self.game.make_call(copy.deepcopy(dealPPX), "pass")
        dealPPXPP = self.game.make_call(copy.deepcopy(dealPPXP), "pass")
        dealPPXPPR = self.game.make_call(copy.deepcopy(dealPPXPP), "redouble")
        dealPPXPPRP = self.game.make_call(copy.deepcopy(dealPPXPPR), "pass")

        dealR = self.game.make_call(copy.deepcopy(deal), "redouble")
        dealPX = self.game.make_call(copy.deepcopy(dealP), "double")
        dealPR = self.game.make_call(copy.deepcopy(dealP), "redouble")
        dealPPXPX = self.game.make_call(copy.deepcopy(dealPPXP), "double")
        dealPPXPR = self.game.make_call(copy.deepcopy(dealPPXP), "redouble")
        dealPPXPPRPX = self.game.make_call(copy.deepcopy(dealPPXPPRP), "double")
        dealPPXPPRPR = self.game.make_call(copy.deepcopy(dealPPXPPRP), "redouble")
        
        self.assertEqual(dealPPXPPRP.error, None)
        self.assertNotEqual(dealR.error, None)
        self.assertNotEqual(dealPX.error, None)
        self.assertNotEqual(dealPR.error, None)
        self.assertNotEqual(dealPPXPX.error, None)
        self.assertNotEqual(dealPPXPR.error, None)
        self.assertNotEqual(dealPPXPPRPX.error, None)
        self.assertNotEqual(dealPPXPPRPR.error, None)

    def test_possible_actions(self):
        deal = self.game.Deal()
        deal = self.game.set_dealer(deal, "South")
        actions = self.game.possible_action_indices(deal)
        actions_set = frozenset(actions)
        self.assertTrue(bridgegame.Action.from_level_and_strain('1', 'Clubs') \
                in actions_set)
        self.assertTrue(bridgegame.Action.from_level_and_strain('7', 'notrump') \
                in actions_set)
        self.assertTrue(bridgegame.Action.from_call('pass') in actions_set)
        self.assertFalse(bridgegame.Action.from_suit_and_rank('Club', 'Seven') \
                in actions_set)
        deal = self.game.make_bid(deal, "1", "Hearts")
        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_call(deal, "pass")
        actions = self.game.possible_action_indices(deal)
        actions_set = frozenset(actions)
        self.assertTrue(bridgegame.Action.from_suit_and_rank('Club', 'Seven') \
                in actions_set)
        self.assertFalse(bridgegame.Action.from_level_and_strain('7', 'notrump') \
                in actions_set)


    def test_info(self):
        deal = self.game.Deal()
        deal = self.game.set_dealer(deal, "South")
        deal = self.game.give_card(deal, "East", "Spade", "Two")
        self.assertEqual(deal.num_actions(), 0)
        self.assertEqual(deal.next_to_act(), "South")

        deal = self.game.make_call(deal, "pass")
        self.assertEqual(deal.num_actions(), 1)
        self.assertEqual(deal.next_to_act(), "West")

        deal = self.game.make_bid(deal, "1", "Hearts")
        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_call(deal, "pass")
        self.assertEqual(deal.error, None)
        self.assertEqual(deal.num_actions(), 4)
        self.assertEqual(deal.next_to_act(), "South")
        self.assertEqual(deal.contract_strain(), None)
        self.assertEqual(deal.contract_level(), None)
        self.assertEqual(deal.contract_seat(), None)
        self.assertEqual(deal.trick_counts()['North'], 0)
        self.assertEqual(deal.trick_counts()['West'], 0)

        deal = self.game.make_call(deal, "pass")
        self.assertEqual(deal.num_actions(), 5)
        self.assertEqual(deal.next_to_act(), "North")
        self.assertEqual(deal.contract_strain(), "Hearts")
        self.assertEqual(deal.contract_level(), "1")
        self.assertEqual(deal.contract_seat(), "West")

        deal = self.game.play_card(deal, "Club", "Eight")
        deal = self.game.play_card(deal, "Club", "Ace")
        deal = self.game.play_card(deal, "Club", "Two")
        self.assertEqual(deal.num_actions(), 8)
        deal = self.game.play_card(deal, "Club", "Nine")
        self.assertEqual(deal.num_actions(), 9)
        self.assertEqual(deal.next_to_act(), "East")
        self.assertDictEqual(deal.trick_counts(), 
                {"North": 0, "East": 1, "South": 0, "West": 0})

    def test_error_info(self):
        deal = self.game.Deal()
        deal = self.game.set_dealer(deal, "South")
        self.assertEqual(deal.next_to_act(), "South")
        deal = self.game.play_card(deal, "Club", "Eight")
        self.assertEqual(deal.next_to_act(), "South")

    def test_first_to_mention(self):
        deal = self.game.Deal()
        deal = self.game.set_dealer(deal, "South")
        deal = self.game.make_bid(deal, "1", "Clubs")
        deal = self.game.make_bid(deal, "3", "Clubs")
        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_bid(deal, "3", "Diamonds")

        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_bid(deal, "3", "Hearts")
        deal = self.game.make_call(deal, "pass")

        deal = self.game.make_bid(deal, "4", "Hearts")
        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_bid(deal, "5", "Clubs")

        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_call(deal, "pass")
        deal = self.game.make_call(deal, "pass")

        self.assertEqual(deal.contract_level(), "5")
        self.assertEqual(deal.contract_strain(), "Clubs")
        self.assertEqual(deal.contract_seat(), "West")

    def test_play_of_anothers_card(self):
        deal = self.game.Deal()
        deal = self.game.set_dealer(deal, "South")
        deal = self.game.give_card(deal, "South", "Club", "Two")
        deal = self.game.make_bid(deal, "1", "notrump")
        for _ in range(3):
            deal = self.game.make_call(deal, "pass")
        deal = self.game.play_card(deal, "Club", "Three")
        self.assertEqual(deal.error, None)
        deal = self.game.play_card(deal, "Club", "Two")
        self.assertNotEqual(deal.error, None)

    def test_duplicate_play(self):
        deal = self.game.Deal()
        deal = self.game.set_dealer(deal, "South")
        deal = self.game.make_bid(deal, "1", "notrump")
        for _ in range(3):
            deal = self.game.make_call(deal, "pass")
        deal = self.game.play_card(deal, "Club", "Two")
        self.assertEqual(deal.error, None)
        deal = self.game.play_card(deal, "Club", "Two")
        self.assertNotEqual(deal.error, None)

    def test_revoke(self):
        deal = self.game.Deal()
        deal = self.game.set_dealer(deal, "South")
        deal = self.game.make_bid(deal, "1", "notrump")
        for _ in range(3):
            deal = self.game.make_call(deal, "pass")
        deal = self.game.play_card(deal, "Club", "Ace")
        deal = self.game.play_card(deal, "Club", "Ten")
        deal = self.game.play_card(deal, "Club", "Nine")
        deal = self.game.play_card(deal, "Club", "Eight")  # West takes trick
        one_notrump_deal = copy.deepcopy(deal)

        deal = self.game.play_card(deal, "Club", "Seven")
        deal = self.game.play_card(deal, "Spade", "Ace")
        self.assertEqual(deal.error, None)
        deal = self.game.give_card(deal, "North", "Club", "Three")
        self.assertNotEqual(deal.error, None)

        deal = copy.deepcopy(one_notrump_deal)
        deal = self.game.play_card(deal, "Club", "Seven")
        deal = self.game.give_card(deal, "North", "Club", "Three")
        self.assertEqual(deal.error, None)
        deal = self.game.play_card(deal, "Spade", "Ace")
        self.assertNotEqual(deal.error, None)

        deal = copy.deepcopy(one_notrump_deal)
        deal = self.game.play_card(deal, "Club", "Seven")
        deal = self.game.play_card(deal, "Spade", "Ace")
        deal = self.game.play_card(deal, "Club", "Three")
        deal = self.game.play_card(deal, "Club", "Four")  # South wins trick
        deal = self.game.play_card(deal, "Club", "Five")
        self.assertEqual(deal.error, None, msg=deal)
        deal = self.game.play_card(deal, "Club", "Six")
        self.assertNotEqual(deal.error, None, msg=deal)

    def test_strains(self):
        def winner(self, strain, lead_suit, other_suit, other_rank):
            deal = self.game.Deal()
            deal = self.game.set_dealer(deal, "South")
            deal = self.game.make_bid(deal, "1", strain)
            for _ in range(3):
                deal = self.game.make_call(deal, "pass")
            deal = self.game.play_card(deal, lead_suit, "Eight")
            deal = self.game.play_card(deal, other_suit, other_rank)
            deal = self.game.play_card(deal, lead_suit, "King")
            deal = self.game.play_card(deal, lead_suit, "Three")
            self.assertEqual(deal.error, None)
            return deal.next_to_act()

        self.assertEqual(winner(self, "Diamonds", "Diamond", "Diamond", "Ace"), "North")
        self.assertEqual(winner(self, "notrump", "Club", "Club", "Ace"), "North")
        self.assertEqual(winner(self, "Diamonds", "Club", "Diamond", "Ace"), "North")
        self.assertEqual(winner(self, "Diamonds", "Club", "Diamond", "Two"), "North")

        self.assertEqual(winner(self, "notrump", "Club", "Spade", "Ace"), "East")
        self.assertEqual(winner(self, "notrump", "Club", "Diamond", "Ace"), "East")
        self.assertEqual(winner(self, "notrump", "Club", "Heart", "Ace"), "East")
        self.assertEqual(winner(self, "Diamonds", "Club", "Spade", "Ace"), "East")
        self.assertEqual(winner(self, "Diamonds", "Diamond", "Club", "Ace"), "East")
        self.assertEqual(winner(self, "Diamonds", "Diamond", "Heart", "Ace"), "East")
        self.assertEqual(winner(self, "Diamonds", "Diamond", "Spade", "Ace"), "East")

        deal = self.game.Deal()
        deal = self.game.set_dealer(deal, "South")
        deal = self.game.make_bid(deal, "1", "Clubs")
        for _ in range(3):
            deal = self.game.make_call(deal, "pass")
        deal = self.game.play_card(deal, "Heart", "Eight")
        deal = self.game.play_card(deal, "Club", "Seven")
        deal = self.game.play_card(deal, "Club", "Nine")
        deal = self.game.play_card(deal, "Heart", "Ace")
        self.assertEqual(deal.next_to_act(), "East")

        deal = self.game.Deal()
        deal = self.game.set_dealer(deal, "South")
        deal = self.game.make_bid(deal, "1", "Clubs")
        for _ in range(3):
            deal = self.game.make_call(deal, "pass")
        deal = self.game.play_card(deal, "Heart", "Eight")
        deal = self.game.play_card(deal, "Club", "Nine")
        deal = self.game.play_card(deal, "Club", "Seven")
        deal = self.game.play_card(deal, "Heart", "Ace")
        self.assertEqual(deal.next_to_act(), "North")


    def test_lin(self):
        for lindata in self.good_lin:
            boards, errors = self.lin.parse(Reader(lindata), self.game)
            self.assertEqual(errors, {})
            self.assertEqual(len(boards), 1)


    @absltest.skip
    def test_commentary(self):
        lindata = """
                vg|Swiss Interclub Teams Championship,Final - Seg 3_4,I,1,16,Begues I,69,Contact I,54|
                rs|1NW-2,2SN-1,3NW+1,3NW-1,1NW=,2SN-1,4HN+1,4HN+1,3CS=,2CS+1,1SE+1,PASS,3SS-2,1NS-2,3SS=,4SS-1,3DS=,4DS-1,5CS=,4CS+1,4SE+2,4SE+2,4HN+1,5HN=,3CN-1,2CN=,1SN+1,2SN=,3DS+1,,3NS+3,|
                pn|Latinov J,Piedra F,Walter S,Caponi C,Nikolenkov,Varenne,Magnusson,Cabaj|pg||
                qx|c6|st||md|4S97HQJ94DQT42CK87,SAJ4H765DAJ7CQT54,SK852HKT3DK98CJ63,SQT63HA82D653CA92|sv|e|nt|athene: without firm agreements that 3@C is sign-off, you have to pass 2@C, i think|pg||
                nt|chthonic: Ouch|pg||
                an|vugraphzaf: North not happy!!!|pg||
                nt|davidstern: 4@S for EW looks the go here|pg||
                nt|lestergold: Waterlow is the anchor player of this partnershuip probably under 'instructions' from hackett to play his steady game andleave the fireworks to partner|pg||
                mb|p|nt|mdgraham: chance missed for EW there - the little red card stayed in the box|pg||
                nt|davidstern: Obviously no diamond lead in the OR|pg||
                mb|p|nt|athene: 1@S opener or 2NT?|pg||
                nt|athene: double is a possible alternative for south|pg||
                nt|athene: but he prefers to get the good suit in|pg||
                nt|april7: 5@C or 3NT for the lucky position of the @D suit|pg||
                nt|vugraphzaf: No -4 agreed|pg||
                mb|p|mb|p|nt|athene: 1@S forcing, no major (X would be hearts; 1@H would be spades)|pg||
                nt|davidstern: Even so how do you make six|pg||
                nt|davidstern: On a heart lead would you really finesse?|pg||
                nt|jzinsli: wau pass at thrid position :(|pg||
                nt|cedricgva: jolie séquence|pg||
                pg||
                """
        deal = self.lin.parse_single(Reader(lindata), self.game)
        self.assertEqual(deal.events[2].commentary[1],
          'athene: double is a possible alternative for south')
        self.assertEqual(deal.events[5].commentary[-1], 'cedricgva: jolie séquence')
        self.assertEqual(deal.events[0].explanation, 'vugraphzaf: North not happy!!!')
        deal_msg = self.game.played_game_from_deal(deal)
        deal_2 = self.game.deal_from_played_game(deal_msg)
        deal_2_msg = self.game.played_game_from_deal(deal_2)
        self.assertDealEqual(deal_2, deal)
        self.assertEqual(deal_2_msg, deal_msg)

    def test_views(self):
        orig_lin = """
                   rs|2HN+1|
                   pn|,,,,,,,|pg||
                   qx|o9|st||md|3SK97H96DQJT98CA42,SAQJ5HKT4DA7CKJT5,ST86HAQ832DK52C73,S432HJ75D643CQ986|pg||
                   mb|p|mb|p|mb|1D!|mb|d|mb|1H|mb|p|mb|1N|mb|d|mb|2H|mb|p|mb|p|mb|p|pg||
                   pc|c6|pc|c2|pc|cK|pc|c3|pg||
                   pc|cJ|pc|c7|pc|c8|pc|cA|pg||
                   pc|h6|pc|h4|pc|hQ|pc|h7|pg||
                   pc|d2|pc|d3|pc|dQ|pc|dA|pg||
                   pc|c5|pc|h2|pc|c9|pc|c4|pg||
                   pc|hA|pc|h5|pc|h9|pc|hT|pg||
                   pc|h3|pc|hJ|pc|s7|pc|hK|pg||
                   pc|cT|pc|h8|pc|cQ|pc|s9|pg||
                   pc|dK|pc|d4|pc|d8|pc|d7|pg||
                   mc|9|pg||
                   """
        kib_lin = {11: """
                       rs|2HN+1|
                       pn|,,,,,,,|pg||
                       qx|o9|st||md|3SK97H96DQJT98CA42,SAQJ5HKT4DA7CKJT5,ST86HAQ832DK52C73,S432HJ75D643CQ986|pg||
                       mb|p|mb|p|mb|1D!|mb|d|mb|1H|mb|p|mb|1N|mb|d|mb|2H|mb|p|mb|p|
                       """,
        }
        act_lin = {11: """
                       rs|2HN+1|
                       pn|,,,,,,,|pg||
                       qx|o9|st||md|3,SAQJ5HKT4DA7CKJT5,,|pg||
                       mb|p|mb|p|mb|1D!|mb|d|mb|1H|mb|p|mb|1N|mb|d|mb|2H|mb|p|mb|p|
                       """,
                   12: """
                       rs|2HN+1|
                       pn|,,,,,,,|pg||
                       qx|o9|st||md|3,,,S432HJ75D643CQ986|pg||
                       mb|p|mb|p|mb|1D!|mb|d|mb|1H|mb|p|mb|1N|mb|d|mb|2H|mb|p|mb|p|mb|p|
                       """,
                   13: """
                       rs|2HN+1|
                       pn|,,,,,,,|pg||
                       qx|o9|st||md|3SK97H96DQJT98CA42,,ST86HAQ832DK52C73,|pg||
                       mb|p|mb|p|mb|1D!|mb|d|mb|1H|mb|p|mb|1N|mb|d|mb|2H|mb|p|mb|p|mb|p|
                       pc|c6|
                       """,
                   14: """
                       rs|2HN+1|
                       pn|,,,,,,,|pg||
                       qx|o9|st||md|3SK97H96DQJT98CA42,SAQJ5HKT4DA7CKJT5,,|pg||
                       mb|p|mb|p|mb|1D!|mb|d|mb|1H|mb|p|mb|1N|mb|d|mb|2H|mb|p|mb|p|mb|p|
                       pc|c6|pc|c2|
                       """,
                   15: """
                       rs|2HN+1|
                       pn|,,,,,,,|pg||
                       qx|o9|st||md|3SK97H96DQJT98CA42,,ST86HAQ832DK52C73,|pg||
                       mb|p|mb|p|mb|1D!|mb|d|mb|1H|mb|p|mb|1N|mb|d|mb|2H|mb|p|mb|p|mb|p|
                       pc|c6|pc|c2|pc|cK|
                       """}
        tab_lin = {11: """
                       rs|2HN+1|
                       pn|,,,,,,,|pg||
                       qx|o9|st||md|,,,|pg||
                       mb|p|mb|p|mb|1D!|mb|d|mb|1H|mb|p|mb|1N|mb|d|mb|2H|mb|p|mb|p|
                       """,
                   13: """
                       rs|2HN+1|
                       pn|,,,,,,,|pg||
                       qx|o9|st||md|3SK97H96DQJT98CA42,,,|pg||
                       mb|p|mb|p|mb|1D!|mb|d|mb|1H|mb|p|mb|1N|mb|d|mb|2H|mb|p|mb|p|mb|p|
                       pc|c6|
                       """}
        orig_deal = self.lin.parse_single(Reader(orig_lin), self.game)
        for n, lin in kib_lin.items():
            kib_deal = self.lin.parse_single(Reader(lin), self.game)
            print(kib_deal, orig_deal)
            self.assertDealEqual(self.game.kibitzer_view(orig_deal, n), kib_deal)
            #TODO kib_msg = self.game.played_game_from_deal(kib_deal)
            #TODO kib_deal_2 = self.game.deal_from_played_game(kib_msg)
            #TODO self.assertDealEqual(kib_deal_2, kib_deal)

        for n, lin in tab_lin.items():
            tab_deal = self.lin.parse_single(Reader(lin), self.game)
            self.assertDealEqual(self.game.table_view(orig_deal, n), tab_deal)
            #TODO tab_msg = self.game.played_game_from_deal(tab_deal)
            #TODO tab_deal_2 = self.game.deal_from_played_game(tab_msg)
            #TODO self.assertDealEqual(tab_deal_2, tab_deal)

        for n, lin in act_lin.items():
            act_deal = self.lin.parse_single(Reader(lin), self.game)
            self.assertDealEqual(self.game.actor_view(orig_deal, n), act_deal)
            #TODO act_msg = self.game.played_game_from_deal(act_deal)
            #TODO act_deal_2 = self.game.deal_from_played_game(act_msg)
            #TODO self.assertDealEqual(act_deal_2, act_deal)

        pass_lin = """
                   rs|PASS|
                   pn|,,,,,,,|pg||
                   qx|o9|st||md|3SK97H96DQJT98CA42,SAQJ5HKT4DA7CKJT5,ST86HAQ832DK52C73,S432HJ75D643CQ986|pg||
                   nt|not in view|pg||
                   mb|p|mb|p|mb|p|mb|p|pg||
                   """
        act_pass_lin = """
                   rs|PASS|
                   pn|,,,,,,,|pg||
                   qx|o9|st||md|3SK97H96DQJT98CA42,SAQJ5HKT4DA7CKJT5,ST86HAQ832DK52C73,S432HJ75D643CQ986|pg||
                   mb|p|mb|p|mb|p|mb|p|pg||
                   """
        pass_deal = self.lin.parse_single(Reader(pass_lin), self.game)
        act_pass_deal = self.lin.parse_single(Reader(act_pass_lin), self.game)
        self.assertDealEqual(self.game.actor_view(pass_deal, 4), act_pass_deal)

    def test_table_score(self):
        num_diffs = 0
        for case_datum in [
                # passed out
                ("", "pass", 0, 0),
                # making partscores,
                ("", "1NS=", 90, None),
                ("", "2NS=", 120, None),
                ("", "2SS=", 110, None),
                ("", "3HS=", 140, None),
                ("", "4DS=", 130, None),
                ("", "3CS=", 110, None),
                # game, {,grand-}slam; V+NV.
                ("NSEW", "4HS=", 620, None),
                ("NSEW", "6HS=", 1430, None),
                ("NSEW", "7HS=", 2210, None),
                ("", "4HS=", 420, None),
                ("", "6HS=", 980, None),
                ("", "7HS=", 1510, None),
                # overtricks
                ("", "2HS+2", 170, None),
                ("", "4HS+2", 480, None),
                ("", "6HS+1", 1010, None),
                # undertricks
                ("NSEW", "1HS-1", None, 100),
                ("", "1HS-1", None, 50),
                ("NSEW", "1HS-7", None, 700),
                # successful doubles
                ("NSEW", "1HSx-1", None, 200),
                ("NSEW", "1HSx-2", None, 500),
                ("NSEW", "1HSx-3", None, 800),
                ("NSEW", "1HSx-4", None, 1100),
                ("", "1HSx-1", None, 100),
                ("", "1HSx-2", None, 300),
                ("", "1HSx-3", None, 500),
                ("", "1HSx-4", None, 800),
                # unsuccessful redoubles
                ("NSEW", "1HSxx-2", None, 1000),
                # unsuccessful doubles
                ("NSEW", "1HSx+1", 360, None),
                ("", "2HSx=", 470, None),
                # successful redoubles
                ("", "1DSxx=", 230, None),
                ("NSEW", "1HSxx+4", 2320, None),
                # seating
                ("NS", "3NS=", 600, None),
                ("NS", "3NN=", 600, None),
                ("EW", "3NS=", 400, None),
                ("EW", "3NN=", 400, None),
                ("NS", "3NE=", None, 400),
                ("NS", "3NW=", None, 400),
                ("EW", "3NE=", None, 600),
                ("EW", "3NW=", None, 600),
        ]:
            (vuln, result_string, ns_expected, ew_expected) = case_datum
            vuln_list = {
                    "": [],
                    "NS": ["North", "South"],
                    "EW": ["East", "West"],
                    "NSEW": ["North", "South", "East", "West"],
            }[vuln]
            result_tuple, _ = self.lin.parse_result(result_string)
            deal = self.game.set_result(self.game.Deal(), *result_tuple)
            result_event = deal.result
            expected = (ns_expected, ew_expected)
            actual = self.game.table_score(result_event, vuln_list)
            if actual != expected:
                print("{} failed: {} != {}".format(case_datum, actual, expected))
                num_diffs += 1
        self.assertEqual(num_diffs, 0)


    def test_comparison_score(self):
        for diff, expected in [
                (0, 0),
                (-10, 0),
                (10, 0),
                (20, 1),
                (-20, -1),
                (250, 6),  # non-vul game swing
                (450, 10),  # vul game swing
                (3990, 23),
                (26000, 24),
        ]:
            actual, _ = self.game.comparison_score(diff, "IMPs")
            self.assertEqual(actual, expected, str(diff))


    @absltest.skip
    def test_equivalent_info(self):
        for i, (transition, orig_transition) in enumerate(zip(
                fsa.traverse_online(self.game),
                fsa.traverse_online(origgame.Game()))):
            if transition[1:] != orig_transition[1:]:
                a = transition[3]
                e = orig_transition[3]
                h = transition[0]._history[:10]
                pdb.set_trace()

    @absltest.skip
    def test_bench_traverse(self):
        for _ in fsa.traverse_online(origgame.Game()):
            pass

    def test_bench_copy_deal(self):
        repeat = 1000
        for name, game in self.games.items():
            for lindata in self.good_lin:
                deal = self.lin.parse_single(Reader(lindata), game)
                start = time.perf_counter()
                for _ in range(repeat):
                    copy.deepcopy(deal)
                t = (time.perf_counter() - start) / repeat
                print(f"{name} copy.deepcopy: {1000*t}ms")

                start = time.perf_counter()
                for _ in range(repeat):
                    deal.copy_replay_state()
                t = (time.perf_counter() - start) / repeat
                print(f"{name} copy.copy_replay_state: {1000*t}ms")

    def test_bench_kibitzer_view(self):
        repeat = 1000
        for lindata in self.good_lin:
            for name, game in self.games.items():
                deal = self.lin.parse_single(Reader(lindata), game)
                start = time.perf_counter()
                for _ in range(repeat):
                    game.kibitzer_view(deal, deal.num_actions() - 1)
                t = (time.perf_counter() - start) / repeat
                print(f"{name}.kibitzer_view: {1000*t}ms")


class NCardTest(absltest.TestCase):
    def setUp(self):
        self.games = {
            'debug': bridgegame.Game(mode=bridgegame.MODE_DEBUG),
            'fast': bridgegame.Game(mode=bridgegame.MODE_FAST),
        }
        self.games = {
            f"{k}_{n}-card": bridgegame.Game(mode=v.mode,num_ranks=n)
                for k,v in self.games.items() for n in range(1,13)
        }
        if FLAGS.games:
            self.games = {n:self.games[n] for n in FLAGS.games.split(",")}


    def test_random_deal(self):
        for name, game in self.games.items():
                deal = game.random_deal(rng=random.Random())


if __name__ == "__main__":
    absltest.main()
