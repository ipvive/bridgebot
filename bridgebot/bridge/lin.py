"""Lin file manipulations.""" 
import collections
import sys
import logging
import pdb


Token = collections.namedtuple("Token", ["command", "argument"])


class Board(object):
    """Represents a board of cards, played at one or more tables."""
    def __init__(self, name, tables=None):
        self.name = name
        self.tables = tables or {}


class Parser(object):
    """.lin file parser."""
    def parse(self, reader, game):
        """Returns dict of Boards containing game.Deal objects."""
        # TODO(njt): skip line after error; continue with file.
        all_boards = {}
        error_counts = {}
        lin_tokens, err = self.tokenize(reader)
        if err:
            logging.info("Failed to tokenize %s: %s", reader.name, err)
            return {}, {}
        header, lin_tokens, err = self.parse_header(lin_tokens)
        if err:
            logging.info("Failed to parse header of %s: %s", reader.name, err)
            return {}, {}
        while lin_tokens:
            next_tokens = None
            for i, t in enumerate(lin_tokens):
                if i > 0 and t.command == 'qx':
                    lin_tokens, next_tokens = lin_tokens[:i], lin_tokens[i:]
                    break
            deal, err = self.parse_deal(header, lin_tokens, game)
            lin_tokens = next_tokens
            if err:
                logging.debug("Failed to parse deal in %s: %s", reader.name, err)
                s = str(err)
                if s not in error_counts:
                    error_counts[s] = 1
                else:
                    error_counts[s] += 1
            elif deal.error:
                logging.info("Failed to replay deal %s%s in %s: %s",
                        deal.table_name, deal.board_name, reader.name, deal.error)
            else:
                if deal.board_name not in all_boards:
                    all_boards[deal.board_name] = Board(deal.board_name)
                all_boards[deal.board_name].tables[deal.table_name] = deal
        return all_boards, error_counts

    def parse_single(self, reader, game):
        lin_tokens, err = self.tokenize(reader)
        if err:
            logging.info("Failed to tokenize %s: %s", reader.name, err)
            return None
        header, lin_tokens, err = self.parse_header(lin_tokens)
        if err:
            logging.info("Failed to parse header of %s: %s", reader.name, err)
            return None
        deal, err = self.parse_deal(header, lin_tokens, game)
        if err:
            logging.info("Failed to parse deal in %s: %s", reader.name, err)
        elif deal.error:
            logging.info("Failed to replay deal %s%s in %s: %s",
                    deal.table_name, deal.board_name, reader.name, deal.error)
        return deal

    def parse_header(self, lin_tokens):
        header = {}
        for i, token in enumerate(lin_tokens):
            if token.command == 'qx':
                return header, lin_tokens[i:], None
            elif token.command not in header:
                header[token.command] = token.argument
            elif token.command in frozenset(['pg', 'pn']):
                pass
            else:
                return None, None, "Duplicate command in header"
        return header, [], None

    def parse_result(self, result):
        strains = {"C": "Clubs", "D": "Diamonds", "H": "Hearts",
                   "S": "Spades", "N": "notrump"}
        players = {"S": "South", "W": "West", "N": "North", "E": "East"}
      
        if result.lower() == "pass":
            return tuple(["passed_out"] * 5), None
        if len(result) < 4: 
            return None, "Result too short" 
        if (result[1] not in strains or
            result[2] not in players):
            return None, "Malformed result"

        level = result[0]
        strain = strains[result[1]]
        player = players[result[2]]
        double, outcome = "", ""
        if result[3] == 'x' :
            if len(result) > 4 and result[4] == 'x' :
                double = "redoubled"
                outcome = result[5:]
            else :
                double = "doubled"
                outcome = result[4:]
        else :
            double = "undoubled" 
            outcome = result[3:]

        return (level, strain, player, double, outcome), None 

    def parse_deal(self, header, lin_tokens, game):
        suits, suit_names = {}, ["Spade", "Heart", "Diamond", "Club"]
        ranks = {"2": "Two", "3": "Three", "4": "Four", "5": "Five",
                "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
                "T": "Ten", "10": "Ten", "J": "Jack", "Q": "Queen", "K": "King",
                "A": "Ace"}
        strains, strain_names = {}, ["notrump", "Spades", "Hearts", "Diamonds", "Clubs"]
        for v in strain_names:
            strains[v[0].lower()] = v
            strains[v[0].upper()] = v
        for v in suit_names:
            suits[v[0].lower()] = v
            suits[v[0].upper()] = v

        deal = game.Deal()
        deal_token = lin_tokens[0]
        if deal_token.command != 'qx':
            return None, "Expected qx token"
        deal.table_name, deal.board_name = deal_token.argument[0], deal_token.argument[1:]

        if 'pn' not in header:
            return None, "Player names not found"
        players = header['pn'].split(",")
        if deal.table_name == 'o':
            players = players[:4]
        elif deal.table_name == 'c':
            players = players[4:]
        else:
            return None, "Unexpected room code"
        if len(players) != 4:
            return None, "Expected 4 players"
        deal = game.set_players(deal, *players[:4])

        try:
            board_num = int(deal.board_name.split(",")[0])
        except:
            return None, "Invalid board number"
        deal = game.set_board_number(deal, board_num)

        if 'rs' not in header:
            return None, "Results not found"
        results = header['rs'].split(",")
        while len(results) <= board_num * 2 + 1:
            # TODO: update indexing to current vugraph practise after ADDING TO TEST_LIN.
            results *= 2 # hack when board_num is shifted by 16
        if deal.table_name == 'o':
            result, err = self.parse_result(results[board_num * 2])
        else:
            result, err = self.parse_result(results[board_num * 2 + 1])
        if err is not None:
            return None, err
        deal = game.set_result(deal, *result)
        if err is not None:
            return None, err
 
        if "vg" in header and header["vg"].find("Reisinger") != -1:
            deal.scoring = "Matchpoints"
        else:
            deal.scoring = "IMPs"
        for pos, token in enumerate(lin_tokens[1:]):
            if not deal:
                logging.error("Internal error {}[{}]".format(lin_tokens, pos))
                return None, "internal error"
            if deal.error:
                return None, deal.error
            if token.command == 'md':
                cards = token.argument[1:].split(",")
                if len(cards) < 4:
                    if len(cards) == 3:
                        cards.append(None)
                    else:
                        return None, "expected three or four hands"
                cards_to_give = []
                for i, seat in enumerate(["South", "West", "North", "East"]):
                    if i >= len(cards) or not cards[i]:
                        continue
                    suit = None
                    for card in cards[i]:
                        if card in suits:
                            suit = suits[card]
                        elif card in ranks:
                            rank = ranks[card]
                            if not suit:
                                return None, "missing suit in hand specification"
                            if hasattr(game, 'give_cards'):
                                cards_to_give.append((seat, suit, rank))
                            else:
                                deal = game.give_card(deal, seat, suit, rank)
                        else:
                            return None, "unexpected card in hand specification"
                if cards_to_give:
                    deal = game.give_cards(deal, cards_to_give)
            elif token.command == 'mb':
                if token.argument.lower() == 'p' or token.argument == 'p!' or token.argument == '-':
                    deal = game.make_call(deal, "pass")
                elif token.argument.lower() == 'd' or token.argument == 'd!':
                    deal = game.make_call(deal, "double")
                elif token.argument.lower() == 'r' or token.argument == 'r!':
                    deal = game.make_call(deal, "redouble")
                else:
                    if len(token.argument) < 2:
                        return None, "invalid bid"
                    num, strain = token.argument[0], token.argument[1]
                    if num not in [str(n) for n in range(1,8)] or strain not in strains:
                        return None, "invalid bid"
                    deal = game.make_bid(deal, num, strains[strain])
            elif token.command == 'pc':
                suit, rank = token.argument[0], token.argument[1:]
                if suit not in suits or rank not in ranks:
                    return None, "invalid play"
                deal = game.play_card(deal, suits[suit], ranks[rank])
            elif token.command == 'mc':
                deal = game.accept_claim(deal, token.argument)
            elif token.command == 'an':
                deal = game.add_explanation(deal, token.argument)
            elif token.command == 'nt':
                deal = game.add_commentary(deal, token.argument)
        if not deal.result:
            return None, "no claim in deal"
        return deal, None

    def tokenize(self, reader):
        tokens = []
        while True:
            line = reader.readline()
            if not line:
                break
            while line.find("|pg||") == -1:
                next_line = reader.readline()
                if not next_line:
                    break
                line = line + next_line
            line = line.strip()
            while line:
                if len(line) < 3:
                    return None, "partial line"
                if line[2] != '|':
                    pos = line.find('|')
                    return None, "Expected 2 letters in lin command"
                if line[:2] == 'nt':
                    end_pos = line.find('|pg||')
                    if end_pos == -1:
                        if line == "nt||":
                            return tokens, None
                        return None, "Malformed lin commentary payload"
                else:
                    end_pos = line[3:].find('|')
                    if end_pos == -1:
                        return None, "Malformed lin payload"
                    end_pos = end_pos + 3
                tokens.append(Token(line[:2], line[3:end_pos]))
                line = line[end_pos + 1:].strip()
        return tokens, None
