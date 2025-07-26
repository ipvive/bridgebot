"""Create tokens file for simple bridge language, designed for use with masked LM"""
from bridgebot.bridge import game as bridgegame


class Tokenizer(object):
    """Tokenizer for formal bridge language."""
    def tokenize_score(self, score):
        if score == 0:
            return "="
        else:
            return "{0:+}".format(score)

    def tokenize_event(self, event):
        tokens = []
        if event.is_bid() or event.is_play():
            tokens.append("{}_{}".format(event.tokens[0], event.tokens[1]))
            tokens.append("{}_{}".format(event.tokens[2], event.tokens[3]))
        elif event.is_call():
            tokens.append("{}_{}".format(event.tokens[0], event.tokens[1]))
            tokens.append(event.tokens[2])
        elif event.is_contract():
            tokens.append(event.tokens[0])
            tokens.append(event.tokens[1])
            tokens.append("{}_{}".format(event.tokens[2], event.tokens[3]))
            tokens.append(event.tokens[4])
        elif event.is_result():
            if event.tokens[0] == "passed_out":
                tokens.extend(["passed_out"] * 4)
            else:
                tokens.append("{}_{}".format(event.tokens[0], event.tokens[1]))
                tokens.append(event.tokens[2])
                tokens.append(event.tokens[3])
                tokens.append(event.tokens[4])
        else:
            tokens.extend(event.tokens)
        return tokens
 
    def tokenize_view(self, view, rng):
        tokens = []
        for i, seat in enumerate(bridgegame._seats.tokens):
            tokens.extend([seat, "sits", seat])
        if view.vulnerability:
            tokens.append("vulnerable")
            tokens.extend(view.vulnerability)
        for i, seat in enumerate(view.game.seats):
            cards = []
            for j, suit in enumerate(view.game.suits):
                for k, rank in enumerate(view.game.ranks):
                    if view.dealt_cards[i, j, k]:
                        cards.append("{}_{}".format(suit, rank))
            if cards:
                rng.shuffle(cards)
                tokens.extend([seat, "gets"])
                tokens.extend(cards)
        for event in view.events:
            tokens.extend(self.tokenize_event(event))
        return tokens

    def tokens_to_ids(self, tokens):
        return [bridgegame.all_tokens.index[t] for t in tokens]

    def ids_to_tokens(self, ids):
        return [bridgegame.all_tokens.rindex[i] for i in ids]
