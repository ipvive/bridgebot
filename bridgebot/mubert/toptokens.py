"""Tokenization for top network queries and responses."""

import tensorflow as tf

MAX_TOKENS=32

class Tokens(object):
    def __init__(self, tokens):
        self.tokens = tokens
        self.index = {v:i for i, v in enumerate(tokens)}
        self.rindex = {i:v for i, v in enumerate(tokens)}


class Tokenizer(object):
    """Tokenizer for formal bridge language."""
    def __init__(self):
        self.used_tokens = Tokens([
            "[PAD]", "[PROBLEM_1]", "[VARIANT_1]", "[IS_REAL]", "[YES]", "[NO]", "[VARIANT_2]"])
        self.unused_tokens = Tokens(
                ["[[UNUSED_{}]]".format(i)
                    for i in range(MAX_TOKENS - len(self.used_tokens.tokens))])
        self.all_tokens = Tokens(self.used_tokens.tokens + self.unused_tokens.tokens)

    def tokens_to_ids(self, tokens):
        return [self.all_tokens.index[t] for t in tokens]

    def ids_to_tokens(self, ids):
        return [self.all_tokens.rindex[i] for i in ids]
