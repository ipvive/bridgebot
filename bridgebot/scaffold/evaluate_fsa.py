from absl import app, flags
import io
import os
import numpy as np
import random
import tensorflow as tf

# from bridge.fastgame import wrapper as bridgegame
from bridge.fastgame import wrapper as bridgegame
from bridge import lin
from bridge import tokens 
from scaffold import fsa as gfsa

import pickle

import pdb

FLAGS = flags.FLAGS


flags.DEFINE_string("tfrecord_lin", None, "Glob of tfrecord lin files")


def evaluate(fsa):
    game = bridgegame.Game()
    tokenizer = tokens.Tokenizer()
    action_index = bridgegame._actions.index
    card_index = bridgegame._cards.index
    parser = lin.Parser()
    lin_data = tf.data.TFRecordDataset(FLAGS.tfrecord_lin)
    for data in lin_data:
        reader = io.StringIO(data.numpy().decode("utf-8"))
        reader.name = reader.readline().strip()
        boards, err = parser.parse(reader, game)
        print(reader.name)
        for board in boards.values():
            for table in board.tables.values():
                num_actions = table.num_actions()
                obervables = [gfsa.get_obervables(game.kibitzer_view(table, i))
                    for i in range(num_actions)]
                actions = [tokenizer.tokenize_event(table.action(i))[1]
                    for i in range(num_actions)] 
                for i in range(num_actions-1):
                    if actions[i+1] not in card_index: 
                        if obervables[i+1] != fsa.apply(obervables[i], action_index[actions[i]]):
                            pdb.set_trace()
                    else:
                        break


def main(argv):
    test_fsa_file = "test_fsa"
    if os.path.exists(test_fsa_file):
        with open(test_fsa_file, 'rb') as fp:
            fsa = pickle.load(fp)
    else:
        interior,  transitions, num_actions = gfsa.traverse(bridgegame.Game())
        fsa = gfsa.FSA(interior, transitions, num_actions)    
        with open(test_fsa_file, 'wb') as fp:
            pickle.dump(fsa, fp)

    evaluate(fsa) 


if __name__ == "__main__":
    app.run(main)
