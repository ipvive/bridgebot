from absl import app, flags
import numpy as np
import random
import skmultiflow as skm 

from bridge import game as bridgegame

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_actions", 5, "number of actions from deal")

flags.DEFINE_integer("num_deals", 10000, "number of deals to train on")


dims = [
        ("level", lambda x: x.last_bid_level(), bridgegame._levels.tokens),
        ("strain", lambda x: x.last_bid_strain(), bridgegame._strains.tokens),
        ("seat", lambda x: x.last_bid_seat(), bridgegame._seats.tokens),
        ("doubled",
            lambda x: x.last_double() and x.last_double().tokens[0] or None,
            ["double", "redouble"]),
        ("passes", lambda x: x.trailing_pass_count(), list(range(3))),
        ("to_act", lambda x: x.next_to_act(), bridgegame._seats.tokens),
        ("ns_vul", lambda x: x.are_ns_vulnerable(), None),
        ("ns_vul", lambda x: x.are_ew_vulnerable(), None),
        ("error", lambda x: x.has_error(), None),
]


def get_obervables(deal, dims):
    results = [x[1](deal) for x in dims]
    vals = []
    for result, dim in zip(results, dims):
        if dim[2]:
            vals.extend(result == x for x in dim[2])
        else:
            vals.append(result)
    return vals


def get_labels(obervables, dims):
    labels = {}
    idx = 0
    for d in dims:
        if d[2]:
            labels[d[0]] = None
            for v in d[2]:
                if obervables[idx]:
                    labels[d[0]] = v
                idx += 1
        else:
            labels[d[0]] = int(obervables[idx]) == 1
            idx += 1
    return labels


def random_play(game, num_deals, num_actions):
    num_action_labels = len(bridgegame._actions.tokens)
    for i in range(num_deals):
        deal = game.random_deal(random.Random())
        action_path = np.random.randint(0, num_action_labels, size=num_actions)
        obervables = [get_obervables(deal, dims)]
        for j, action in enumerate(action_path):
            deal = game.execute_action_index(deal, action)
            obervables.append(get_obervables(deal, dims))
        obervables = np.array(obervables)
        X = np.concatenate([obervables[:-1], action_path.reshape((-1,1))], axis=1)
        y = obervables[1:]
        yield X, y 


def train():
    game = bridgegame.Game()
    action_labels = bridgegame._actions.tokens
    data_gen = random_play(game, FLAGS.num_deals, FLAGS.num_actions)
    tree = skm.trees.iSOUPTreeRegressor()
    for X, y in data_gen: 
        tree.partial_fit(X, y)
    return tree


def evaluate(tree):
    game = bridgegame.Game()
    y_pred = []
    y_true = []
    data_gen = random_play(game, FLAGS.num_deals, FLAGS.num_actions)
    for X, y in data_gen: 
        pred = tree.predict(X) > 0.5
        if np.any(y[0] != pred[0]):
            print(get_labels(X[0,:-1], dims), bridgegame._actions.tokens[X[0,-1]])
            print(get_labels(y[0], dims)) 
            print(get_labels(pred[0], dims)) 
            print("\n" + "=" * 20 + "\n")
        y_true.extend(y)
        y_pred.extend(pred)
    y_true = np.array(y_true, dtype=np.int)
    y_pred = np.array(y_pred, dtype=np.int)
    print('Mean absolute error: {}'.format(np.mean(np.abs(y_true - y_pred))))
    print(tree.get_model_description())


def main(argv):
    trained_tree = train()
    evaluate(trained_tree)


if __name__ == "__main__":
    app.run(main)
