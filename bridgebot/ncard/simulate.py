from __future__ import division

import asyncio
import copy
from dataclasses import dataclass, field
import multiprocessing
import os
import queue
import random
import textwrap
import threading
from typing import Mapping

import absl.app
import absl.flags as flags
import absl.logging as logging
import grpc
import numpy as np

import bridgebot.ncard.game as bridgegame
from bridgebot.ncard import chords
import bridgebot.pb.alphabridge_pb2 as alphabridge_pb2
import bridgebot.pb.alphabridge_pb2_grpc as alphabridge_pb2_grpc

import pdb


class Node(object):
    def __init__(self, deal, last_action):
        self.deal = deal
        self.policy = None
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.legal_actions = None
        self.last_action = last_action

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def __repr__(self):
        s = ""
        s += f"[...,{self.last_action}] vis={self.visit_count} val={self.value()}\n"
        s += f"children: ({self.children.keys()})\n"
        for ix, child in self.children.items():
            s += textwrap.indent(child.__repr__(), "    ")
        return s


@dataclass
class ScoreInfo:
    outcome_count: int = 0
    outcomes: Mapping[tuple, int] = field(default_factory=lambda: {})
    root: Node = None


class SimulatedGame(object):
    def __init__(self, game, deal, table_idx, visit_fractions=None):
        self.game = game
        self._current_deal = copy.deepcopy(deal)
        self.child_visit_fractions = visit_fractions or []
        self.table_idx = table_idx

    @classmethod
    def from_played_game(cls, game, played_game, table_idx):
        deal, cvf = data_from_played_game(game, played_game)
        return SimulatedGame(game, deal, table_idx, visit_fractions=cvf)

    def is_finished(self, config):
        return (self.current_deal().is_final() or
                self.current_deal().num_actions() >= config.max_moves)

    def apply(self, action_idx):
        new_deal = self.game.execute_action_ids(self.current_deal(), [action_idx])
        self._current_deal = new_deal

    def next_side_to_act(self):
        return self.current_deal().next_to_act_index() % 2

    def current_deal(self):
        return self._current_deal

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        try:
            self.child_visit_fractions.append({k: c.visit_count / sum_visits
                for k, c in root.children.items()
            })
        except:
            pdb.set_trace()
        logging.vlog(1, "%s", self.child_visit_fractions[-1])


def played_game_from_sim(sim):
    played_game = sim.game.played_game_from_deal(sim.current_deal())
    for i, cvf in enumerate(sim.child_visit_fractions):
        mvf = [0.0] * len(sim.game.actions)
        for j, vf in cvf.items():
            mvf[j] = vf
        played_game.actions[i].mcts_visit_fraction.extend(mvf)
    return played_game


def data_from_played_game(game, played_game):
    deal = game.deal_from_played_game(played_game)
    child_visit_fractions = []
    for a in played_game.actions:
        cvf = {}
        for j, vf in enumerate(a.mcts_visit_fraction):
            if vf > 0:
                cvf[j] = vf
        child_visit_fractions.append(cvf)
    return deal, child_visit_fractions


def make_played_board(sims):
    played_games = []
    for sim in sims:
        played_games.append(played_game_from_sim(sim))
    played_board = alphabridge_pb2.PlayedBoard(tables=played_games)
    return played_board


async def run_simulate(config, inference_pipes, game, board_io):
    saved_games = board_io.get_saved_games()
    await simulate_one_board(config, inference_pipes, game, board_io, saved_games)
    i = 1
    while True:
        logging.info("simulated {} boards".format(i))
        await simulate_one_board(config, inference_pipes, game, board_io)
        i += 1


async def simulate_one_board(config, inference_pipes, game, board_io, saved_games=None):
    deal = game.random_deal()
    sims = [SimulatedGame(game, deal, table_idx) for table_idx in range(config.num_tables)]
    played_sims = [None] * len(sims)
    while any(sim is None for sim in played_sims):
        tasks = [asyncio.create_task(
            play_one_table(config, inference_pipes, game, i, sims[i], board_io))
            for i, played_sim in enumerate(played_sims) if played_sim is None]
        new_sims = await asyncio.gather(*tasks)
        for sim in new_sims:
            if sim is not None:
                played_sims[sim.table_idx] = sim

    played_board = make_played_board(played_sims)
    game.score_played_board(played_board)
    if played_board.tables[0].actions == played_board.tables[1].actions:
        logging.warning("identical play at both tables")
    if logging.vlog_is_on(5):
        for i, sim in enumerate(played_sims):
            logging.vlog(5, f"==== table {i}\n{sim.current_deal()}")
    board_io.finish_board(played_board)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
async def play_one_table(config, inference_pipes, game, table_number, sim, board_io):
    if table_number % 2 == 1:
        inference_pipes = inference_pipes[::-1]
    while not sim.is_finished(config):
        side_idx = sim.next_side_to_act()
        action_idx, root = await run_mcts(
              config, sim, inference_pipes[side_idx], game)
        if action_idx is None:
            # This can happen if we explore only illegal moves.
            # This is rare, so we don't mind the occasional loss of time.
            logging.warning("abandoning board")
            return None
        logging.vlog(4, "search tree:\n%s", root)
        sim.apply(action_idx)
        sim.store_search_statistics(root)
        played_game = played_game_from_sim(sim)
        logging.debug("took action #%d", sim.current_deal().num_actions())
        board_io.update_table(sim.table_idx, played_game)
    return sim


@dataclass
class InferenceItem:
    features: dict
    action_id_path: list
    action_idx: int
    search_path: object
    node: Node
    scores: dict
    pipe: object  # alphabridge_pb2.sth
    game: object


async def run_inference(q):
    while True:
        try:
            item = await q.get()
        except:
            break
        logging.vlog(4, "requesting prediction for %s", item.action_id_path)
        predictions = await item.pipe.Predict(item.features, wait_for_ready=True,
              timeout=600)
        logging.vlog(3, "received prediction for path %s", item.action_id_path)
        process_predictions(predictions, item.action_id_path, item.action_idx,
                            item.search_path, item.node, item.scores, item.game)
        q.task_done()

# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
#
# First, however, we must choose a baseline outcome agains which to compare.
# We keep track of all predicted outcomes encountered,
# find the median score, and select the most common outcome with that score.
#
# Since one of `value_gt` and `value_geq` changes,
# we keep one mcts tree per score
# To prevent getting stuck, we calculate median score globally.
async def run_mcts(config, simulation, inference_pipe, game):
    scores: Mapping[int, ScoreInfo] = {}
    q = asyncio.Queue(10)
    inference_task = asyncio.create_task(run_inference(q))

    first_root, prior_par_outcome, score = await init_root(
            config, simulation, inference_pipe, game)
    scores[score] = ScoreInfo(outcome_count=1, outcomes={}, root=first_root)
    scores[score].outcomes[tuple(prior_par_outcome)] = 1

    tokenizer = bridgegame.Tokenizer(game)
    for i in range(config.num_simulations_per_move):
        score_info, prior_par_outcome, score = choose_score(scores)
        if not score_info.root:
            score_info.root = copy.deepcopy(first_root)
        root = score_info.root
        node, action_idx = root, None
        search_path = [node]

        while True:
            if node.legal_actions is None:
                node.legal_actions = np.array(
                        game.possible_action_indices(node.deal))
            if len(search_path) >= config.max_moves:
                break
            if node.deal.is_final():
                break
            if node.policy is None:
                # this is possible when we revisit a child before processing preds
                break

            #print(search_path)
            action_idx = select_child(config, node)
            if action_idx not in node.children:
                break
            node = node.children[action_idx]
            search_path.append(node)
            action_idx = None

        if action_idx is not None:
            action_id_path = [n.last_action for n in search_path[1:]] + [action_idx]
            utokens = tokenizer.tokenize_action_ids(action_id_path)
            logging.vlog(2, "exploring[%d] %s", i, action_id_path)
            action_chord = alphabridge_pb2.Chord(micro_token_id=utokens[-1])
            view_chords = [alphabridge_pb2.Chord(micro_token_id=c)
                           for c in tokenizer.tokenize_view(node.deal)]
            queries = [alphabridge_pb2.Chord(micro_token_id=c)
                       for c in tokenizer.tokenize_action_ids(node.legal_actions)]
            par_outcome_chord=alphabridge_pb2.Chord(
                micro_token_id=tokenizer.tokens_to_ids([prior_par_outcome])[0])
            features = alphabridge_pb2.FeaturesMicroBatch(
                  view_chords=view_chords,
                  par_outcome=par_outcome_chord,
                  queries=queries,
                  )
            #print(f"creating child {action_idx} for {node}")
            child_deal = copy.deepcopy(node.deal)
            game.execute_action_ids(child_deal, [action_idx])

            if action_idx in node.children:
                pdb.set_trace()
            child_node = Node(child_deal, action_idx)
            prune_illegal_actions(child_node)
            node.children[action_idx] = child_node
            search_path.append(child_node)
            backpropagate_virtual(search_path)
            await q.put(InferenceItem(
                features=features, action_id_path=action_id_path, action_idx=action_idx,
                search_path=search_path, node=node, scores=scores, pipe=inference_pipe, game=game))
        elif node.deal.is_final():
            actual_score = game.table_score(node.deal.result, node.deal.vulnerability)
            actual_score = (actual_score[0] or 0) - (actual_score[1] or 0)
            value_gt = 1. * (actual_score > score)
            value_geq = 1. * (actual_score >= score)
            value = 0.5 * (value_gt + value_geq)
            backpropagate(search_path, value)
        elif len(search_path) >= config.max_moves:
            value_gt = 0.
            value_geq = 1.
            value = 0.5 * (value_gt + value_geq)
            backpropagate(search_path, value)
        else:
            pass
        action_idx = None
  
    logging.vlog(2, "queue size before join: %d", q.qsize())
    await q.join()
    prune_illegal_actions(root)
    return select_action(config, simulation, root), root


def process_predictions(predictions, action_id_path, action_idx, search_path, node, scores, game):
    tokenizer = bridgegame.Tokenizer(game)
    value_geq = predictions.prediction[0].value_geq
    value_gt = predictions.prediction[0].value_gt
    # TODO: use logit to constrain values between (0,1).
    if value_gt < 0:
        value_gt = 0
    if value_gt > 1:
        value_gt = 1
    if value_geq < 0:
        value_geq = 0
    if value_geq > 1:
        value_geq = 1
    value = 0.5 * (value_gt + value_geq)
    par_outcome_tokens = tokenizer.ids_to_tokens(
            [predictions.prediction[0].par_outcome.micro_token_id])[0]
    try:
        par_outcome_event = bridgegame.Event.from_tokens(par_outcome_tokens)
    except:
        pdb.set_trace()
    policy_probs = np.array(predictions.prediction[0].policy)
    child_node = search_path[-1]
    child_node.policy = policy_probs

    score = game.table_score(par_outcome_event, node.deal.vulnerability)
    score = (score[0] or 0) - (score[1] or 0)
    increment_outcome(scores, score, par_outcome_tokens)
    backpropagate_actual(search_path, value)


def increment_outcome(scores, score, outcome_tokens):
    key = tuple(outcome_tokens)
    if score not in scores:
        scores[score] = ScoreInfo(outcome_count=1)
    else:
        scores[score].outcome_count += 1
    if key not in scores[score].outcomes:
        scores[score].outcomes[key] = 1
    else:
        scores[score].outcomes[key] += 1


async def init_root(config, simulation, inference_pipe, game):
    deal = simulation.current_deal()
    view = simulation.game.kibitzer_view(deal, deal.num_actions())
    tokenizer = bridgegame.Tokenizer(game)
    logging.vlog(7, "%s", view)
    view_chords = [alphabridge_pb2.Chord(micro_token_id=c)
                   for c in tokenizer.tokenize_view(view)]
    legal_action_ixs = game.possible_action_indices(view)
    queries = [alphabridge_pb2.Chord(micro_token_id=c)
               for c in tokenizer.tokenize_action_ids(legal_action_ixs)]
    features = alphabridge_pb2.FeaturesMicroBatch(
            view_chords=view_chords,
            queries=queries)
    logging.vlog(2, "requesting root prediction")
    predictions = await inference_pipe.Predict(features, wait_for_ready=True,
            timeout=600)
    logging.vlog(2, "received root prediction")
    policy_probs = predictions.prediction[0].policy

    root = Node(deal, None)
    root.policy = np.array(policy_probs)
    root.visit_count = 1

    add_exploration_noise(config, root)
    deprecate_illegal_actions(simulation, root)
    outcome_tokens = tokenizer.ids_to_tokens([
        predictions.prediction[0].par_outcome.micro_token_id])[0]
    outcome_event = bridgegame.Event.from_tokens(outcome_tokens)
    score = game.table_score(outcome_event, view.vulnerability)
    return root, outcome_tokens, (score[0] or 0) - (score[1] or 0)


def choose_score(scores):
    total_outcomes = sum([v.outcome_count for _, v in scores.items()])
    outcomes_so_far = 0
    for k in sorted(scores.keys()):
        outcomes_so_far += scores[k].outcome_count
        if outcomes_so_far * 2 >= total_outcomes:
            score_info, score = scores[k], k
    best_v = 0
    for k, v in score_info.outcomes.items():
        if v > best_v:
            best_v = v
            best_k = k
    return score_info, best_k, score


def deprecate_illegal_actions(simulation, root):
    deal = simulation.current_deal()
    root.legal_actions = np.array(simulation.game.possible_action_indices(deal))
    for a in range(len(root.policy)):
        if a not in root.legal_actions:
            root.policy[a] = 0.


def prune_illegal_actions(root):
    root.children = {action_idx: child for action_idx, child in root.children.items()
        if action_idx in root.legal_actions}


def select_action(config, simulation, root):
    action_ids = [k for k, v in root.children.items()]
    visit_counts = [v.visit_count for k, v in root.children.items()]
    # TODO: add policy bias
    if logging.vlog_is_on(9):
      logging.debug(f"visit counts = {visit_counts}")
    best = np.argmax(visit_counts)
    return action_ids[best]


def select_child(config, node):
    """Select the child based on UCB score."""
    # NOTE: assembling values and visit_counts is >50% of CPU usage of selfplay.
    # TODO(njt): store these values in np.array in parent if optimizing.


    visit_counts = np.zeros(len(node.legal_actions))
    values = np.zeros(len(node.legal_actions), dtype=np.float32)
    for i, c in node.children.items():
        if i not in node.legal_actions:
            pdb.set_trace()
        ii, = np.where(node.legal_actions == i)
        visit_counts[ii] = c.visit_count
        values[ii] = c.value()

    legal_policy = np.zeros(len(node.legal_actions))
    for i in range(len(node.legal_actions)):
        legal_policy[i] = node.policy[i]

    scores = ucb_score(config, node, legal_policy, values, visit_counts)
    if logging.vlog_is_on(9):
        logging.debug(f"values = {values} policy = {node.policy} scores = {scores}")
    m = np.argmax(scores)
    return node.legal_actions[m]


def ucb_score(config, parent_node, policy, value, visit_count):
    """Upper confidence bound score.

    ucb_score for a node is based on its value,
    plus an exploration bonus based on the prior.
    """
    pb_c_base = np.log((parent_node.visit_count + config.ucb_pb_c_base + 1) /
        config.ucb_pb_c_base) + config.ucb_pb_c_init
    pb_c = pb_c_base * np.sqrt(parent_node.visit_count) / (visit_count + 1)
    prior_score = pb_c * policy
    return prior_score + value


# At the end of a simulation, we propagate the value up the tree to its root.
def backpropagate(search_path, value):
    #print(f"backprop_s: search_path[-1] = {search_path[-1]}")
    for i, node in enumerate(search_path):
        node.visit_count += 1
        if i > 0 and search_path[i-1].deal.next_to_act_index() % 2 == 1:
            node.value_sum += 1 - value
        else:
            node.value_sum += value
    #print(f"backprop_f: search_path[-1] = {search_path[-1]}")

def backpropagate_virtual(search_path):
    #print(f"backprop_vs: search_path[-1] = {search_path[-1]}")
    for node in search_path:
        node.visit_count += 1
        # FIXME: make virtual loss work for both sides
        # note that we want to change to 1 - value when LAST to act side is E-W.
        # also note that we don't care about root value.
    #print(f"backprop_vf: search_path[-1] = {search_path[-1]}")

def backpropagate_actual(search_path, value):
    #print(f"backprop_as: search_path[-1] = {search_path[-1]}")
    for i, node in enumerate(search_path):
        if i > 0 and search_path[i-1].deal.next_to_act_index() % 2 == 1:
            node.value_sum += 1 - value
        else:
            node.value_sum += value
    #print(f"backprop_af: search_path[-1] = {search_path[-1]}")


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config, node):
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(node.policy))
    frac = config.root_exploration_fraction
    if logging.vlog_is_on(9):
        logging.debug(f"noise = {noise} frac = {frac}")
        logging.debug(f"before, policy = {node.policy}")
    node.policy = node.policy * (1 - frac) + noise * frac
    if logging.vlog_is_on(9):
        logging.debug(f"after, policy = {node.policy}")


FLAGS = flags.FLAGS


flags.DEFINE_integer("n", 2, "number of card ranks")
flags.DEFINE_integer(
    "num_parallel_inferences", 1, "Number of inferences to run in parallel for selfplay.")

flags.DEFINE_integer("num_simulations_per_move", 800,
    "Number of MCTS simulations per move in a game.")

flags.DEFINE_string("replay_buffer_address", "localhost:10000",
    "Address of replay buffer service")

flags.DEFINE_string("inference_pipe_address", "localhost:20000",
    "Address of inference pipe service")

flags.DEFINE_integer("concurrency", 10, "number of concurrent simulations")

flags.DEFINE_integer("parallelism", 1, "number of parallel processes")


class Saver:
    def __init__(self, board_writer, table, column_family, column_template,
            num_tables, row_keys, clear_rows=True):
        self.board_writer = board_writer
        self.table = table
        self.column_family = column_family
        self.column_template = column_template
        self.num_tables = num_tables
        self.row_keys = row_keys
        self.clear_rows = clear_rows
        self.q = queue.Queue()
        self.scanned = False

    def get_saved(self, row_key):
        if isinstance(row_key, str):
            row_key = row_key.encode('utf8')
        if not self.scanned:
            self.saved = self.scan()
            self.scanned = True
        if row_key in self.saved:
            return self.saved[row_key]
        elif self.table:
            logging.info("did not find saved games for %s", row_key)
        return None

    def request_start_board(self, row_key, played_game):
        self.q.put((self.do_start_board, (row_key, played_game)))

    def request_update_table(self, row_key, table_idx, played_game):
        self.q.put((self.do_update_table, (row_key, table_idx, played_game)))
        #logging.debug("request_update_table: queue size = %d", self.q.qsize())

    def request_finish_board(self, row_key, played_board):
        self.q.put((self.do_finish_board, (row_key, played_board)))
        logging.debug("request_finish_board: queue size = %d", self.q.qsize())

    def run(self):
        while True:
            try:
                fn, args = self.q.get()
                #logging.vlog(1, "doing %s", fn)
                fn(*args)
            except Exception as err:
                logging.fatal("run() caught exception: %s", err)

    def do_start_board(self, row_key, played_game):
        if self.table:
            row = self.table.row(row_key)
            ser = played_game.SerializeToString()
            for table_idx in range(self.num_tables):
                column = self.column_template.format(table_idx)
                row.set_cell(self.column_family, column, ser)
            row.commit()

    def do_update_table(self, row_key, table_idx, played_game):
        if self.table:
            row = self.table.row(row_key)
            ser = played_game.SerializeToString()
            column = self.column_template.format(table_idx)
            row.set_cell(self.column_family, column, ser)
            row.commit()

    def do_finish_board(self, row_key, played_board):
        if self.table and self.clear_rows:
            row = self.table.row(row_key)
            row.delete()
            row.commit()
        self.board_writer.Put(played_board, wait_for_ready=True)

    def scan(self):
        return {}


class BoardSaver:
    def __init__(self, saver, row_key):
        self.saver = saver
        self.row_key = row_key

    def get_saved_games(self):
        return self.saver.get_saved(self.row_key)

    def start_board(self, played_game):
        self.saver.request_start_board(self.row_key, played_game)

    def update_table(self, table_idx, played_game):
        self.saver.request_update_table(self.row_key, table_idx, played_game)

    def finish_board(self, played_board):
        self.saver.request_finish_board(self.row_key, played_board)


async def run_simulations(subshard_idx):
    config = alphabridge_pb2.SimulationConfig(
        num_tables=2,
        max_moves=150,
        num_simulations_per_move=FLAGS.num_simulations_per_move,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        ucb_pb_c_base=19652,
        ucb_pb_c_init=1.25,
        num_parallel_inferences=FLAGS.num_parallel_inferences)

    logging.info("connecting to inference pipe at %s", FLAGS.inference_pipe_address)
    pipe_channel = grpc.aio.insecure_channel(FLAGS.inference_pipe_address)
    pipe = alphabridge_pb2_grpc.InferencePipeStub(pipe_channel)
    logging.info("connecting to replay buffer at %s", FLAGS.replay_buffer_address)
    replay_buffer_channel = grpc.insecure_channel(FLAGS.replay_buffer_address)
    replay_buffer = alphabridge_pb2_grpc.ReplayBufferStub(replay_buffer_channel)

    game = bridgegame.Game(n=FLAGS.n)

    table = None
    row_keys = [f"{i:05d}" for i in range(FLAGS.concurrency)]
    ix1 = (len(row_keys) * subshard_idx) // FLAGS.parallelism
    ix2 = (len(row_keys) * (1 + subshard_idx)) // FLAGS.parallelism
    row_keys = row_keys[ix1:ix2]
    saver = Saver(replay_buffer, table, "bigtable_column_family",
            "bigtable_column_template", config.num_tables, row_keys)

    logging.info("starting io thread")
    io_thread = threading.Thread(target=lambda: saver.run(), daemon=True)
    io_thread.start()

    logging.info("creating %d concurrent simulation tasks", len(row_keys))
    tasks = [asyncio.create_task(
        run_simulate(config, (pipe, pipe), game, BoardSaver(saver, row_key)))
        for row_key in row_keys]
    logging.info("gathering...")
    await asyncio.gather(*tasks)


def async_run(subshard_idx):
    asyncio.run(run_simulations(subshard_idx))

def main(_):
    if FLAGS.parallelism == 1:
        async_run(0)
    else:
        procs = []
        for i in range(FLAGS.parallelism):
            p = multiprocessing.Process(target=async_run, args=(i,))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


if __name__ == "__main__":
    absl.app.run(main)
