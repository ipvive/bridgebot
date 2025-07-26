from __future__ import division

import asyncio
import copy
import multiprocessing
import os
import queue
import random
import textwrap
import threading

import absl.app
import absl.flags as flags
import absl.logging as logging
import grpc
import numpy as np
from google.cloud import bigtable

import bridgebot.bridge.game as bridgegame
from bridgebot.bridge import tokens
import bridgebot.pb.alphabridge_pb2 as alphabridge_pb2
import bridgebot.pb.alphabridge_pb2_grpc as alphabridge_pb2_grpc


class Node(object):
  def __init__(self, seat_to_act, last_action):
    self.policy = None
    self.visit_count = 0
    self.value_sum = 0
    self.seat_to_act = seat_to_act
    self.children = {}
    self.legal_actions = None
    self.last_action = last_action

  def has_children(self):
    return len(self.children) > 0

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

  def __repr__(self):
    s = ""
    s += f"[...,{self.last_action}] vis={self.visit_count} val={self.value()} nta={self.seat_to_act}\n"
    s += f"children: ({self.children.keys()})\n"
    for ix, child in self.children.items():
      s += textwrap.indent(child.__repr__(), "    ")
    return s


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
    new_deal = self.game.execute_action_index(self.current_deal(), action_idx)
    self._current_deal = new_deal

  def next_side_to_act(self):
    return self.current_deal().next_action_verb_index() % 2

  def current_deal(self):
    return self._current_deal

  def store_search_statistics(self, root):
    sum_visits = sum(child.visit_count for child in root.children.values())
    self.child_visit_fractions.append({k: c.visit_count / sum_visits
        for k, c in root.children.items()
    })
    logging.vlog(1, "%s", self.child_visit_fractions[-1])


def played_game_from_sim(sim):
  played_game = sim.game.played_game_from_deal(sim.current_deal())
  for i, cvf in enumerate(sim.child_visit_fractions):
    mvf = [0.0] * bridgegame.num_actions
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


async def run_simulate(config, inference_pipes, game, rng, board_io):
  saved_games = board_io.get_saved_games()
  await simulate_one_board(config, inference_pipes, game, rng, board_io, saved_games)
  i = 1
  while True:
    logging.info("simulated {} boards".format(i))
    await simulate_one_board(config, inference_pipes, game, rng, board_io)
    i += 1


async def simulate_one_board(config, inference_pipes, game, rng, board_io, saved_games=None):
  if not saved_games:
    deal = game.random_deal(rng)
    sims = [SimulatedGame(game, deal, table_idx) for table_idx in range(config.num_tables)]
    board_io.start_board(game.played_game_from_deal(deal))
  else:
    sims = [SimulatedGame.from_played_game(game, played_game, i)
            for i, played_game in enumerate(saved_games)]
    logging.info("Restored partially-played deals with %s actions",
            [sim.current_deal().num_actions() for sim in sims])

  played_sims = [None] * len(sims)
  while any(sim is None for sim in played_sims):
    tasks = [asyncio.create_task(
          play_one_table(config, inference_pipes, i, sims[i], rng, board_io))
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
  pdb.set_trace()
  board_io.finish_board(played_board)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
async def play_one_table(config, inference_pipes, table_number, sim, rng, board_io):
  if table_number % 2 == 1:
    inference_pipes = inference_pipes[::-1]
  while not sim.is_finished(config):
    side_idx = sim.next_side_to_act()
    action_idx, root = await run_mcts(config, sim, inference_pipes[side_idx], rng)
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


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
async def run_mcts(config, simulation, inference_pipe, rng):
  deal = simulation.current_deal()
  view = simulation.game.actor_view(deal, deal.num_actions())
  logging.vlog(7, "%s", view)
  contract_seat = deal.contract_seat_index()

  view_tokens = tokens.Tokenizer().tokenize_view(view, rng)
  features = alphabridge_pb2.FeaturesMicroBatch(
          view_token=view_tokens,
          action_path=[alphabridge_pb2.ActionIndexPath()] * config.num_parallel_inferences)
  logging.vlog(2, "requesting root prediction")
  predictions = await inference_pipe.Predict(features, wait_for_ready=True,
          timeout=600)
  logging.vlog(2, "received root prediction")
  value = predictions.prediction[0].value
  policy_probs = predictions.prediction[0].policy

  root = Node(deal.next_action_verb_index(), None)
  root.policy = np.array(policy_probs)
  root.visit_count = 1
  if FLAGS.policy_only:
    deprecate_illegal_actions(simulation, root)
    return np.argmax(root.policy), root

  add_exploration_noise(config, root)
  deprecate_illegal_actions(simulation, root)

  # TODO: modify loop
  #   gen 10 paths 10 virtual children
  #   run batch through inference
  #   replace virtual childen with real ones.

  for i in range(config.num_simulations_per_move):
    node, action_idx = root, None
    search_path = [node]

    while len(search_path) < config.max_moves:
      is_authorized_actor = root.seat_to_act == node.seat_to_act
      if simulation.game.all_same_side_for_index(
          contract_seat, root.seat_to_act, node.seat_to_act):
        is_authorized_actor = True
      action_idx = select_child(config, node, is_authorized_actor)
      if action_idx not in node.children:
        break
      node = node.children[action_idx]
      search_path.append(node)

    action_path = [n.last_action for n in search_path[1:]] + [action_idx]
    logging.vlog(2, "exploring[%d] %s", i, action_path)
    features = alphabridge_pb2.FeaturesMicroBatch(
            view_token=view_tokens,
            action_path=[alphabridge_pb2.ActionIndexPath(action_index=action_path)] *\
               config.num_parallel_inferences)
    logging.vlog(4, "requesting prediction for %s", action_path)
    predictions = await inference_pipe.Predict(features, wait_for_ready=True,
            timeout=600)
    next_to_act = predictions.prediction[0].next_to_act
    value = predictions.prediction[0].value
    logging.vlog(3, "received prediction for path %s: NTA=%d value=%f", action_path, next_to_act, value)
    policy_probs = np.array(predictions.prediction[0].policy)

    child_node = Node(next_to_act, action_idx)
    child_node.policy = policy_probs
    node.children[action_idx] = child_node
    search_path.append(child_node)
    backpropagate(simulation.game, search_path, value, root.seat_to_act)

  prune_illegal_actions(simulation, root)
  return select_action(config, simulation, root), root


def deprecate_illegal_actions(simulation, root):
  deal = simulation.current_deal()
  root.legal_actions = frozenset(simulation.game.possible_action_indices(deal))
  for a in range(len(root.policy)):
      if a not in root.legal_actions:
          root.policy[a] = 0.


def prune_illegal_actions(simulation, root):
  root.children = {action_idx: child for action_idx, child in root.children.items()
      if action_idx in root.legal_actions}


def select_action(config, simulation, root):
  visit_counts = [(child.visit_count, action_idx)
                  for action_idx, child in root.children.items()]
  # TODO: add policy bias
  if logging.vlog_is_on(9):
    logging.debug(f"visit counts = {visit_counts}")
  _, action_idx = softmax_sample(visit_counts)
  return action_idx


def select_child(config, node, is_authorized_actor):
  """Select the child based on UCB score or weighted sampling."""
  # NOTE: assembling values and visit_counts is >50% of CPU usage of selfplay.
  # TODO(njt): store these values in np.array in parent if optimizing.
  visit_counts = np.zeros(len(node.policy))
  for i, c in node.children.items():
    visit_counts[i] = c.visit_count

  if is_authorized_actor:
    values = np.zeros(len(node.policy), dtype=np.float32)
    for i, c in node.children.items():
      values[i] = c.value()
    # TODO: inputs some_fn(inputs, virtual_descendant_count)
    #        maybe:
    #          visit_counts += virtual_visit_count
    #          values += virtual_visit_count * [-config.virtual_penalty]
    min_value, max_value = values.min(), values.max()
    if min_value < max_value:
        values = (values - min_value) / (max_value - min_value)
    scores = ucb_score(config, node, node.policy, values, visit_counts)
    if logging.vlog_is_on(9):
      logging.debug(f"values = {values} policy = {node.policy} scores = {scores}")
    m = np.argmax(scores)
    return m

  else:
    return incremental_random_sample(node.policy, visit_counts) # TODO + virtual_counts


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


def incremental_random_sample(policy, visit_counts):
  s = visit_counts.sum()
  p = policy / policy.sum()
  if s > 0:
    visit_freq = visit_counts / (s + 1)
    p = np.maximum(p - visit_freq, 0)
    p = p / p.sum()
  return np.random.choice(len(p), p=p)


# At the end of a simulation, we propagate the value up the tree to its root.
def backpropagate(game, search_path, value, root_actor_index):
  if root_actor_index % 2 == 1:
    value = -value
  for node in search_path:
    node.value_sum += value
    node.visit_count += 1


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


def softmax(logits):
  m = np.max(logits)
  e = np.exp(logits - m)
  p = e / e.sum()
  return p


def softmax_sample(a):
  if len(a) == 0:
    return None, None
  p = softmax([c for c, n in a])
  ax = np.random.choice(len(p), p=p)
  return a[ax]


FLAGS = flags.FLAGS


flags.DEFINE_integer(
    "num_parallel_inferences", 1, "Number of inferences to run in parallel for selfplay.")

flags.DEFINE_integer("num_simulations_per_move", 800,
    "Number of MCTS simulations per move in a game.")

flags.DEFINE_string("replay_buffer_address", "localhost:10000",
    "Address of replay buffer service")

flags.DEFINE_string("inference_pipe_address", "localhost:20000",
    "Address of inference pipe service")

flags.DEFINE_integer("concurrency", 100, "number of concurrent simulations")

flags.DEFINE_integer("parallelism", 1, "number of parallel processes")

flags.DEFINE_string("bigtable_instance", None, "bigtable instance id for checkpoints")

flags.DEFINE_string("bigtable_table", "simulation-checkpoint",
        "bigtable table id for checkpoints")

flags.DEFINE_string("bigtable_column_family", "b",
        "bigtable column_family id for checkpoints")

flags.DEFINE_string("bigtable_column_template", "t{:02d}",
        "bigtable column id template for checkpoints")

flags.DEFINE_string("shard_id", "TESTONLY", "shard identity name")

flags.DEFINE_bool("policy_only", False, "no mcts just follow policy")


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
        logging.debug("request_update_table: queue size = %d", self.q.qsize())

    def request_finish_board(self, row_key, played_board):
        self.q.put((self.do_finish_board, (row_key, played_board)))
        logging.debug("request_finish_board: queue size = %d", self.q.qsize())

    def run(self):
        while True:
            try:
                fn, args = self.q.get()
                logging.vlog(1, "doing %s", fn)
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
        saved = {}
        if self.table:
            logging.info("starting scan")
            row_gen = self.table.read_rows(start_key=self.row_keys[0],
                    end_key=self.row_keys[-1], end_inclusive=True)
            for row in row_gen:
                try:
                    tables = []
                    for table_idx in range(self.num_tables):
                        column = self.column_template.format(table_idx).encode(
                                'utf8')
                        ser = row.cells[self.column_family][column][0].value
                        played_game = alphabridge_pb2.PlayedGame()
                        played_game.ParseFromString(ser)
                        tables.append(played_game)
                    saved[row.row_key] = tables
                except:
                    logging.warning("problem reading row for key %s", row_key)
            logging.info("finished scan")
        return saved


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

    game = bridgegame.Game()
    rng = random.Random()

    if FLAGS.bigtable_instance:
        client = bigtable.Client()
        instance = client.instance(FLAGS.bigtable_instance)
        table = instance.table(FLAGS.bigtable_table)
    else:
        table = None
    row_keys = [f"{FLAGS.shard_id}:{i:05d}" for i in range(FLAGS.concurrency)]
    ix1 = (len(row_keys) * subshard_idx) // FLAGS.parallelism
    ix2 = (len(row_keys) * (1 + subshard_idx)) // FLAGS.parallelism
    row_keys = row_keys[ix1:ix2]
    saver = Saver(replay_buffer, table, FLAGS.bigtable_column_family,
            FLAGS.bigtable_column_template, config.num_tables, row_keys)

    logging.info("starting io thread")
    io_thread = threading.Thread(target=lambda: saver.run(), daemon=True)
    io_thread.start()

    logging.info("creating %d concurrent simulation tasks", len(row_keys))
    tasks = [asyncio.create_task(
        run_simulate(config, (pipe, pipe), game, rng, BoardSaver(saver, row_key)))
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
