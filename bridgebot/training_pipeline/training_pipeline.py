import apache_beam as beam
import grpc
import io
import math
import numpy as np
import os.path
import random
import tensorflow as tf
import tensorflow_datasets as tfds
import re

from bridge import lin
from pb import alphabridge_pb2
from pb import alphabridge_pb2_grpc
import bridge.game as bridgegame
import bridge.tokens
from mubert import toptokens
from jargon.bert import tokenization as jargon_tokenization


def _pad(v, n, fill):
    if len(v) >= n:
        return v[:n]
    else:
        return v + [fill] * (n - len(v))


def make_masked(view_tokens, masked_prob, max_seq_length,
                max_predictions_per_seq, rng):
    """ Returns masked_tokens, masked indices, and correct labels for masked tokens. """
    masked_tokens = list(view_tokens)
    masked_positions = []
    masked_labels = []

    num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(masked_tokens) * masked_prob))))

    cand_indexes = list(range(min(len(masked_tokens), max_seq_length)))
    rng.shuffle(cand_indexes)

    for index in cand_indexes:
        if len(masked_labels) >= num_to_predict:
            break
        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
              masked_token = view_tokens[index]
            # 10% of the time, replace with random word
            else:
              masked_token = rng.choice(bridgegame.all_tokens.tokens)

        masked_tokens[index] = masked_token
        masked_positions.append(index)
        masked_labels.append(view_tokens[index])

    return (masked_tokens, masked_positions, masked_labels)


def _position_to_features(position, max_seq_length, max_predictions_per_seq,
        num_lookahead_steps, masked_prob, rng):
  """Converts an alphabridge_pb2.PlayedBoardPosition into a features dict."""
  game_obj = bridgegame.Game()
  tokenizer = bridge.tokens.Tokenizer()

  played_game = position.board.tables[position.table_index]
  deal = game_obj.deal_from_played_game(played_game)
  view = game_obj.actor_view(deal, position.action_index)
  view_tokens = tokenizer.tokenize_view(view, rng)
  (input_tokens, masked_positions, masked_labels) = make_masked(
      view_tokens, masked_prob, max_seq_length, max_predictions_per_seq, rng)
  score_token = tokenizer.tokenize_score(played_game.result.comparison_score)

  target_actor_tokens = []
  target_action_tokens = []
  target_policies = []
  target_mask = []
  for i in range(num_lookahead_steps + 1):
    n = position.action_index + i
    if n < deal.num_actions():
      event_tokens = tokenizer.tokenize_event(deal.action(n))
      target_actor_tokens.append(event_tokens[0])
      target_action_tokens.append(event_tokens[1])
      policy = list(played_game.actions[n].mcts_visit_fraction)
      if not policy:
          action_index = bridgegame._actions.index[event_tokens[1]]
          policy = [0.0] * bridgegame.num_actions
          policy[action_index] = 1.0
      target_policies.append(policy)
      target_mask.append(1.0)
    else:
      target_policies.append([0.0] * bridgegame.num_actions)

  d = {
      "input_ids": _pad(
        tokenizer.tokens_to_ids(input_tokens), max_seq_length, 0),
      "masked_positions": _pad(masked_positions, max_predictions_per_seq, 0),
      "masked_ids": _pad(
        tokenizer.tokens_to_ids(masked_labels), max_predictions_per_seq, 0),
      "masked_weights": _pad(
        [1.0] * len(masked_labels), max_predictions_per_seq, 0.0),
      "actions": _pad(
        tokenizer.tokens_to_ids(target_action_tokens),
        num_lookahead_steps + 1, 0),
      "target_actors": _pad(
        tokenizer.tokens_to_ids(target_actor_tokens),
        num_lookahead_steps + 1, 0),
      "target_policies": target_policies,
      "target_score": tokenizer.tokens_to_ids([score_token])[0],
  }
  return d


def _features_from_played_position_builder(config):
  rng = random.Random()
  def _features_from_played_position(ser_position):
    position = alphabridge_pb2.PlayedBoardPosition()
    position.ParseFromString(ser_position)
    return (position.board.board_id.source_uri,
            _position_to_features(position, config.max_seq_length,
              config.max_predictions_per_seq, config.num_lookahead_steps,
              config.masked_prob, rng))
  return _features_from_played_position


def _position_and_text_to_top_features(position, corpus_text,
      max_player_seq_length, max_jargon_seq_length,
      max_top_seq_length, jargon_vocab_file,
      jargon_do_lower_case, redact_commentator_names, rng):
  """Converts an alphabridge_pb2.PlayedBoardPosition and text into a top features dict."""
  game_obj = bridgegame.Game()
  player_tokenizer = bridge.tokens.Tokenizer()
  jargon_tokenizer = jargon_tokenization.FullTokenizer(
    vocab_file=jargon_vocab_file,
    do_lower_case=jargon_do_lower_case)
  top_tokenizer = toptokens.Tokenizer()

  played_game = position.board.tables[position.table_index]
  deal = game_obj.deal_from_played_game(played_game)
  view = game_obj.actor_view(deal, position.action_index)
  view_tokens = player_tokenizer.tokenize_view(view, rng)

  comments = [a.kibitzer_comment or a.explanation
    for a in played_game.annotations if a.action_index == position.action_index]
  comment = rng.choice(comments)

  comment = re.sub(r'.*([1-7]@[sShHdDcc]).*', r'\1', comment)
  corpus_text = re.sub(r'.*([1-7]@[sShHdDcc]).*', r'\1', corpus_text)
  comment_tokens = jargon_tokenizer.tokenize(comment)
  corpus_tokens = jargon_tokenizer.tokenize(corpus_text)

  if redact_commentator_names:
    def redact_through_first_colon(tokens):
      if ':' in tokens:
        idx = tokens.index(':') + 1
        tokens[:idx] = ['[MASK]'] * idx
      return tokens
    comment_tokens = redact_through_first_colon(comment_tokens)
    corpus_tokens = redact_through_first_colon(corpus_tokens)

  top_query = ['[PROBLEM_1]', '[VARIANT_2]', '[IS_REAL]']
  target_response = ['[YES]', '[NO]']
  rng.shuffle(target_response)
  target_response = target_response[:1]
  target_positions = [2]
  target_weights = [1.0]

  if target_response[0] is '[YES]':
    jargon_tokens = ['[CLS]'] + comment_tokens + ['[SEP]']
  else:
    jargon_tokens = ['[CLS]'] + corpus_tokens + ['[SEP]']

  assert top_tokenizer.tokens_to_ids(['[PAD]'])[0] == 0

  d = {
      "player_input_ids": _pad(
        player_tokenizer.tokens_to_ids(view_tokens), max_player_seq_length, 0),
      "jargon_input_ids": _pad(
        jargon_tokenizer.convert_tokens_to_ids(jargon_tokens), max_jargon_seq_length, 0),
      "query_ids": _pad(
        top_tokenizer.tokens_to_ids(top_query), max_top_seq_length, 0),
      "target_ids": _pad(
        top_tokenizer.tokens_to_ids(target_response), max_top_seq_length, 0),
      "target_positions": _pad(target_positions, max_top_seq_length, 0),
      "target_weights": _pad(target_weights, max_top_seq_length, 0.0),
  }
  return d


def _top_features_builder(config):
  rng = random.Random()
  def _top_features_from_board_and_text(data):
    position = alphabridge_pb2.PlayedBoardPosition()
    position.ParseFromString(data['board'])
    return (position.board.board_id.source_uri,
            _position_and_text_to_top_features(position, data['text'],
              config.max_player_seq_length, config.max_jargon_seq_length,
              config.max_top_seq_length, config.jargon_vocab_file,
              config.jargon_do_lower_case, config.redact_commentator_names, rng))
  return _top_features_from_board_and_text


def _board_to_played_board(board, game, source_uri):
  board_id = alphabridge_pb2.BoardId(source_uri=source_uri)
  tables = [game.played_game_from_deal(t) for t in board.tables.values()]
  played_board = alphabridge_pb2.PlayedBoard(board_id=board_id, tables=tables)
  game.score_played_board(played_board)
  return played_board.SerializeToString()


class _lin_to_played_boards(beam.DoFn):
  def __init__(self):
    self.parser = lin.Parser()
    self.game = bridgegame.Game()
    cls = self.__class__
    self.parsed_boards = beam.metrics.Metrics.counter(cls, 'parsed boards')
    self.empty_boards = beam.metrics.Metrics.counter(cls, 'empty boards')
    self.early_boards = beam.metrics.Metrics.counter(cls, 'early boards')
    self.bad_board_number = beam.metrics.Metrics.counter(cls, 'bad board number')
    self.error_counters = {}
    self.match_number_matcher = re.compile(r"(\d+).lin$")

  def process(self, linlines):
    with io.StringIO(linlines.decode("utf-8")) as reader:
      filename = reader.readline().strip()
      m = self.match_number_matcher.search(filename)
      if m:
        n = int(m.group(1))
        if n < 25000:
          self.early_boards.inc()
          return
      else:
        self.bad_board_number.inc()
        return
      
      reader.name = filename
      boards, error_counts = self.parser.parse(reader, self.game)

      for name, count in error_counts.items():
        if name not in self.error_counters:
          self.error_counters[name] = beam.metrics.Metrics.counter(
              self.__class__, name)
        self.error_counters[name].inc(count)

      for k, board in boards.items():
        uri = "{}/{}".format(filename, k)
        num_actions = sum(table.num_actions()
            for table in board.tables.values())
        if num_actions > 0:
          self.parsed_boards.inc()
          yield _board_to_played_board(board, self.game, uri)
        else:
          self.empty_boards.inc()


def _add_random_key(num_keys=1<<14):
    def _add_key(item):
      return (random.randint(0, num_keys), item)
    return _add_key


def _select_comments_in_board(comment_regex):
    def _select_in_board(ser_played_board):
      played_board = alphabridge_pb2.PlayedBoard()
      played_board.ParseFromString(ser_played_board)
      for table in played_board.tables:
        selected_annotations = [annotation for annotation in table.annotations
          if comment_regex.search(annotation.kibitzer_comment)] 
        del table.annotations[:]
        table.annotations.extend(selected_annotations)
      return played_board.SerializeToString()
    return _select_in_board


class _select_comments(beam.DoFn):
  def __init__(self, comment_regex):
    cls = self.__class__
    self.dropped = beam.metrics.Metrics.counter(cls, 'dropped corpus comments')
    self.passed = beam.metrics.Metrics.counter(cls, 'selected corpus comments')
    self.comment_regex = comment_regex

  def process(self, text):
    if self.comment_regex.search(text):
      self.passed.inc()
      yield text
    else:
      self.dropped.inc()


class _split(beam.DoFn):
  def process(self, elem):
    _, group = elem
    for item in group:
      yield item


class _zip_boards_and_corpus(beam.DoFn):
  max_size = 256
  def __init__(self):
    self.corpus_queue = []
    cls = self.__class__
    self.empty_corpus_queue = beam.metrics.Metrics.counter(cls, 'empty corpus queue')

  def process(self, kv):
    boards = kv[1]['boards']
    corpus = kv[1]['corpus']
    if len(corpus) > 0:
      self.corpus_queue = corpus + self.corpus_queue
      size = min(self.max_size, len(self.corpus_queue))
      self.corpus_queue = self.corpus_queue[:size]
    for board in boards:
      if len(self.corpus_queue) > 0:
        text = self.corpus_queue.pop(0)
        yield {'board': board, 'text': text}
      else:
        self.empty_corpus_queue.inc()

 
def _features_info(config):
  return tfds.features.FeaturesDict({
    "input_ids": tfds.features.Tensor(
        dtype=tf.int32, shape=[config.max_seq_length]),
    "masked_positions": tfds.features.Tensor(
        dtype=tf.int32, shape=[config.max_predictions_per_seq]),
    "masked_ids": tfds.features.Tensor(
        dtype=tf.int32, shape=[config.max_predictions_per_seq]),
    "masked_weights": tfds.features.Tensor(
        dtype=tf.float32, shape=[config.max_predictions_per_seq]),
    "actions": tfds.features.Tensor(
        dtype=tf.int32, shape=[config.num_lookahead_steps + 1]),
    "target_actors": tfds.features.Tensor(
        dtype=tf.int32, shape=[config.num_lookahead_steps + 1]),
    "target_policies": tfds.features.Tensor(
        dtype=tf.float32, shape=[config.num_lookahead_steps + 1, config.num_actions]),
    "target_score": tfds.features.Tensor(
        dtype=tf.int32, shape=[]),
    })


def _top_features_info(config):
  return tfds.features.FeaturesDict({
    "player_input_ids": tfds.features.Tensor(
        dtype=tf.int32, shape=[config.max_player_seq_length]),
    "jargon_input_ids": tfds.features.Tensor(
        dtype=tf.int32, shape=[config.max_jargon_seq_length]),
    "query_ids": tfds.features.Tensor(
        dtype=tf.int32, shape=[config.max_top_seq_length]),
    "target_ids": tfds.features.Tensor(
        dtype=tf.int32, shape=[config.max_top_seq_length]),
    "target_positions": tfds.features.Tensor(
        dtype=tf.int32, shape=[config.max_top_seq_length]),
    "target_weights": tfds.features.Tensor(
        dtype=tf.float32, shape=[config.max_top_seq_length]),
    })


class SupervisedConfig(tfds.core.BuilderConfig):
  def __init__(self, name, train_input_files=None, eval_input_files=None,
      max_seq_length=256, max_predictions_per_seq=20,
      num_lookahead_steps=3, num_actions=90, masked_prob=0.05, **kwargs):
    super().__init__(name=name, version='0.0.5', **kwargs)
    self.max_seq_length = max_seq_length
    self.max_predictions_per_seq = max_predictions_per_seq
    self.num_lookahead_steps = num_lookahead_steps
    self.num_actions = num_actions
    self.masked_prob = masked_prob
    self.train_input_files = train_input_files
    self.eval_input_files = eval_input_files

class SupervisedBuilder(tfds.core.BeamBasedBuilder):
  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        features=_features_info(self.builder_config)
    )

  def _split_generators(self, dl_manager):
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(
                input_file_pattern=self.builder_config.train_input_files),
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(
                input_file_pattern=self.builder_config.eval_input_files),
        ),
    ]

  def _build_pcollection(self, pipeline, input_file_pattern):
    """Generate examples as dicts."""
    return ( pipeline
            | 'input' >> beam.io.ReadFromTFRecord(input_file_pattern)
            | 'parse lin' >> beam.ParDo(_lin_to_played_boards())
            | 'random positions' >> beam.ParDo(
                _random_positions(num=10, density=0.2))
            | 'add random key' >>  beam.Map(_add_random_key())
            | 'group' >> beam.GroupByKey()
            | 'split' >> beam.ParDo(_split())
            | 'make features' >>
              beam.Map(_features_from_played_position_builder(
                  self.builder_config)) )


class _random_positions(beam.DoFn):
  def __init__(self, num, density, filter_for_comments=False):
    self.num = num
    self.density = density
    self.filter_for_comments = filter_for_comments

  def process(self, ser_played_board):
    played_board = alphabridge_pb2.PlayedBoard()
    played_board.ParseFromString(ser_played_board)
    if self.filter_for_comments:
      valid_aix = [list(set(annotation.action_index
        for annotation in table.annotations))
        for table in played_board.tables]
    else:
      valid_aix = [list(range(len(table.actions)))
          for table in played_board.tables]
    num_actions = [len(v) for v in valid_aix]
    total_position = sum(num_actions)
    k = math.ceil(self.density * total_position)
    if k > self.num:
      k = self.num
    choices = random.sample(range(total_position), k)
    orig_source_uri = played_board.board_id.source_uri
    for choice in choices:
      t_ix, a_ix = 0, choice
      while a_ix >= num_actions[t_ix]:
        t_ix, a_ix = t_ix + 1, a_ix - num_actions[t_ix]
      played_board.board_id.source_uri = "{}:{}".format(orig_source_uri, choice)
      yield alphabridge_pb2.PlayedBoardPosition(board=played_board,
        table_index=t_ix, action_index=valid_aix[t_ix][a_ix]).SerializeToString()


class replay_buffer_sample_batch(beam.DoFn):
  def __init__(self, config):
    self.address = config.replay_buffer_address
    self.sample_batch_size = config.sample_batch_size
    self.replay_buffer = None

  def process(self, _):
    if not self.replay_buffer:
      options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
      channel = grpc.insecure_channel(self.address, options=options)
      self.replay_buffer = alphabridge_pb2_grpc.ReplayBufferStub(channel)

    req = alphabridge_pb2.SampleBatchRequest(batch_size=self.sample_batch_size)
    batch = self.replay_buffer.SampleBatch(req, wait_for_ready=True)
    for position in batch.position:
      yield position.SerializeToString()


def _features_to_example(features):
  def _to_feature(v):
    a = np.array(v)
    if a.dtype == np.int32 or a.dtype == np.int64:
      return tf.train.Feature(int64_list=tf.train.Int64List(
          value=a.flatten()))
    elif a.dtype == np.float32 or a.dtype == np.float64:
      return tf.train.Feature(float_list=tf.train.FloatList(
          value=a.flatten()))
    else:
      pdb.set_trace()

  example = tf.train.Example(features=tf.train.Features(feature={
    k: _to_feature(v) for k, v in features[1].items()}))
  return example.SerializeToString()

def _reinforcement_learning_filenames(config):
  """Generates filenames: TFRecord/TFExample features from ReplayBuffer."""
  for i in range(config.num_batches):
    batch_dir = os.path.join(config.tfrecord_dir, "batch-{:05d}".format(i))
    prefix = os.path.join(batch_dir, "shard")
    index_filepath = os.path.join(batch_dir, "index")
    with beam.Pipeline(options=config.beam_options) as pipeline :
      (
          pipeline
          | 'start' >> beam.Create(list(range(config.num_shards)))
          | 'sample' >> beam.ParDo(replay_buffer_sample_batch(config))
          | 'to_features' >>
            beam.Map(_features_from_played_position_builder(config))
          | 'serialize' >> beam.Map(_features_to_example)
          | 'write' >> beam.io.tfrecordio.WriteToTFRecord(
            file_path_prefix=prefix,
            file_name_suffix=".rec",
            num_shards=config.num_shards)
          | 'write index' >> beam.io.WriteToText(
              index_filepath, shard_name_template="")
      )
    with tf.io.gfile.GFile(index_filepath, "r") as f:
      yield [l.rstrip() for l in f.readlines()]


class ReinforcementConfig(object):
  def __init__(self, replay_buffer_address, tfrecord_temp_dir,
      num_batches=10, beam_options=None,
      use_tpu=False, sample_batch_size=512, num_shards=1,
      max_seq_length=256, max_predictions_per_seq=20,
      num_lookahead_steps=3, num_actions=90, masked_prob=0.05):
    self.use_tpu = use_tpu
    self.sample_batch_size = sample_batch_size
    self.replay_buffer_address = replay_buffer_address
    self.tfrecord_dir = tfrecord_temp_dir
    self.num_shards = num_shards
    self.num_batches = num_batches
    self.beam_options = beam_options

    self.max_seq_length = max_seq_length
    self.max_predictions_per_seq = max_predictions_per_seq
    self.num_lookahead_steps = num_lookahead_steps
    self.num_actions = num_actions
    self.masked_prob = masked_prob


class ReinforcementBuilder(object):
  def __init__(self, config):
    self.config = config

  def as_dataset(self):
    filenames_gen = lambda: _reinforcement_learning_filenames(self.config)
    filenames_fn = lambda: tf.data.Dataset.from_generator(
        filenames_gen,
        output_types=tf.string,
        output_shapes=self.config.num_shards)
    if self.config.use_tpu:
      import tpudata
      filenames_ds = tpudata.ControllerDataset(filenames_fn)
    else:
      filenames_ds = filenames_fn()

    filenames_ds = filenames_ds.unbatch()

    raw_ds = tf.data.TFRecordDataset(filenames_ds)

    def _ex_feature_info(fi):
      if fi.dtype == tf.int32:
        return tf.io.FixedLenFeature(fi.shape, tf.int64)
      else:
        return tf.io.FixedLenFeature(fi.shape, fi.dtype)

    features_info = {k: _ex_feature_info(v)
        for k,v in _features_info(self.config).items()}

    def _parse_fn(pb):
      features = tf.io.parse_single_example(pb, features_info)
      for k in features:
        if features[k].dtype == tf.int64:
          features[k] = tf.cast(features[k], tf.int32)
      return features

    features_ds = raw_ds.map(_parse_fn)
    return features_ds


class TopConfig(tfds.core.BuilderConfig):
  def __init__(self, name,
      train_lin_files=None,
      eval_lin_files=None,
      corpus_files=None,
      jargon_vocab_file=None,
      jargon_do_lower_case=None,
      redact_commentator_names=None,
      max_player_seq_length=256,
      max_jargon_seq_length=256,
      max_top_seq_length=256,
      comment_regex='',
      num_merge_keys=1<<14,
      **kwargs):
    super().__init__(name=name, version='0.0.6', **kwargs)
    self.max_player_seq_length = max_player_seq_length
    self.max_jargon_seq_length = max_jargon_seq_length
    self.max_top_seq_length = max_top_seq_length
    self.jargon_vocab_file = jargon_vocab_file
    self.jargon_do_lower_case = jargon_do_lower_case
    self.redact_commentator_names = redact_commentator_names
    self.train_lin_files = train_lin_files
    self.test_lin_files = eval_lin_files
    self.corpus_files = corpus_files
    self.comment_regex = re.compile(comment_regex)
    self.num_merge_keys = num_merge_keys


class TopBuilder(tfds.core.GeneratorBasedBuilder):
  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        features=_top_features_info(self.builder_config)
    )

  def _split_generators(self, dl_manager, pipeline):
    return {
      'train': self._generate_examples(pipeline, "train",
        lin_file_pattern=self.builder_config.train_lin_files,
        corpus_file_pattern=self.builder_config.corpus_files),
      'test': self._generate_examples(pipeline, "test",
        lin_file_pattern=self.builder_config.test_lin_files,
        corpus_file_pattern=self.builder_config.corpus_files),
    }

  def _generate_examples(self, pipeline, run_type, lin_file_pattern, corpus_file_pattern):
    """Generate examples as dicts."""
    def _board_has_comments(ser_played_board):
      played_board = alphabridge_pb2.PlayedBoard()
      played_board.ParseFromString(ser_played_board)
      return sum(len(table.annotations) for table in played_board.tables) > 0

    comment_regex = self.builder_config.comment_regex
    num_merge_keys = self.builder_config.num_merge_keys

    corpus_pipeline = (pipeline
        | run_type + ' read corpus' >> beam.io.ReadFromText(corpus_file_pattern)
        | run_type + ' filter corpus' >> beam.ParDo(_select_comments(comment_regex))
        | run_type + ' add random key corpus' >>  beam.Map(_add_random_key(num_merge_keys))
        )

    lin_pipeline = (pipeline
        | run_type + ' read lin' >> beam.io.ReadFromTFRecord(lin_file_pattern)
        | run_type + ' parse lin' >> beam.ParDo(_lin_to_played_boards())
        | run_type + ' filter comments' >> beam.Map(_select_comments_in_board(comment_regex))
        | run_type + ' filter boards' >> beam.Filter(_board_has_comments)
        | run_type + ' random positions' >> beam.ParDo(
          _random_positions(num=10, density=0.6, filter_for_comments=True))
        | run_type + ' add random key lin' >>  beam.Map(_add_random_key(num_merge_keys))
        )

    return ({'boards': lin_pipeline, 'corpus': corpus_pipeline}
        | run_type + ' merge' >> beam.CoGroupByKey()
        | run_type + ' transmogrify' >> beam.ParDo(_zip_boards_and_corpus())
        | run_type + ' make features' >>
          beam.Map(_top_features_builder(self.builder_config))
        )
