from typing import Generator, Mapping, Sequence, Text, Tuple

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import scaffold.dims
import scaffold.fsa
from bridge.fastgame import wrapper as bridgegame


Batch = Mapping[Text, Tuple]

AUTOTUNE = tf.data.experimental.AUTOTUNE


def _replace_none(t):
    if isinstance(t, tuple):
        return tuple(_replace_none(x) for x in t)
    if isinstance(t, dict):
        return {key: _replace_none(val) for key, val in t.items()}
    return -1 if t is None else t


class FSABuilder(tfds.core.GeneratorBasedBuilder):
    """Abridged bidding (-contract_seat). Practical to enumerate as FSA."""
    VERSION = tfds.core.Version('0.0.1')
    action_question_names = ('next_bid', 'next_call', 'next_card')
    state_question_names = ('last_bid', 'pass_position', 'next_to_act', 'stage')

    def _observable_features_info(self, observable):
         if isinstance(observable, scaffold.dims.OneHotObservable):
            return tfds.features.Tensor(
                dtype=tf.int32, shape=[len(observable.dims)])
         elif isinstance(observable, scaffold.dims.AbsoluteObservable):
            raise NotImplementedError
         elif isinstance(observable, scaffold.dims.FuzzyObservable):
            raise NotImplementedError

    def _info(self):
        action_features = {
            name: self._observable_features_info(scaffold.dims.questions[name])
            for name in self.action_question_names}
        state_features = {
            name: self._observable_features_info(scaffold.dims.questions[name])
            for name in self.state_question_names}

        features = tfds.features.FeaturesDict({
            "action_observables": action_features,
            "current_observables": state_features,
            "future_observables": state_features,
        })
        return tfds.core.DatasetInfo( builder=self, features=features)

    def _split_generators(self, dl_manager, pipeline):
        return {
            'train': self._generate_examples(),
            'test': self._generate_examples(),
        }

    def _generate_examples(self):
        """Generate examples as dicts."""
        for result in scaffold.fsa.traverse_online(bridgegame.Game()):
            deal, current_observables, action_idx, future_observables = result
            action_observables = \
                    scaffold.dims.extract_observables_from_action_id_and_seat(
                            action_idx, deal.next_to_act_index())
            def observables_dict(names, observables):
                return {names[i]: ans
                    for i, ans in enumerate(_replace_none(observables))}
            result = {
                "action_observables": _replace_none(action_observables),
                "current_observables":
                    observables_dict(self.state_question_names, current_observables),
                "future_observables":
                    observables_dict(self.state_question_names, future_observables),
            }
            yield str((current_observables, action_idx)), result


class HistoryBuilder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.0.1')
    action_question_names = ('next_bid', 'next_call', 'next_card')
    state_question_names = ('last_bid', 'pass_position', 'next_to_act', 'stage')

    def _observable_features_info(self, observable):
         if isinstance(observable, scaffold.dims.OneHotObservable):
            return tfds.features.Tensor(
                dtype=tf.int32, shape=[len(observable.dims)])

    def _info(self):
        action_features = {
            name: self._observable_features_info(scaffold.dims.questions[name])
            for name in self.action_question_names}
        state_features = {
            name: self._observable_features_info(scaffold.dims.questions[name])
            for name in self.state_question_names}

        features = tfds.features.FeaturesDict({
            "action_observables": action_features,
            "current_observables": state_features,
            "future_observables": state_features,
        })
        return tfds.core.DatasetInfo( builder=self, features=features)

    def _split_generators(self, dl_manager, pipeline):
        return {
            'train': self._generate_examples(),
            'test': self._generate_examples(),
        }

    def _generate_examples(self):
        """Generate examples as dicts."""
        for result in scaffold.fsa.traverse_online(bridgegame.Game()):
            deal, current_observables, action_idx, future_observables = result
            action_observables = \
                    scaffold.dims.extract_observables_from_action_id_and_seat(
                            action_idx, deal.next_to_act_index())
            def observables_dict(names, observables):
                return {names[i]: ans
                    for i, ans in enumerate(_replace_none(observables))}
            result = {
                "action_observables": _replace_none(action_observables),
                "current_observables":
                    observables_dict(self.state_question_names, current_observables),
                "future_observables":
                    observables_dict(self.state_question_names, future_observables),
            }
            yield str((current_observables, action_idx)), result


def generate_data(builder, *,
        is_training: bool,
        # batch_dims should be:
        # [device_count, per_device_batch_size] or [total_batch_size]
        batch_dims: Sequence[int],
        threadpool_size: int = 48,
        max_intra_op_parallelism: int = 1,
) -> Generator[Batch, None, None]:
    """Loads the given split of the dataset."""
    total_batch_size = np.prod(batch_dims)

    print(builder._info())
    builder.download_and_prepare()
    ds = builder.as_dataset()

    options = tf.data.Options()
    options.threading.private_threadpool_size = threadpool_size
    options.threading.max_intra_op_parallelism = (
            max_intra_op_parallelism)
    options.experimental_optimization.map_parallelization = True

    if is_training:
        options.experimental_deterministic = False
        ds = ds["train"]
        ds = ds.with_options(options)
        if jax.process_count() > 1:
            # Only cache if we are reading a subset of the dataset.
            ds = ds.cache()
        ds = ds.repeat()
        ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=0)
    else:
        ds = ds["test"]
        ds = ds.with_options(options)

    for batch_size in reversed(batch_dims):
        ds = ds.batch(batch_size)

    ds = ds.prefetch(AUTOTUNE)

    yield from tfds.as_numpy(ds)

