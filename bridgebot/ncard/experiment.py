# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
from typing import Generator, Mapping, Text, Tuple
import cProfile
import logging
import pathlib
import threading
import collections
import pickle

from typing import Any, Callable, Dict, Generator, Iterable, Mapping, Optional, Sequence, TypeVar

from absl import app
from absl import flags
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import experiment
from jaxline import platform
from jaxline import utils as jl_utils
from ml_collections import config_dict
import numpy as np
import optax

from third_party.perceiver.train import utils
from bridgebot.ncard import training_pipeline
from bridgebot.ncard import model
from bridgebot.ncard import game as bridgegame
from bridgebot.ncard import chords

import pdb

OptState = Tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]
Scalars = Mapping[Text, jnp.ndarray]
SnapshotNT = collections.namedtuple("SnapshotNT", ["id", "pickle_nest"])
CheckpointNT = collections.namedtuple("CheckpointNT", ["active", "history"])



class Experiment(experiment.AbstractExperiment):
  NON_BROADCAST_CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_state': 'state',
      '_opt_state': 'opt_state',
  }

  def __init__(self, mode, init_rng, config):
    """Initializes experiment."""

    super(Experiment, self).__init__(mode=mode, init_rng=init_rng)
    self.mode = mode
    self.init_rng = init_rng
    self.config = config

    # Checkpointed experiment state.
    self._params = None
    self._state = None
    self._opt_state = None

    # Input pipelines.
    self._train_input = None

    self.game = bridgegame.Game(n=2)
    tokenizer = bridgegame.Tokenizer(self.game)
    self.codecs = {
            "bool": _compile_codec(
                tokenizer, chords.all_bool_chords(self.game)),
            "outcome": _compile_codec(
                tokenizer, chords.all_outcome_chords(self.game)),
            }
    data_builder_cfg = training_pipeline.ReinforcementConfig(
        self.game, 'localhost:10000', '/tmp', sample_batch_size=32)
    self.data_builder = training_pipeline.ReinforcementBuilder(data_builder_cfg)

    self.forward = hk.transform_with_state(self._forward_fn)

    # NOTE: We "donate" the `params, state, opt_state` arguments which allows
    # JAX (on some backends) to reuse the device memory associated with these
    # inputs to store the outputs of our function (which also start with
    # `params, state, opt_state`).
    self._update_func = self._update_func

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(self, global_step: int, rng: jnp.ndarray,
           *unused_args, **unused_kwargs):
    """See base class."""

    if self._train_input is None:
      self._initialize_train()

    inputs = next(self._train_input)
    labels = {
        "target_mask": inputs["target_mask"],
        "target_ids": inputs["target_ids"],
        "target_policy": inputs["target_policy"],
        "is_playing": inputs["is_playing"]
        }
    del inputs["target_mask"]
    del inputs["target_ids"]
    del inputs["target_policy"]

    self._params, self._state, self._opt_state, scalars = (
        self._update_func(
            self._params, self._state, self._opt_state, inputs, labels,
            rng, global_step))

    # scalars = jl_utils.get_first(scalars)
    scalars["learning_rate"] = scalars["learning_rate"].mean()
    return scalars

  def _build_train_input(self):
    ds = self.data_builder.as_dataset()
    ds = ds.batch(64)
    # ds = ds.batch(1)  # TODO: add parallelism dimension.
    return ds.as_numpy_iterator()

  def _initialize_train(self):
    self._train_input = jl_utils.py_prefetch(self._build_train_input)

    total_batch_size = self.config.training.batch_size
    steps_per_epoch = (self.config.training.examples_per_epoch /
            self.config.training.batch_size)
    total_steps = self.config.training.n_epochs * steps_per_epoch
    # Scale by the (negative) learning rate.
    self._lr_schedule = utils.get_learning_rate_schedule(
        total_batch_size, steps_per_epoch, total_steps, self.config.optimizer)

    self._optimizer = utils.make_optimizer(
        self.config.optimizer,
        self._lr_schedule)

    # Check we haven't already restored params
    if self._params is None:
      logging.info('Initializing parameters.')

      inputs = next(self._train_input)

      #init_net = jax.pmap(lambda init_rng, inputs: self.forward.init(
      #    init_rng, inputs, is_training=True))
      #init_opt = jax.pmap(self._optimizer.init)
      #init_rng = jl_utils.bcast_local_devices(self.init_rng)
      init_net = lambda init_rng, inputs: self.forward.init(
              init_rng, inputs, is_training=True)
      init_opt = self._optimizer.init
      init_rng = self.init_rng

      self._params, self._state = init_net(init_rng, inputs)
      self._opt_state = init_opt(self._params)

  def _forward_fn(self, inputs, is_training):
      return model.forward_function(
              inputs, is_training,
              self.config.model.perceiver_kwargs.chord_config,
              self.config.model.perceiver_kwargs.encoder_config,
              self.config.model.perceiver_kwargs.decoder_config)

  def _loss_fn(
      self,
      params: hk.Params,
      state: hk.State,
      inputs: dict,
      labels: dict,
      rng: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, Tuple[Scalars, hk.State]]:
    (logits, t), state = self.forward.apply(
        params, state, rng, inputs, is_training=True)

    scaled_loss, (metrics, dbg) = model.loss_function(self.codecs, logits, inputs, labels)
    return scaled_loss, (metrics, dbg, t, state)


  def _update_func(
      self,
      params: hk.Params,
      state: hk.State,
      opt_state: OptState,
      inputs: dict,
      labels: dict,
      rng: jnp.ndarray,
      global_step: int,
  ) -> Tuple[hk.Params, hk.State, OptState, Scalars]:
    """Applies an update to parameters and returns new state."""
    # This function computes the gradient of the first output of loss_fn and
    # passes through the other arguments unchanged.
    grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)
    scaled_grads, (loss_scalars, dbg, t, state) = grad_loss_fn(
        params, state, inputs, labels, rng[0])
    grads = scaled_grads  # jax.lax.psum(scaled_grads, axis_name='i')

    #pdb.set_trace()
    # Grab the learning rate to log before performing the step.
    learning_rate = self._lr_schedule(global_step)

    # Compute and apply updates via our optimizer.
    updates, opt_state = self._optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    n_params = 0
    for k in params.keys():
      for l in params[k]:
        n_params = n_params + np.prod(params[k][l].shape)

    # Scalars to log (note: we log the mean across all hosts/devices).
    scalars = {'learning_rate': learning_rate,
               'n_params (M)': float(n_params/1e6),
               'global_gradient_norm': optax.global_norm(grads)}
    loss_scalars = {f'train_{k}': v for k, v in loss_scalars.items()}
    scalars.update(loss_scalars)
    #scalars = jax.lax.pmean(scalars, axis_name='i')

    return params, state, opt_state, scalars

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, global_step, rng, **unused_args):
    """See base class."""
    return {}  # YAGNI? we would have to do sth to get static eval data.

def maybe_device_get(x):
  """Device get tensor if it is a jnp.ndarray."""
  return jax.device_get(x) if isinstance(x, jnp.ndarray) else x

GLOBAL_CHECKPOINT_DICT = {}

class NaiveDiskCheckpointer:
  """A Checkpointer reliant on an in-memory global dictionary."""

  def __init__(self, config, mode: str):
    self._max_checkpoints_to_keep = config.max_checkpoints_to_keep
    del mode

  def get_experiment_state(self, ckpt_series: str):
    """Returns the experiment state for a given checkpoint series."""
    if ckpt_series not in GLOBAL_CHECKPOINT_DICT:
      active = threading.local()
      new_series = CheckpointNT(active, [])
      GLOBAL_CHECKPOINT_DICT[ckpt_series] = new_series
    if not hasattr(GLOBAL_CHECKPOINT_DICT[ckpt_series].active, "state"):
      GLOBAL_CHECKPOINT_DICT[ckpt_series].active.state = (
          config_dict.ConfigDict())
    return GLOBAL_CHECKPOINT_DICT[ckpt_series].active.state

  def save(self, ckpt_series: str) -> None:
    """Saves the checkpoint."""
    snapshot_state = config_dict.ConfigDict()
    for k, v in self.get_experiment_state(ckpt_series).items():
      if k == "experiment_module":
        snapshot_state[k] = v.snapshot_state()
      else:
        snapshot_state[k] = v
    # Ensure buffers do not get donated as training loop runs ahead.
    snapshot_state = jax.tree_map(maybe_device_get, snapshot_state)
    filenames = pathlib.Path(checkpoint_dir).glob(ckpt_series + "-??????")
    filenames = [fn for fn in filenames]
    if filenames:
        last_id = int(str(sorted(filenames)[-1])[-6:])
        next_id = last_id + 1
    else:
        next_id = 1

    path = pathlib.Path(checkpoint_dir) / f"{ckpt_series}-{next_id:06d}"
    with open(path, "wb") as f:
        pickle.dump(snapshot_state, f)
        logging.info("Saved checkpoint to %s.", path)



  def can_be_restored(self, ckpt_series: str) -> bool:
    filenames = pathlib.Path(checkpoint_dir).glob(ckpt_series + "-??????")
    filenames = [fn for fn in filenames]
    return not not filenames

  def restore(self, ckpt_series: str) -> None:
    """Restores the checkpoint."""
    filenames = pathlib.Path(checkpoint_dir).glob(ckpt_series + "-??????")
    filenames = [fn for fn in filenames]
    with open(sorted(filenames)[-1], "rb") as f:
      snapshot_state = pickle.load(f).to_dict()
    current_state = self.get_experiment_state(ckpt_series)
    for k, v in current_state.items():
      if k == "experiment_module":
        v.restore_from_snapshot(snapshot_state[k])
      else:
        current_state[k] = snapshot_state[k]
    logging.info("Returned checkpoint from %s.", sorted(filenames)[-1])

  def restore_path(self, ckpt_series: str) -> Optional[str]:
    """Returns the restore path for the checkpoint, or None."""
    if not self.can_be_restored(ckpt_series):
      return None
    return GLOBAL_CHECKPOINT_DICT[ckpt_series].history[-1].id

  def wait_for_checkpointing_to_finish(self) -> None:
    """Waits for any async checkpointing to complete."""


checkpoint_dir = None

def main(argv):
    global checkpoint_dir
    checkpoint_dir = os.path.join(
            flags.FLAGS.output_base_dir,
            flags.FLAGS.config.checkpoint_dir)
    platform.main(Experiment, argv,
                  checkpointer_factory=NaiveDiskCheckpointer)


def _compile_codec(tokenizer, all_chords):
    b = chords.MachineBuilder(tokenizer.all_tokens)
    start = b.traverse(all_chords)
    codec = b.compile([start])[0]
    return chords.id_codec(codec, tokenizer.all_tokens)
    

flags.DEFINE_string('output_base_dir',
          'data/scaffold_perceiver_experiment_out',
          'base path for exoeriment output')
if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  flags.mark_flag_as_required('config')
  logging.getLogger('apache_beam').setLevel(logging.ERROR)
  #cProfile.run('app.run(main)')
  app.run(main)
