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

"""A reference training pipeline for Perceiver/Perceiver IO on ImageNet.

We use the Jaxline (https://github.com/deepmind/jaxline) training framework.
Two sets of hyperparameters are provided, the hyperparameters we used for the
Perceiver IO paper, and scaled-down hyperparameters for local testing.
This script should run out-of-the-box with the local hyper parameters.
The scaled-up hyperparameters requires a distributed learning setup to run,
and this script will need to be adapted to your specific setup.
"""

import functools
import os
from typing import Generator, Mapping, Text, Tuple

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import base_config
from jaxline import experiment
from jaxline import platform
from jaxline import utils as jl_utils
from ml_collections import config_dict
import numpy as np
import optax

from scaffold.perceiver.perceiver.train import utils

from scaffold.perceiver.perceiver import perceiver
from scaffold.perceiver import datasource
from scaffold.perceiver import dims_perceiver

import pdb

OptState = Tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]
Scalars = Mapping[Text, jnp.ndarray]


class Experiment(experiment.AbstractExperiment):
  """bridge rule dynamics experiment."""

  # A map from object properties that will be checkpointed to their name
  # in a checkpoint. Currently we assume that these are all sharded
  # device arrays.
  CHECKPOINT_ATTRS = {
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
    self._eval_input = None

    # Datasource
    if self.config.model.dataset == 'fsa':
        self.data_builder = datasource.FSABuilder()
    else:
        raise NotImplementedError

    self.forward = hk.transform_with_state(self._forward_fn)

    # NOTE: We "donate" the `params, state, opt_state` arguments which allows
    # JAX (on some backends) to reuse the device memory associated with these
    # inputs to store the outputs of our function (which also start with
    # `params, state, opt_state`).
    self._update_func = jax.pmap(self._update_func, axis_name='i',
                                 donate_argnums=(0, 1, 2))
    self._eval_batch = jax.jit(self._eval_batch)

  def _forward_fn(
      self,
      inputs: datasource.Batch,
      is_training: bool,
  ) -> jnp.ndarray:

    return dims_perceiver.forward_function(inputs, is_training,
        self.data_builder.action_question_names,
        self.data_builder.state_question_names,
        **self.config.model.perceiver_kwargs)

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

    self._params, self._state, self._opt_state, scalars = (
        self._update_func(
            self._params, self._state, self._opt_state, inputs, rng, global_step
            ))

    scalars = jl_utils.get_first(scalars)
    return scalars

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

      init_net = jax.pmap(lambda *a: self.forward.init(*a, is_training=True))
      init_opt = jax.pmap(self._optimizer.init)

      # Init uses the same RNG key on all hosts+devices to ensure everyone
      # computes the same initial state.
      init_rng = jl_utils.bcast_local_devices(self.init_rng)

      self._params, self._state = init_net(init_rng, inputs)
      self._opt_state = init_opt(self._params)

  def _build_train_input(self) -> Generator[datasource.Batch, None, None]:
    """See base class."""
    num_devices = jax.device_count()
    global_batch_size = self.config.training.batch_size
    per_device_batch_size, ragged = divmod(global_batch_size, num_devices)

    if ragged:
      raise ValueError(
          f'Global batch size {global_batch_size} must be divisible by '
          f'num devices {num_devices}')

    return datasource.generate_data(
        self.data_builder,
        is_training=True,
        batch_dims=[jax.local_device_count(), per_device_batch_size])

  def _loss_fn(
      self,
      params: hk.Params,
      state: hk.State,
      inputs: datasource.Batch,
      rng: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, Tuple[Scalars, hk.State]]:
    logits, state = self.forward.apply(
        params, state, rng, inputs, is_training=True)

    scaled_loss, metrics = dims_perceiver.loss_fn(
            logits, inputs['future_observables'],
            self.data_builder.state_question_names,
            self.config.training.experiment_none_is_uniform_output)
    return scaled_loss, (metrics, state)

  def _update_func(
      self,
      params: hk.Params,
      state: hk.State,
      opt_state: OptState,
      inputs: datasource.Batch,
      rng: jnp.ndarray,
      global_step: int,
  ) -> Tuple[hk.Params, hk.State, OptState, Scalars]:
    """Applies an update to parameters and returns new state."""
    # This function computes the gradient of the first output of loss_fn and
    # passes through the other arguments unchanged.
    grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)
    scaled_grads, (loss_scalars, state) = grad_loss_fn(
        params, state, inputs, rng)
    grads = jax.lax.psum(scaled_grads, axis_name='i')

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
    scalars = jax.lax.pmean(scalars, axis_name='i')

    return params, state, opt_state, scalars

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, global_step, rng, **unused_args):
    """See base class."""
    global_step = np.array(jl_utils.get_first(global_step))
    scalars = jax.device_get(self._eval_epoch(jl_utils.get_first(rng)))

    logging.info('[Step %d] Eval scalars: %s', global_step, scalars)
    return scalars

  def _eval_batch(
      self,
      params: hk.Params,
      state: hk.State,
      inputs: datasource.Batch,
      rng: jnp.ndarray,
  ) -> Scalars:
    """Evaluates a batch."""
    logits, state = self.forward.apply(
        params, state, rng, inputs, is_training=True)

    scaled_loss, metrics = dims_perceiver.loss_fn(
            logits, inputs['future_observables'],
            self.data_builder.state_question_names,
            self.config.training.experiment_none_is_uniform_output)
    return metrics

  def _build_eval_input(self) -> Generator[datasource.Batch, None, None]:
    return datasource.generate_data(
        self.data_builder,
        is_training=False,
        batch_dims=[self.config.evaluation.batch_size])

  def _eval_epoch(self, rng):
    """Evaluates an epoch."""
    num_samples = 0.
    summed_scalars = None

    params = jl_utils.get_first(self._params)
    state = jl_utils.get_first(self._state)

    for inputs in self._build_eval_input():
      num_samples += list(inputs['future_observables'].values())[0].shape[0]
      scalars = self._eval_batch(params, state, inputs, rng)

      # Accumulate the sum of scalars for each step.
      scalars = jax.tree_map(lambda x: jnp.sum(x, axis=0), scalars)
      if summed_scalars is None:
        summed_scalars = scalars
      else:
        summed_scalars = jax.tree_multimap(jnp.add, summed_scalars, scalars)

    mean_scalars = jax.tree_map(lambda x: x / num_samples, summed_scalars)
    return mean_scalars


def main(argv):
    flags.FLAGS.config.checkpoint_dir = os.path.join(
            flags.FLAGS.output_base_dir,
            flags.FLAGS.config.checkpoint_dir)
    platform.main(Experiment, argv)

if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  flags.DEFINE_string('output_base_dir',
          'data/scaffold_perceiver_experiment_out',
          'base path for exoeriment output')

  app.run(main)
