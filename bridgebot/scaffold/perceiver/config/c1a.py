import os
import sys

from absl import app
from absl import flags
from jaxline import base_config
from ml_collections import config_dict

FLAGS = flags.FLAGS


N_TRAIN_EXAMPLES = 7594

def get_training_steps(batch_size, n_epochs):
  return (N_TRAIN_EXAMPLES * n_epochs) // batch_size


def get_config():
  """Return config object for training."""
  use_debug_settings = False
  config = base_config.get_base_config()

  # Experiment config.
  local_batch_size = 64
  # Modify this to adapt to your custom distributed learning setup
  num_devices = 1
  config.train_batch_size = local_batch_size * num_devices
  config.n_epochs = 110

  def _default_or_debug(default_value, debug_value):
    return debug_value if use_debug_settings else default_value

  n_train_examples = N_TRAIN_EXAMPLES

  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              optimizer=dict(
                  base_lr=5e-4,
                  max_norm=10.0,  # < 0 to turn off.
                  schedule_type='constant_cosine',
                  weight_decay=1e-1,
                  decay_pos_embs=True,
                  scale_by_batch=True,
                  cosine_decay_kwargs=dict(
                      init_value=0.0,
                      warmup_epochs=0,
                      end_value=0.0,
                  ),
                  step_decay_kwargs=dict(
                      decay_boundaries=[0.5, 0.8, 0.95],
                      decay_rate=0.1,
                  ),
                  constant_cosine_decay_kwargs=dict(
                      constant_fraction=0.5,
                      end_value=0.0,
                  ),
                  optimizer='lamb',
                  # Optimizer-specific kwargs:
                  adam_kwargs=dict(
                      b1=0.9,
                      b2=0.999,
                      eps=1e-8,
                  ),
                  lamb_kwargs=dict(
                      b1=0.9,
                      b2=0.999,
                      eps=1e-6,
                  ),
              ),
              # Don't specify output_channels - it's not used for
              # classifiers.
              model=dict(
                  dataset = 'fsa',
                  perceiver_kwargs=dict(
                      embedding_config=dict(
                          num_fourier_channels=8,  # TODO-a ,12
                          num_categorical_channels=10,  #TODO-b ,15
                          question_name_embedding_size=4, #TODO-c ,4
                      ),
                      encoder_config=dict(
                          num_self_attends_per_block=_default_or_debug(2, 2), #TODO-d ,8
                          # Weights won't be shared if num_blocks is set to 1.
                          num_blocks=_default_or_debug(2, 2),
                          z_index_dim=64,
                          num_z_channels=128,
                          num_cross_attend_heads=1, #TODO -e ,1 
                          num_self_attend_heads=2, #TODO -f ,4
                          cross_attend_widening_factor=1,
                          self_attend_widening_factor=1,
                          dropout_prob=0.0, #TODO -g ,0.1
                          # Position encoding for the latent array.
                          z_pos_enc_init_scale=0.02,
                          cross_attention_shape_for_attn='kv',
                          use_query_residual=True,
                          ),
                      decoder_config=dict(
                      ),
                      experiment_replace_none_randomly=False,
                  ),
              ),
              training=dict(
                  examples_per_epoch=n_train_examples,
                  label_smoothing=0.1,
                  n_epochs=config.get_oneway_ref('n_epochs'),
                  batch_size=config.get_oneway_ref('train_batch_size'),
                  experiment_none_is_uniform_output=False,
              ),
              evaluation=dict(
                  subset='test',
                  batch_size=2,
              ),
          )
      )
  )

  # Training loop config.
  config.training_steps = get_training_steps(
      config.get_oneway_ref('train_batch_size'),
      config.get_oneway_ref('n_epochs'))
  config.log_train_data_interval = 60
  config.log_tensors_interval = 60
  config.save_checkpoint_interval = 300
  config.eval_specific_checkpoint_dir = ''
  config.best_model_eval_metric = 'eval_top_1_acc'
  config.checkpoint_dir = os.path.basename(__file__)[:-3]
  config.train_checkpoint_all_hosts = False

  config.lock()
  return config
