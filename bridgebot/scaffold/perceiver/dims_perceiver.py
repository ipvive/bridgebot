from absl import flags
from collections import namedtuple
import haiku as hk
import jax
import jax.numpy as jnp

from scaffold.perceiver.perceiver import perceiver
from scaffold.perceiver.perceiver import io_processors
from scaffold.perceiver.perceiver import position_encoding
import scaffold.perceiver.perceiver.train.utils as perceiver_utils
import scaffold.dims as dims

import pdb

def circular_fourier_features(pos, num_bands, period):
    """Generate a circular Fourier frequency position encoding with linear spacing.

    Args:
      pos: The n values in Z/period Z
        A jnp array of shape (n,).
      num_bands: The number of bands to use.
      period: desired period for the embedding
    Returns:
      embedding: A 1D jnp array of shape [n, n_channels].
           sin(pi*f_1*dim_1), cos(pi*f_1 dim1, ..., sin(pi*f_k*dim_1), ...,
         where dim_i is pos[:, i] and f_k is the kth frequency band.
    """
    freq_bands = jnp.linspace(1/period, num_bands/period, num=num_bands, endpoint=True)
    per_pos_features = pos[:, None] * freq_bands[None, :]
    return jnp.concatenate(
          [jnp.sin(2 * jnp.pi * per_pos_features),
           jnp.cos(2 * jnp.pi * per_pos_features)], axis=-1)


DimsConfig = namedtuple('DimsConfig', [
    'num_fourier_channels',
    'num_categorical_channels',
    'categorical_embed_layers',
    'question_name_embedding_size',
    'experiment_replace_none_randomly',
    ])


def set_positions(dimensions, positions, dims_config):
    num_fourier_channels = dims_config.num_fourier_channels
    num_categorical_channels = dims_config.num_categorical_channels
    categorical_embed_layers = dims_config.categorical_embed_layers
    assert num_fourier_channels % 2 == 0
    batch_size = len(positions)
    fourier_embeddings = [jnp.zeros((batch_size, dims_config.num_fourier_channels))
            for _ in range(dims.num_axes)]
    categorical_embeddings = jnp.zeros(
        (batch_size, dims_config.num_categorical_channels))
    batch_size = positions.shape[0] 
    for i, dim in enumerate(dimensions):
        pos = positions[:,i].reshape((batch_size, -1))
        mask = (pos != -1)
        if dims_config.experiment_replace_none_randomly:
            replacements = jax.random.randint(hk.next_rng_key(),
                    pos.shape, -1, len(dim.labels))
            pos = replacements * (1 - mask) + pos * mask
            mask = (pos != -1)
        if isinstance(dim, dims.CircleDim):
            fourier_embeddings[dim.axis] = circular_fourier_features(
                pos, num_fourier_channels // 2, dim.period).reshape(
                (batch_size, -1)) * mask
        if isinstance(dim, dims.RangeDim):
            fourier_embeddings[dim.axis] = \
                position_encoding.generate_fourier_features(
                pos, num_fourier_channels // 2, (1 + dim.max_value,),
                concat_pos=False, sine_only=False).reshape(
                (batch_size, -1)) * mask
        if isinstance(dim, dims.CategoricalDim):
            embedding = categorical_embed_layers[dim.name](pos)
            categorical_embeddings += embedding.reshape((batch_size,-1)) * mask

    output = jnp.concatenate(fourier_embeddings + [categorical_embeddings], axis=-1)
    output = output[:, None, :]
    return output


def get_position_logits(dimensions, inputs, dims_config):
    batch_size = inputs.shape[0]
    num_fourier_channels = dims_config.num_fourier_channels
    num_categorical_channels = dims_config.num_categorical_channels
    categorical_embed_layers = dims_config.categorical_embed_layers
    # drop positional embeddings
    fourier_length = num_fourier_channels * dims.num_axes
    fourier_inputs = inputs[:,:,:fourier_length]
    catetorical_inputs = inputs[
            :,:,fourier_length:fourier_length + num_categorical_channels]
    fourier_inputs = jnp.reshape(fourier_inputs,
        (batch_size, -1, dims.num_axes, num_fourier_channels))
    logits = {}
    for dim in dimensions:
        if isinstance(dim, dims.CircleDim):
            dim_inputs = fourier_inputs[:,:,dim.axis,:]
            embedding_matrix = circular_fourier_features(
                jnp.array(range(dim.period)), num_fourier_channels // 2, dim.period)
        elif isinstance(dim, dims.RangeDim):
            dim_inputs = fourier_inputs[:,:,dim.axis,:]
            embedding_matrix = position_encoding.generate_fourier_features(
                jnp.array(range(dim.max_value + 1)).reshape(-1,1), num_fourier_channels // 2,
                (1 + dim.max_value,), concat_pos=False, sine_only=False)
        elif isinstance(dim, dims.CategoricalDim):
            dim_inputs = catetorical_inputs
            embedding_matrix = categorical_embed_layers[dim.name].embeddings
        else:
            pdb.set_trace()

        embedding_decoder = io_processors.EmbeddingDecoder(
            embedding_matrix=embedding_matrix, name=dim.name)
        logits[dim.name] = embedding_decoder(dim_inputs)
    return logits


class OneHotPreprocessor(hk.Module):
    def __init__(self, observable, dims_config, name=None):
        super().__init__(name=name)
        self.observable = observable
        self.dims_config = dims_config

    def __call__(self, inputs, is_training, **kwargs):
        pos_emb = set_positions(self.observable.dims, inputs,
                dims_config=self.dims_config)
        return pos_emb, None, inputs


class AbsolutePreprocessor(hk.Module):
    def __init__(*args, **kwargs):
        raise NotImplementedError  # TODO
    # def __call__(...):


class FuzzyPreprocessor(hk.Module):
    def __init__(*args, **kwargs):
        raise NotImplementedError  # TODO
    # def __call__(...):


class DimsPreprocessor(hk.Module):
    def __init__(self, question_names, all_questions, dims_config, name=None):
        super().__init__(name=name)
        self.question_names = question_names
        self.dims_config = dims_config
        modalities = {name: self.get_preprocessor(name, all_questions[name])
                for name in self.question_names}
        # this class concatinated prositional embedding in padding,
        # consider adding instead
        self.multimodal = io_processors.MultimodalPreprocessor(
                modalities,
                key_order = question_names,
                min_padding_size=dims_config.question_name_embedding_size)

    def get_preprocessor(self, name, observable):
        if isinstance(observable, dims.OneHotObservable):
            return OneHotPreprocessor(observable, self.dims_config)
        if isinstance(observable, dims.AbsoluteObservable):
            return AbsolutePreprocessor(observable)
        if isinstance(observable, dims.FuzzyObservable):
            return FuzzyPreprocessor(observable)

    def __call__(self, inputs, is_training, pos=None, network_input_is_1d=True):
        return self.multimodal(inputs, is_training=is_training)


class DimsDecoder(perceiver.BasicDecoder):
    # TODO(njt): re-evaluate this decision if we use dims_io in a
    #            different way.
    # we want to have decoder_query == inputs_with_pos, with residual.
    # (this is an automorphism, and is mostly invariant;
    #  e.g., dealt_cards never subtracts, and only adds incrementally)
    def __init__(self, num_action_questions=3,
            qk_channels=None, v_channels=None,
            num_heads=1, name=None):
        super().__init__(
            output_num_channels=1,
            position_encoding_type='none',
            qk_channels=qk_channels,
            v_channels=v_channels,
            use_query_residual=True,
            output_w_init=None,
            num_heads=num_heads,
            final_project=False,
            name=name)
        self.num_action_questions = num_action_questions

    def output_shape(self, inputs):
        return (inputs[:,self.num_action_questions:,:].shape, None)

    def decoder_query(self, inputs, modality_sizes=None,
            inputs_without_pos=None, subsampled_points=None):
        # we always assume that action questions are fist in key_order for multimodal
        return inputs[:,self.num_action_questions:,:] # truncate next action pos


class OneHotPostprocessor(hk.Module):
    def __init__(self, observable, dims_config, name=None):
        super().__init__(name=name)
        self.observable = observable
        self.dims_config = dims_config

    def __call__(self, inputs, is_training, **kwargs):
        logits = get_position_logits(self.observable.dims,
            inputs, self.dims_config)
        return logits


class AbsolutePostprocessor(hk.Module):
    def __init__(*args, **kwargs):
        raise NotImplementedError  # TODO
    # def __call__(...):


class FuzzyPostprocessor(hk.Module):
    def __init__(*args, **kwargs):
        raise NotImplementedError  # TODO
    # def __call__(...):


class DimsPostprocessor(hk.Module):
    """
    how do we predict none of the above (N/A / None / -1)?
      we shouldn't care, since it should not matter!
    """
    # FIXME: tests for game logic to ensure that shouldn't==doesn't.
    def __init__(self, question_names, all_questions,
            dims_config, name=None):
        super().__init__(name=name)
        self.question_names = question_names
        self.dims_config = dims_config
        modalities = {name: self.get_postprocessor(name, all_questions[name])
                for name in self.question_names}
        self.multimodal = io_processors.MultimodalPostprocessor(
            modalities, key_order=self.question_names)

    def get_postprocessor(self, name, observable):
        if isinstance(observable, dims.OneHotObservable):
            return OneHotPostprocessor(observable, self.dims_config)
        if isinstance(observable, dims.AbsoluteObservable):
            return AbsolutePostprocessor(observable, self.dims_config)
        if isinstance(observable, dims.FuzzyObservable):
            return FuzzyPostprocessor(observable, self.dims_config)

    def __call__(self, inputs, is_training, modality_sizes=None):
        modality_sizes = {name: modality_sizes[name]
            for name in self.question_names}
        return self.multimodal(inputs, is_training=is_training,
                    modality_sizes=modality_sizes)


def forward_function(inputs, is_training, action_question_names,
        state_question_names, embedding_config,
        encoder_config, decoder_config,
        experiment_replace_none_randomly):
    inputs = {**inputs['action_observables'], **inputs['current_observables']}
    categorical_embed_layers = {}
    for name, dim in dims.dims.items():
        if isinstance(dim, dims.CategoricalDim):
            categorical_embed_layers[name] = hk.Embed(
                vocab_size=len(dim.labels),
                embed_dim=embedding_config['num_categorical_channels'],
                name=dim.name)
    dims_config = DimsConfig(
        categorical_embed_layers=categorical_embed_layers,
        experiment_replace_none_randomly=experiment_replace_none_randomly,
        **embedding_config)
    input_preprocessor = DimsPreprocessor(
        action_question_names + state_question_names,
        dims.questions, dims_config)
    encoder = perceiver.PerceiverEncoder(**encoder_config)
    decoder = DimsDecoder(
        num_action_questions=len(action_question_names),
        **decoder_config)
    output_postprocessor = DimsPostprocessor(
        state_question_names, dims.questions, dims_config)
    model = perceiver.Perceiver(
        encoder=encoder,
        decoder=decoder,
        input_preprocessor=input_preprocessor,
        output_postprocessor=output_postprocessor)

    return model(inputs, is_training=is_training)


def loss_fn(logits, labels, question_names, experiment_none_is_uniform_output):
    questions = {name: dims.questions[name] for name in question_names}
    loss = {}
    accuracy = {}
    for name, observable in questions.items():
        if isinstance(observable, dims.OneHotObservable):
            loss[name] = {}
            accuracy[name] = {}
            for i, dim in enumerate(observable.dims):
                pos = labels[name][:,i]
                mask = (pos != -1).reshape((pos.shape[0], -1))
                dim_logits = logits[name][dim.name]
                dim_labels = jax.nn.one_hot(pos, dim_logits.shape[-1]) * mask
                if experiment_none_is_uniform_output:
                    dim_labels = dim_labels + (1 - mask) / dim_logits.shape[-1]
                dim_loss = perceiver_utils.softmax_cross_entropy(
                    dim_logits, dim_labels)
                loss[name][dim.name] = dim_loss
                accuracy[name][dim.name] = perceiver_utils.topk_correct(
                    dim_logits, pos, topk=(1,))['top_1_acc'] + (1 - mask)
        else:
            raise NotImplementedError

    def mean_and_scale(x):
        return jnp.mean(x) / jax.device_count()

    loss = jax.tree_map(mean_and_scale, loss)
    accuracy = jax.tree_map(jnp.mean, accuracy)

    def flatten_means(nested):
        if not isinstance(nested, dict):
            return None, nested
        flattened = {}
        for key, value in nested.items():
            flat_per_key, mean_per_key = flatten_means(value)
            if flat_per_key is not None:
                for sub_key, sub_value in flat_per_key.items():
                    flattened[f"{key}_{sub_key}"] = sub_value
            flattened[key] = mean_per_key
        mean = jnp.mean(jnp.array([flattened[key] for key in nested.keys()]))
        return flattened, mean


    flat_loss, mean_loss = flatten_means(loss)
    flat_acc, mean_acc = flatten_means(accuracy)

    metrics = {"loss": mean_loss, "acc": mean_acc}
    metrics.update({key + "_loss": value for key, value in flat_loss.items()})
    metrics.update({key + "_acc": value for key, value in flat_acc.items()})

    return mean_loss, metrics
