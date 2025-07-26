from absl.testing import absltest
import jax
import jax.numpy as jnp
import haiku as hk

import scaffold.perceiver.dims_perceiver as dims_perceiver
import scaffold.dims as dims

import pdb
import pprint


def get_input_dict(question_names, raw_inputs):
    return {name: jnp.array([d[i] for d in raw_inputs])
        for i, name in enumerate(question_names)}


def get_dims_config():
    num_fourier_channels = 8
    num_categorical_channels = 17
    question_name_embedding_size = 2
    categorical_embed_layers = {}
    experiment_replace_none_randomly=False
    for name, dim in dims.dims.items():
        if isinstance(dim, dims.CategoricalDim):
            categorical_embed_layers[name] = hk.Embed(vocab_size=len(dim.labels),
                embed_dim=num_categorical_channels, name=dim.name)
    return dims_perceiver.DimsConfig(
        num_fourier_channels=num_fourier_channels,
        num_categorical_channels=num_categorical_channels,
        question_name_embedding_size=question_name_embedding_size,
        categorical_embed_layers=categorical_embed_layers,
        experiment_replace_none_randomly=experiment_replace_none_randomly)


class DimsPreprocessorTest(absltest.TestCase):
    def test_range_and_circle(self):
        question_names = ('next_bid', 'pass_position', 'next_to_act')
        raw_inputs = (((6, 4, 1), (0,), (2,)),
                     ((-1, 2, 0), (1,), (-1,)))
        def apply_test():
            dims_config = get_dims_config()
            preprocessor = dims_perceiver.DimsPreprocessor(
                question_names, dims.questions, dims_config)
            return (dims_config, preprocessor(get_input_dict(question_names, raw_inputs), is_training=False))
        rng = jax.random.PRNGKey(1)
        transform = hk.transform(apply_test)
        params = transform.init(rng=rng)
        dims_config, (output, _, _) = transform.apply(params, rng=rng)
        self.assertEqual(output[0, 2, -1], output[1, 2, -1])
        self.assertEqual(output.shape, (2, 3,
            dims_config.num_fourier_channels * 6 + \
            dims_config.num_categorical_channels + 2))

    def test_categorical(self):
        question_names = ('last_bid', 'pass_position', 'next_to_act', 'stage')
        raw_inputs = (((6, 4, 1, 0), (0,), (2,), (0,)),
                     ((6, -1, 0, 1), (1,), (1,), (1,)))
        def apply_test():
            dims_config = get_dims_config()
            preprocessor = dims_perceiver.DimsPreprocessor(
                question_names, dims.questions, dims_config)
            return (dims_config, preprocessor(get_input_dict(question_names, raw_inputs), is_training=False))

        rng = jax.random.PRNGKey(1)
        transform = hk.transform(apply_test)
        params = transform.init(rng=rng)
        dims_config, (output, _, _) = transform.apply(params, rng=rng)
        self.assertEqual(output[0, 3, -1], output[1, 3, -1])
        self.assertEqual(output.shape, (2, 4,
            dims_config.num_fourier_channels * 6 + \
            dims_config.num_categorical_channels + 2))


class DimsPostprocessorTest(absltest.TestCase):
    def test_onehot(self):
        question_names = ('last_bid', 'pass_position', 'next_to_act', 'stage')
        raw_inputs = (((6, 4, 1, 0), (0,), (2,), (0,)),
                      ((6, -1, 0, 1), (1,), (1,), (1,)))
        def apply_test():
            dims_config = get_dims_config()
            preprocessor = dims_perceiver.DimsPreprocessor(
                question_names, dims.questions, dims_config)
            postprocessor = dims_perceiver.DimsPostprocessor(
                question_names, dims.questions, dims_config)
            pos_embeds, modality_sizes, _ = preprocessor(
                    get_input_dict(question_names, raw_inputs),
                    is_training=False)
            logits = postprocessor(pos_embeds, is_training=False,
                modality_sizes=modality_sizes)
            return logits

        rng = jax.random.PRNGKey(1)
        transform = hk.transform(apply_test)
        params = transform.init(rng=rng)
        logits = transform.apply(params, rng=rng)
        input_dict = get_input_dict(question_names, raw_inputs)
        for name, values in input_dict.items():
            for i, dim in enumerate(dims.questions[name].dims):
                predicted = jax.nn.softmax(logits[name][dim.name])
                expected = tuple(d[i] for d in values)
                for j, v in enumerate(expected):
                    if v != -1:
                        # TODO AY check delta is so large (because our fourier
                        # embeddings are not always orthogonal, but should be)
                        self.assertAlmostEqual(predicted[j,0,v], 1.0, delta=0.11)
                    else:
                        pass


if __name__ == "__main__":
    absltest.main()
