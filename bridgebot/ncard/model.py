from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
from third_party.perceiver import perceiver
from third_party.perceiver import io_processors
from third_party.perceiver.train import utils as perceiver_train_utils
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from bridgebot.ncard import chords

import pdb


PreprocessorOutputT = Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]


class ChordPreprocessor(hk.Module):
    def __init__(self, config, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.config = config

    def __call__(self, inputs: jnp.ndarray, *,
             is_training: bool,
             pos: Optional[jnp.ndarray] = None,
             network_input_is_1d: bool = True) -> PreprocessorOutputT:
        
        input_inputs = jnp.concatenate([
            jnp.reshape(
                inputs["input_par_outcome"],
                (-1, 1, inputs["input_par_outcome"].shape[-1])),
            inputs["input_view_ids"]], axis=1)
        embeddings = self.embedding_layer(input_inputs)
        embeddings = embeddings.sum(axis=2)
        positions = perceiver.position_encoding.FourierPositionEncoding(
                index_dims=embeddings.shape[1:-1],
                num_bands=self.config.embedding_depth // 2,
                concat_pos=False)(embeddings.shape[0])
        positions = positions.reshape(embeddings.shape)
        query_inputs = inputs["query_ids"]
        query_embeddings = self.embedding_layer(query_inputs)
        query_embeddings = query_embeddings.sum(axis=2)
        # NOTE: we use inputs_without_pos to convey query embeddings to decoder.

        return embeddings + positions, None, query_embeddings


class ChordPostprocessor(hk.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, inputs: jnp.ndarray, *,
                 is_training: bool,
                 pos: Optional[jnp.ndarray] = None,
                 modality_sizes: Optional[object] = None) -> jnp.ndarray:
        return inputs


class ChordDecoder(perceiver.BasicDecoder):
    def __init__(self):
        super().__init__(output_num_channels=250,
                         position_encoding_type='none',
                         )  # TODO

    def decoder_query(self, inputs, modality_sizes=None,
                      inputs_without_pos=None, subsampled_points=None):
        return inputs_without_pos


def forward_function(inputs: dict,
                     is_training: bool,
                     chord_config: dict,
                     encoder_config: dict,
                     decoder_config: dict):
    embedding_layer = hk.Embed(
            vocab_size=chord_config.vocab_size,
            embed_dim=chord_config.embedding_depth)
    input_preprocessor = ChordPreprocessor(chord_config, embedding_layer)
    encoder = perceiver.PerceiverEncoder(**encoder_config)
    decoder = ChordDecoder(**decoder_config)
    output_postprocessor = ChordPostprocessor(chord_config)
    model = perceiver.Perceiver(
        encoder=encoder,
        decoder=decoder,
        input_preprocessor=input_preprocessor,
        output_postprocessor=output_postprocessor)

    t = model(inputs, is_training=is_training)
    logits = emb_dec(t, embedding_layer)
    return logits, t


def emb_dec(t, embedding_layer):
    return io_processors.EmbeddingDecoder(
            embedding_matrix=embedding_layer.embeddings)(t)


def loss_function(codecs, logits, inputs, labels):
    loss = {}
    loss["bid"] = {}
    loss["play"] = {}
    accuracy = {}
    target_ids = labels["target_ids"]
    value_gt_loss = -chords.batch_bool_log_likelihood(
            logits[:,0,:], target_ids[:,0], YES=2, NO=3)
    value_geq_loss = -chords.batch_bool_log_likelihood(
            logits[:,1,:], target_ids[:,1], YES=2, NO=3)
    outcome_loss = -chords.batch_log_likelihood(
        logits[:,2,:], codecs["outcome"], target_ids[:,2])
    policy_ll = -chords.batch_bool_log_likelihood(
            logits[:,3:,:], target_ids[:,3:], YES=2, NO=3)
    policy_lsm = jax.nn.log_softmax(policy_ll, where=labels["target_mask"])
    policy_loss_by_action = jnp.where(
            labels["target_mask"],
           -labels["target_policy"] * policy_lsm, 0.)
    policy_loss = jnp.sum(policy_loss_by_action, axis=1)
    loss["bid"]["value_gt"] = jnp.mean(value_gt_loss,
                                       where=~inputs["is_playing"])
    loss["play"]["value_gt"] = jnp.mean(value_gt_loss,
                                       where=inputs["is_playing"])
    loss["bid"]["value_geq"] = jnp.mean(value_geq_loss,
                                       where=~inputs["is_playing"])
    loss["play"]["value_geq"] = jnp.mean(value_geq_loss,
                                       where=inputs["is_playing"])
    loss["bid"]["outcome"] = jnp.mean(outcome_loss,
                                       where=~inputs["is_playing"])
    loss["play"]["outcome"] = jnp.mean(outcome_loss,
                                      where=inputs["is_playing"])
    loss["bid"]["policy"] = jnp.mean(policy_loss,
                                       where=~inputs["is_playing"])
    loss["play"]["policy"] = jnp.mean(policy_loss,
                                       where=inputs["is_playing"])
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

    return mean_loss, (metrics, (logits, value_gt_loss,))


def _pad(v, n, fill):
    if len(v) >= n:
        return v[:n]
    else:
        return v + [fill] * (n - len(v))


def make_inference_features(
        game, tokenizer, view, par_result,
        max_seq_length, max_chord_width, max_legal_actions, legal_action_ixs):
    view_ids = tokenizer.tokenize_view(view)
    view_ids = _pad(view_ids, max_seq_length, [])
    view_ids = [_pad(v, max_chord_width, 0) for v in view_ids]
    par_outcome = tokenizer.tokenize_result(par_result)
    par_outcome_ids = tokenizer.tokens_to_ids(par_outcome)
    par_outcome_ids = [_pad(ids, max_chord_width, 0)
                       for ids in par_outcome_ids]
    other_query_ids = tokenizer.tokens_to_ids(
            [["value_gt"], ["value_geq"], ["outcome"]])
    query_ids = other_query_ids + tokenizer.tokenize_action_ids(
        legal_action_ixs)
    query_ids = _pad(query_ids, 3 + max_legal_actions, [])
    query_ids = [_pad(v, max_chord_width, 0) for v in query_ids]
    target_mask = np.array(
            [1.] * (len(legal_action_ixs)) + \
            [0.] * (max_legal_actions - len(legal_action_ixs)))
    return {
            "input_view_ids": view_ids,
            "input_par_outcome": par_outcome_ids,
            "query_ids": query_ids,
            "target_mask": target_mask,
    }

def make_train_features(
        game, tokenizer, deal, other_table_deal, view,
        played_game, max_chord_width, max_legal_actions, legal_action_ixs):
    raw_policy = played_game.actions[view.num_actions()].mcts_visit_fraction
    policy = [raw_policy[ix] for ix in legal_action_ixs]
    policy = _pad(policy, max_legal_actions, 0)
    side = view.next_to_act_index() % 2
    scores_here = game.table_score(deal.result, deal.vulnerability)
    scores_there = game.table_score(other_table_deal.result,
                                        deal.vulnerability)

    score_diff = (scores_here[side] or -scores_here[1 - side] or 0) - \
            (scores_there[side] or -scores_there[1 - side] or 0)
    value_gt = 1.0 * (score_diff > 0)
    value_geq = 1.0 * (score_diff >= 0)
    target_tokens = [["[YES]"]] * (3 + max_legal_actions)
    if not value_gt:
        target_tokens[0] = ["[NO]"]
    if not value_geq:
        target_tokens[1] = ["[NO]"]
    target_par_outcome = tokenizer.tokenize_result(deal.result)
    target_tokens[2] = target_par_outcome[0]
    target_ids = tokenizer.tokens_to_ids(target_tokens)
    target_ids = [_pad(ids, max_chord_width, 0)
                       for ids in target_ids]
    
    d = {
            "target_ids": target_ids,
            "target_policy": policy,
            "is_playing": 1* (view._impl._state.stage == view._impl._state.STAGE_PLAY)
    }
    return d
