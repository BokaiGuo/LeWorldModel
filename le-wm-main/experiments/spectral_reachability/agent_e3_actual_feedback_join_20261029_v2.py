"""Strict f32 history join; no authority, neural, physical or production entry."""
import numpy as np
from experiments.spectral_reachability import agent_e3_actual_feedback_join_20261029_v1 as v1


def joined_context(context, prior_raw5, submitted_raw5, returned_raw5, observed_latent,
                   mean, std, **authority):
    prior = v1.f32(prior_raw5, (5, 2))
    mu = v1.f32(mean, (2,)); sigma = v1.f32(std, (2,))
    v1.need((sigma > 0).all(), 'positive normalizer')
    history = v1.f32(context['action_history'], (2, 10))
    expected = ((prior - mu) / sigma).reshape(10)
    v1.need(expected.tobytes() == history[-1].tobytes(), 'prior raw5 does not match last history token bits')
    return v1.joined_context(context, prior_raw5, submitted_raw5, returned_raw5,
                             observed_latent, mean, std, **authority)
