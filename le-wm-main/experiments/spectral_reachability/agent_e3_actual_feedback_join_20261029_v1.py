"""Fixture-only E3 same-episode physical/encoding history join.

This is a numerical joining primitive, NOT an authority or production consumer.
Callers must independently authenticate every physical and encoding receipt.
"""
from copy import deepcopy
import hashlib
import numpy as np

NAME = 'agent_e3_actual_feedback_join_20261029_v1'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def f32(value, shape):
    a = np.asarray(value)
    need(a.shape == shape and a.dtype == np.dtype('float32') and np.isfinite(a).all(),
         'finite literal float32 shape')
    return a


def sha(data):
    need(type(data) is bytes and len(data) > 0, 'nonempty literal bytes')
    return hashlib.sha256(data).hexdigest()


def joined_context(context, prior_raw5, submitted_raw5, returned_raw5, observed_latent,
                   mean, std, *, expected_rgb_sha, rgb_bytes, encoding_input_sha,
                   episode_id, physical_episode_id, encoder_episode_id,
                   start, end, physical_end, encoding_boundary, stopped):
    """Build node context only from matching same-episode returned/encoded inputs.

    The caller MUST authenticate source receipts, decoder pixels, model and encoding
    provenance separately; hashes alone do not authenticate who encoded the bytes.
    """
    need(type(episode_id) is str and episode_id and episode_id == physical_episode_id == encoder_episode_id,
         'same episode identity')
    need(type(start) is int and start in (5, 10) and type(end) is int and end == start + 5
         and type(physical_end) is int and physical_end == end
         and type(encoding_boundary) is int and encoding_boundary == end,
         'same actual boundary and complete packet')
    need(type(stopped) is bool and not stopped, 'no next planner after native stop')
    need(type(expected_rgb_sha) is str and sha(rgb_bytes) == expected_rgb_sha == encoding_input_sha,
         'physical RGB bytes joined to encoder input')
    need(type(context) is dict and 'latent_history' in context and 'action_history' in context,
         'actual context')
    z = f32(context['latent_history'], (3, 192))
    f32(context['action_history'], (2, 10))
    znew = f32(observed_latent, (192,))
    old = f32(prior_raw5, (5, 2))
    sent = f32(submitted_raw5, (5, 2))
    got = f32(returned_raw5, (5, 2))
    need(sent.tobytes() == got.tobytes(), 'returned raw action bits differ from submitted')
    need((np.abs(old) <= 1).all() and (np.abs(got) <= 1).all(), 'raw action box')
    mu = f32(mean, (2,)); sigma = f32(std, (2,))
    need((sigma > 0).all(), 'positive normalizer')
    normalized = ((np.concatenate((old, got)) - mu) / sigma).reshape(2, 10)
    out = deepcopy(context)
    out['latent_history'] = np.concatenate((z[1:], znew[None])).astype(np.float32)
    out['action_history'] = normalized.astype(np.float32)
    return out


def full_budget():
    return dict(cells=16, nodes=48, predict=6528, scalar_reverse=1536,
                new_encoder=32, new_projector=32, public=480, physics=4864,
                explicit_render=512, per_cell_cap=512*1024**2,
                per_cell_reserve=64*1024**2, per_cell_wall_seconds=1800,
                total_transport_proven=False, engine_backward='unknown', flops='unknown')
