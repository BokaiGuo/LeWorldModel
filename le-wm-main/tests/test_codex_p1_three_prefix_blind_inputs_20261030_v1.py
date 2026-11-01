import hashlib
import io

import numpy as np
import pytest
from PIL import Image

from experiments.spectral_reachability import codex_p1_three_prefix_blind_inputs_20261030_v1 as m


def fixture():
    pixels = np.zeros((224, 224, 3), np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG')
    png = stream.getvalue()
    cp = dict(file_sha256=hashlib.sha256(png).hexdigest(),
              pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    anchor = dict(index=0, logical_pair=0, role='selected', checkpoint=cp,
                  prefix5_float32_hex=np.zeros((5, 2), '<f4').tobytes().hex(), task_id='id')
    task = dict(binding=dict(task_id='id'), action_mean=[0., 0.], action_std=[1., 1.],
                root_context=dict(latent_history=np.zeros((3, 192)).tolist(),
                                  action_history=np.zeros((2, 10)).tolist()),
                goal=np.zeros(192).tolist())
    return anchor, task, np.zeros((10, 2), np.float32), png


def test_whitelist_rejects_hidden_or_future():
    row = m.make_row(*fixture())
    for key in ('state', 'snapshot', 'future_rgb', 'reward', 'outcome', 'path'):
        with pytest.raises(ValueError, match='whitelist'):
            m.validate_row(dict(row, **{key: 'forbidden'}))


def test_actual_history_and_frame_tamper():
    anchor, task, warm, png = fixture()
    warm[0, 0] = 1
    with pytest.raises(ValueError, match='normalization'):
        m.make_row(anchor, task, warm, png)
    with pytest.raises(ValueError, match='PNG SHA'):
        m.make_row(anchor, task, np.zeros((10, 2)), png+b'x')


def test_role_and_prefix_fail_closed():
    row = m.make_row(*fixture())
    with pytest.raises(ValueError, match='roster'):
        m.validate_row(dict(row, role='greedy'))
    with pytest.raises(ValueError, match='prefix'):
        m.validate_row(dict(row, prefix_raw_f32_hex='00'))
