"""Against real weights: set GBX_CODECS_FISH_S1 to a model directory holding
codec.safetensors (for example Fish Audio Drama 3's). Skipped otherwise."""

import os

import numpy as np
import pytest

MODEL = os.environ.get("GBX_CODECS_FISH_S1")
pytestmark = pytest.mark.skipif(not MODEL, reason="GBX_CODECS_FISH_S1 not set")


@pytest.fixture(scope="module")
def codec():
    from gbx_codecs.fish_s1 import DAC

    return DAC.from_pretrained(MODEL)


def _codes(codec, seconds=3.0):
    import mlx.core as mx

    rate = codec.sample_rate
    t = np.arange(int(rate * seconds)) / rate
    tone = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    idx, lengths = codec.encode(mx.array(tone)[None, None, :])
    return tone, np.array(idx[0, :, : int(lengths[0].item())]).astype(np.int32)


def _decode(codec, codes, step):
    import mlx.core as mx

    codec.reset_streaming_state()
    parts = [
        np.array(codec.streaming_step(mx.array(codes[:, i : i + step])[None])[0, 0].astype(mx.float32))
        for i in range(0, codes.shape[1], step)
    ]
    return np.concatenate(parts)


def test_a_tone_survives_the_round_trip(codec):
    tone, codes = _codes(codec)
    assert codes.shape[0] == 10
    back = _decode(codec, codes, 64)[: len(tone)]
    assert np.corrcoef(back, tone[: len(back)])[0, 1] > 0.9


def test_the_piece_size_changes_nothing_audible(codec):
    _, codes = _codes(codec)
    whole = _decode(codec, codes, 64)
    for step in (1, 8, 32):
        assert np.abs(_decode(codec, codes, step) - whole).max() < 2 / 32768
