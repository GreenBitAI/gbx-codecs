"""Reading reference audio, resampling, and 16-bit PCM."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np


def read_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Mono float32 samples of an audio file at ``sample_rate``.

    WAV (8/16/24/32-bit PCM) is read with the standard library; anything else
    (flac, mp3, float WAV) needs ``miniaudio`` -- ``pip install gbx-codecs[formats]``
    -- which has no wheel for Linux on ARM and builds from source there.
    """
    try:
        return _read_wav(path, sample_rate)
    except (wave.Error, EOFError, ValueError):
        pass
    try:
        import miniaudio
    except ImportError as e:
        raise ValueError(
            f"{path}: only PCM WAV can be read without miniaudio; "
            "convert it to WAV, or `pip install gbx-codecs[formats]`"
        ) from e
    decoded = miniaudio.decode_file(
        str(path),
        output_format=miniaudio.SampleFormat.FLOAT32,
        nchannels=1,
        sample_rate=sample_rate,
    )
    return np.frombuffer(decoded.samples, dtype=np.float32).copy()


def _read_wav(path: str | Path, sample_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        width, channels, rate = w.getsampwidth(), w.getnchannels(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = (np.where(v >= 1 << 23, v - (1 << 24), v)).astype(np.float32) / float(1 << 23)
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"{path}: {width}-byte WAV samples")
    x = x.reshape(-1, channels).mean(axis=1) if channels > 1 else x
    return Resampler(rate, sample_rate)(x, final=True)


def to_pcm16(samples: np.ndarray) -> bytes:
    """Little-endian signed 16-bit PCM."""
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class Resampler:
    """Streaming band-limited resampling by a rational factor.

    Windowed-sinc interpolation with a Kaiser window, computed for every output
    phase once. Feeding a signal in pieces gives exactly what feeding it whole
    does: the input the next piece's outputs still need is carried over.
    """

    def __init__(self, source_rate: int, target_rate: int, half_width: int = 16, beta: float = 8.6):
        g = math.gcd(source_rate, target_rate)
        self.up, self.down = target_rate // g, source_rate // g
        self.identity = self.up == self.down
        self.half_width = half_width
        cutoff = min(1.0, self.up / self.down) * 0.95
        # taps[p, j]: weight of input (base + j - half_width + 1) for an output at
        # fractional input position base + p / up.
        offsets = (
            np.arange(-half_width + 1, half_width + 1)[None, :] - (np.arange(self.up) / self.up)[:, None]
        )
        window = np.kaiser(2 * half_width + 1, beta)
        x = offsets / half_width
        w = np.interp(x, np.linspace(-1, 1, 2 * half_width + 1), window)
        self.taps = (cutoff * np.sinc(cutoff * offsets) * w).astype(np.float32)
        self.reset()

    def reset(self) -> None:
        # Leading zeros stand for the signal before the first sample.
        self._buffer = np.zeros(self.half_width - 1, np.float32)
        self._consumed = 0  # input samples dropped from the buffer's front
        self._next = 0  # index of the next output sample

    def __call__(self, samples: np.ndarray, final: bool = False) -> np.ndarray:
        if self.identity:
            return np.asarray(samples, np.float32)
        buf = np.concatenate([self._buffer, np.asarray(samples, np.float32)])
        if final:
            buf = np.concatenate([buf, np.zeros(self.half_width, np.float32)])
        lead = self.half_width - 1
        available = self._consumed + len(buf) - lead - self.half_width  # last usable centre
        # The last output whose centre, floor(n * down / up), is below `available`.
        last = (available * self.up - 1) // self.down if available > 0 else -1
        if final:
            total_in = self._consumed + len(buf) - lead - self.half_width
            last = min(last, math.ceil(total_in * self.up / self.down) - 1)
        n = np.arange(self._next, last + 1)
        if len(n) == 0:
            out = np.zeros(0, np.float32)
        else:
            pos = n * self.down
            base = pos // self.up - self._consumed + lead  # index into buf of the centre
            phase = pos % self.up
            idx = base[:, None] + np.arange(-self.half_width + 1, self.half_width + 1)[None, :]
            out = (buf[idx] * self.taps[phase]).sum(axis=1).astype(np.float32)
            self._next = int(last) + 1
        # Keep what the next output still reads.
        keep_from = (self._next * self.down) // self.up - self._consumed + lead - (self.half_width - 1)
        keep_from = max(0, min(int(keep_from), len(buf)))
        self._buffer = buf[keep_from:]
        self._consumed += keep_from
        return out
