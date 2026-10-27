import numpy as np

from gbx_codecs.audio import Resampler, to_pcm16


def tone(rate, seconds=1.0, hz=1000.0):
    t = np.arange(int(rate * seconds)) / rate
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_pieces_resample_exactly_as_the_whole():
    x = tone(44100) + 0.1 * np.random.default_rng(0).standard_normal(44100).astype(np.float32)
    whole = Resampler(44100, 24000)(x, final=True)
    r = Resampler(44100, 24000)
    cuts = [0, 1, 2047, 2048, 5000, 30000, 44100]
    pieces = [r(x[a:b], final=b == len(x)) for a, b in zip(cuts, cuts[1:])]
    assert np.array_equal(np.concatenate(pieces), whole)


def test_length_and_tone_survive():
    y = Resampler(44100, 24000)(tone(44100), final=True)
    assert len(y) == 24000
    spectrum = np.abs(np.fft.rfft(y[2000:-2000]))
    peak_hz = spectrum.argmax() * 24000 / len(y[2000:-2000])
    assert abs(peak_hz - 1000) < 5
    assert abs(np.abs(y[2000:-2000]).max() - 0.5) < 0.01


def test_same_rate_is_untouched():
    x = tone(24000)
    assert np.array_equal(Resampler(24000, 24000)(x, final=True), x)


def test_pcm16_clips_and_is_little_endian():
    raw = to_pcm16(np.array([0.0, 1.5, -1.5, 0.5], np.float32))
    assert np.frombuffer(raw, "<i2").tolist() == [0, 32767, -32767, 16383]
