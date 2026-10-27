"""gbx-codecs command line.

python -m gbx_codecs serve                       # the process gbx-lm starts
python -m gbx_codecs check MODEL_DIR             # load the codec, round-trip a tone
python -m gbx_codecs encode MODEL_DIR IN.wav OUT.npy
python -m gbx_codecs decode MODEL_DIR IN.npy OUT.wav
"""

from __future__ import annotations

import argparse
import sys
import wave

import numpy as np

from .worker import CACHE_LIMIT, MAX_DECODE_FRAMES


def _codec(model_dir: str):
    from .fish_s1 import DAC

    return DAC.from_pretrained(model_dir)


def _decode(codec, codes: np.ndarray) -> np.ndarray:
    import mlx.core as mx

    codec.reset_streaming_state()
    parts = [
        np.array(
            codec.streaming_step(mx.array(codes[:, b : b + MAX_DECODE_FRAMES])[None])[0, 0].astype(mx.float32)
        )
        for b in range(0, codes.shape[1], MAX_DECODE_FRAMES)
    ]
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def _write_wav(path: str, samples: np.ndarray, rate: int) -> None:
    from .audio import to_pcm16

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(to_pcm16(samples))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m gbx_codecs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="speak the codec protocol on stdin/stdout")
    c = sub.add_parser("check", help="load the codec and round-trip a test tone")
    c.add_argument("model_dir")
    e = sub.add_parser("encode", help="audio file to codes")
    e.add_argument("model_dir")
    e.add_argument("audio")
    e.add_argument("out")
    d = sub.add_parser("decode", help="codes to a wav file")
    d.add_argument("model_dir")
    d.add_argument("codes")
    d.add_argument("out")
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        from .worker import main as serve

        serve()
        return
    import mlx.core as mx

    mx.set_cache_limit(CACHE_LIMIT)
    codec = _codec(a.model_dir)
    if a.cmd == "check":
        rate = codec.sample_rate
        t = np.arange(rate * 2) / rate
        tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        idx, lengths = codec.encode(mx.array(tone)[None, None, :])
        codes = np.array(idx[0, :, : int(lengths[0].item())]).astype(np.int32)
        back = _decode(codec, codes)[: len(tone)]
        corr = float(np.corrcoef(back, tone[: len(back)])[0, 1])
        print(
            f"codec OK: {codes.shape[0]} codebooks, {codes.shape[1]} frames for 2 s, "
            f"{rate} Hz, round-trip correlation {corr:.3f}"
        )
        sys.exit(0 if corr > 0.5 else 1)
    if a.cmd == "encode":
        from .audio import read_audio

        audio = read_audio(a.audio, codec.sample_rate)
        idx, lengths = codec.encode(mx.array(audio)[None, None, :])
        np.save(a.out, np.array(idx[0, :, : int(lengths[0].item())]).astype(np.int32))
    elif a.cmd == "decode":
        _write_wav(a.out, _decode(codec, np.load(a.codes).astype(np.int32)), codec.sample_rate)


if __name__ == "__main__":
    main()
