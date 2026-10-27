"""The codec process: requests on stdin, replies on stdout, logs on stderr.

Operations:
    hello                                -> version, codecs
    load    codec, path                  -> sample_rate, codebooks, frame_samples
    encode  path                         -> codes (int32 [codebooks, frames])
    begin   rate                         -> starts a decode stream at that output rate
    decode  codes payload                -> pcm16 for those frames (may be empty)
    end                                  -> the stream's remaining pcm16
    quit
One stream at a time: the server serializes speech requests.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import IO, Any

import numpy as np

from . import __version__
from .audio import Resampler, read_audio, to_pcm16
from .protocol import VERSION, read_message, write_message

CODECS = ("fish_s1",)
# Decoding runs in pieces of at most this many frames, and MLX keeps at most
# CACHE_LIMIT of freed buffers. Fish S1 at 64 frames a piece peaked at 5.9 GiB
# with 7 GiB more held in MLX's cache; at 8, 2.8 GiB and 0.3, still 13x real
# time. The audio moves by at most one 16-bit step (see the README).
MAX_DECODE_FRAMES = 8
CACHE_LIMIT = 256 << 20


class Worker:
    def __init__(self) -> None:
        self.codec: Any = None
        self.name: str | None = None
        self.path: str | None = None
        self.resampler: Resampler | None = None

    def _require(self) -> Any:
        if self.codec is None:
            raise RuntimeError("no codec loaded; send `load` first")
        return self.codec

    def handle(self, header: dict[str, Any], payload: bytes) -> tuple[dict[str, Any], bytes]:
        import mlx.core as mx

        op = header.get("op")
        if op == "hello":
            return {"ok": True, "version": __version__, "protocol": VERSION, "codecs": list(CODECS)}, b""
        if op == "load":
            name, path = header["codec"], str(header["path"])
            if name not in CODECS:
                raise ValueError(f"unknown codec {name!r}; this version has {', '.join(CODECS)}")
            if (name, path) != (self.name, self.path):
                from .fish_s1 import DAC

                self.codec, self.name, self.path = DAC.from_pretrained(path), name, path
            codec = self.codec
            return {
                "ok": True,
                "sample_rate": codec.sample_rate,
                "frame_samples": codec.frame_length,
                "codebooks": codec.quantizer.semantic_quantizer.n_codebooks
                + codec.quantizer.quantizer.n_codebooks,
            }, b""
        if op == "encode":
            codec = self._require()
            audio = read_audio(header["path"], codec.sample_rate)
            if audio.size == 0:
                raise ValueError(f"{header['path']} holds no audio")
            idx, lengths = codec.encode(mx.array(audio)[None, None, :])
            codes = np.array(idx[0, :, : int(lengths[0].item())]).astype(np.int32)
            mx.clear_cache()
            return {"ok": True, "dtype": "int32", "shape": list(codes.shape)}, codes.tobytes()
        if op == "begin":
            codec = self._require()
            codec.reset_streaming_state()
            self.resampler = Resampler(codec.sample_rate, int(header.get("rate") or codec.sample_rate))
            return {"ok": True}, b""
        if op == "decode":
            codec = self._require()
            if self.resampler is None:
                raise RuntimeError("no decode stream; send `begin` first")
            codes = np.frombuffer(payload, dtype=np.int32).reshape(header["shape"])
            pieces = [
                np.array(
                    codec.streaming_step(mx.array(codes[:, i : i + MAX_DECODE_FRAMES])[None])[0, 0].astype(
                        mx.float32
                    )
                )
                for i in range(0, codes.shape[1], MAX_DECODE_FRAMES)
            ]
            samples = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
            return {"ok": True, "samples": int(samples.size)}, to_pcm16(self.resampler(samples))
        if op == "end":
            tail = self.resampler(np.zeros(0, np.float32), final=True) if self.resampler else np.zeros(0)
            self.resampler = None
            if self.codec is not None:
                self.codec.reset_streaming_state()
            mx.clear_cache()
            return {"ok": True}, to_pcm16(tail)
        raise ValueError(f"unknown op {op!r}")


def serve(stdin: IO[bytes], stdout: IO[bytes]) -> None:
    import mlx.core as mx

    mx.set_cache_limit(CACHE_LIMIT)
    worker = Worker()
    while True:
        try:
            header, payload = read_message(stdin)
        except EOFError:
            return
        if header.get("op") == "quit":
            write_message(stdout, {"ok": True})
            return
        try:
            reply, data = worker.handle(header, payload)
        except Exception as e:  # reported to the caller, which decides
            traceback.print_exc(file=sys.stderr)
            reply, data = {"ok": False, "error": f"{type(e).__name__}: {e}"}, b""
        write_message(stdout, reply, data)


def main() -> None:
    # The protocol owns stdout; anything a library prints goes to stderr instead.
    protocol_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    serve(sys.stdin.buffer, protocol_out)
