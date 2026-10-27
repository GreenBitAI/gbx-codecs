"""The wire format between gbx-lm and a codec process, over its stdin and stdout.

One message is ``MAGIC``, a 4-byte header length, an 8-byte payload length (both
big-endian), a UTF-8 JSON header, and the payload bytes. Requests carry ``op``;
replies carry ``ok`` and, when false, ``error``. Arrays travel as raw bytes with
their ``dtype`` and ``shape`` in the header.

Kept to the standard library so that the server side can copy it verbatim.
"""

from __future__ import annotations

import json
import struct
from typing import IO, Any

MAGIC = b"GBXC"
VERSION = 1
_LENGTHS = struct.Struct(">IQ")


def write_message(stream: IO[bytes], header: dict[str, Any], payload: bytes = b"") -> None:
    body = json.dumps(header, separators=(",", ":")).encode()
    stream.write(MAGIC + _LENGTHS.pack(len(body), len(payload)) + body + payload)
    stream.flush()


def _read_exactly(stream: IO[bytes], n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            raise EOFError("codec stream closed")
        data += chunk
    return bytes(data)


def read_message(stream: IO[bytes]) -> tuple[dict[str, Any], bytes]:
    prefix = _read_exactly(stream, len(MAGIC) + _LENGTHS.size)
    if prefix[: len(MAGIC)] != MAGIC:
        raise ValueError("not a gbx-codecs message")
    header_len, payload_len = _LENGTHS.unpack(prefix[len(MAGIC) :])
    header = json.loads(_read_exactly(stream, header_len))
    return header, _read_exactly(stream, payload_len)
