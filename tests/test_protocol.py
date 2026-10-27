import io

import numpy as np

from gbx_codecs.protocol import read_message, write_message
from gbx_codecs.worker import serve


def roundtrip(requests):
    stdin = io.BytesIO()
    for header, payload in requests:
        write_message(stdin, header, payload)
    stdin.seek(0)
    stdout = io.BytesIO()
    serve(stdin, stdout)
    stdout.seek(0)
    replies = []
    while stdout.tell() < len(stdout.getvalue()):
        replies.append(read_message(stdout))
    return replies


def test_a_message_survives_the_wire():
    buf = io.BytesIO()
    payload = np.arange(6, dtype=np.int32).tobytes()
    write_message(buf, {"op": "x", "shape": [2, 3]}, payload)
    buf.seek(0)
    header, data = read_message(buf)
    assert header == {"op": "x", "shape": [2, 3]}
    assert data == payload


def test_hello_names_the_codecs_and_errors_come_back_as_replies():
    replies = roundtrip(
        [({"op": "hello"}, b""), ({"op": "decode", "shape": [10, 1]}, b"\0" * 40), ({"op": "quit"}, b"")]
    )
    assert replies[0][0]["ok"] and "fish_s1" in replies[0][0]["codecs"]
    assert replies[1][0]["ok"] is False and "load" in replies[1][0]["error"]
    assert replies[2][0] == {"ok": True}


def test_an_unknown_codec_is_refused():
    (reply, _), _ = roundtrip([({"op": "load", "codec": "nope", "path": "/x"}, b""), ({"op": "quit"}, b"")])
    assert reply["ok"] is False and "fish_s1" in reply["error"]
