"""Versioned length-prefixed JSON, shared without importing ML dependencies."""
from __future__ import annotations

import asyncio
import json
import struct
from typing import BinaryIO

VERSION = 1
MAX_FRAME_BYTES = 64 * 1024 * 1024


class ProtocolError(ValueError):
    pass


def encode_frame(message: dict) -> bytes:
    if message.get("version") != VERSION:
        raise ProtocolError("Unsupported protocol version")
    payload = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if not 0 < len(payload) <= MAX_FRAME_BYTES:
        raise ProtocolError("Invalid frame size")
    return struct.pack("!I", len(payload)) + payload


def decode_payload(payload: bytes) -> dict:
    try:
        message = json.loads(payload.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ProtocolError("Nonfinite JSON")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Invalid JSON frame") from exc
    if not isinstance(message, dict) or message.get("version") != VERSION:
        raise ProtocolError("Unsupported protocol version")
    if not isinstance(message.get("request_id"), str) or not message["request_id"]:
        raise ProtocolError("Missing request ID")
    if not isinstance(message.get("op"), str) or not isinstance(message.get("payload"), dict):
        raise ProtocolError("Invalid message shape")
    return message


def _read_exact(stream: BinaryIO, size: int, *, allow_eof: bool = False) -> bytes | None:
    data = bytearray()
    while len(data) < size:
        piece = stream.read(size - len(data))
        if not piece:
            if not data and allow_eof:
                return None
            raise ProtocolError("Truncated frame")
        data.extend(piece)
    return bytes(data)


def read_frame(stream: BinaryIO) -> dict | None:
    header = _read_exact(stream, 4, allow_eof=True)
    if header is None:
        return None
    size = struct.unpack("!I", header)[0]
    if not 0 < size <= MAX_FRAME_BYTES:
        raise ProtocolError("Invalid frame size")
    return decode_payload(_read_exact(stream, size))


async def read_frame_async(stream: asyncio.StreamReader) -> dict:
    header = await stream.readexactly(4)
    size = struct.unpack("!I", header)[0]
    if not 0 < size <= MAX_FRAME_BYTES:
        raise ProtocolError("Invalid frame size")
    return decode_payload(await stream.readexactly(size))
