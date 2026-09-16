"""Versioned NDJSON control messages."""

from __future__ import annotations

import json
from typing import Any


PROTOCOL_VERSION = 1
EVENT_KINDS = {"REGISTER_GROUP", "DECLARE", "OFFER", "SUBMITTED", "COMPLETED", "INPUT_CLOSED", "FAILED"}
CONTROL_KINDS = {"READY", "GRANT", "FINISHED", "FAILED"}


class ProtocolError(ValueError):
    pass


def encode_message(message: dict[str, Any]) -> bytes:
    _validate_envelope(message)
    return (json.dumps(message, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def decode_message(line: bytes | str) -> dict[str, Any]:
    try:
        message = json.loads(line)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON control message") from exc
    if not isinstance(message, dict):
        raise ProtocolError("control message must be an object")
    _validate_envelope(message)
    return message


def event_message(kind: str, epoch: int, endpoint: int, event_seq: int, payload: dict[str, Any]) -> dict[str, Any]:
    if kind not in EVENT_KINDS:
        raise ProtocolError(f"invalid event kind {kind!r}")
    message = {"protocol": PROTOCOL_VERSION, "kind": kind, "epoch": epoch, "endpoint": endpoint, "event_seq": event_seq, "payload": payload}
    _validate_envelope(message)
    return message


def hello_message(epoch: int, endpoint: int) -> dict[str, Any]:
    return {"protocol": PROTOCOL_VERSION, "kind": "HELLO", "epoch": epoch, "endpoint": endpoint, "payload": {}}


def _validate_envelope(message: dict[str, Any]) -> None:
    if message.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol {message.get('protocol')!r}")
    kind = message.get("kind")
    if not isinstance(kind, str):
        raise ProtocolError("message kind is required")
    if not isinstance(message.get("epoch"), int) or not isinstance(message.get("endpoint"), int):
        raise ProtocolError("message epoch and endpoint are required integers")
    if kind not in {"HELLO", *CONTROL_KINDS} and (not isinstance(message.get("event_seq"), int) or message["event_seq"] <= 0):
        raise ProtocolError("event_seq must be a positive integer")
    if not isinstance(message.get("payload"), dict):
        raise ProtocolError("message payload must be an object")
