"""Compact binary exchange for vector or hidden-state intermediates."""

from dataclasses import dataclass
import hashlib
import struct
from typing import Iterable

import numpy as np


@dataclass
class StatePacket:
    state_id: str
    source_agent: str
    target_agent: str
    dtype: str
    shape: tuple
    encoding: str
    checksum: str
    payload: bytes

    @property
    def payload_bytes(self) -> int:
        return len(self.payload)


class StateExchange:
    """Encode and restore normalized vectors without JSON float expansion."""

    def encode(
        self, vector: Iterable[float], *, state_id: str, source_agent: str,
        target_agent: str, encoding: str = "float16"
    ) -> StatePacket:
        values = np.asarray(list(vector), dtype=np.float32)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError("state vectors must be nonempty, finite and one-dimensional")
        if encoding == "float32":
            payload = values.astype("<f4").tobytes()
            dtype = "float32"
        elif encoding == "float16":
            payload = values.astype("<f2").tobytes()
            dtype = "float16"
        elif encoding == "int8":
            scale = max(float(np.max(np.abs(values))), 1e-8)
            payload = struct.pack("<f", scale) + np.rint(values / scale * 127).astype("i1").tobytes()
            dtype = "int8_scaled"
        else:
            raise ValueError(f"unsupported encoding: {encoding}")
        packet = StatePacket(
            state_id, source_agent, target_agent, dtype, tuple(values.shape), encoding,
            hashlib.sha256(payload).hexdigest(), payload,
        )
        self.decode(packet)
        return packet

    def decode(self, packet: StatePacket) -> np.ndarray:
        if not packet.state_id or not packet.source_agent or not packet.target_agent:
            raise ValueError("state id and agent endpoints are required")
        if len(packet.shape) != 1 or not isinstance(packet.shape[0], int) or packet.shape[0] < 1:
            raise ValueError("state shape must describe a nonempty vector")
        formats = {"float32": ("float32", 4, 0), "float16": ("float16", 2, 0),
                   "int8": ("int8_scaled", 1, 4)}
        if packet.encoding not in formats:
            raise ValueError(f"unsupported encoding: {packet.encoding}")
        dtype, width, header = formats[packet.encoding]
        if packet.dtype != dtype or len(packet.payload) != packet.shape[0] * width + header:
            raise ValueError("state dtype, shape and payload length do not agree")
        if hashlib.sha256(packet.payload).hexdigest() != packet.checksum:
            raise ValueError("state checksum mismatch")
        if packet.encoding == "float32":
            values = np.frombuffer(packet.payload, dtype="<f4")
        elif packet.encoding == "float16":
            values = np.frombuffer(packet.payload, dtype="<f2").astype(np.float32)
        elif packet.encoding == "int8":
            scale = struct.unpack("<f", packet.payload[:4])[0]
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("invalid quantization scale")
            values = np.frombuffer(packet.payload[4:], dtype="i1").astype(np.float32) * scale / 127
        else:
            raise ValueError(f"unsupported encoding: {packet.encoding}")
        if not np.isfinite(values).all():
            raise ValueError("restored state contains non-finite values")
        return values.reshape(packet.shape)
