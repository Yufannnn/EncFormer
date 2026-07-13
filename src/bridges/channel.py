from __future__ import annotations

import socket
import struct
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List

import numpy as np


class Channel(ABC):
    @abstractmethod
    def send(self, tag: str, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def recv(self, tag: str) -> np.ndarray:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class _InProcessEndpoint(Channel):
    def __init__(
        self,
        send_queues: Dict[str, deque],
        recv_queues: Dict[str, deque],
    ) -> None:
        self._send = send_queues
        self._recv = recv_queues

    def send(self, tag: str, data: np.ndarray) -> None:
        if tag not in self._send:
            self._send[tag] = deque()
        self._send[tag].append(data.copy())

    def recv(self, tag: str) -> np.ndarray:
        q = self._recv.get(tag)
        if q is None or len(q) == 0:
            raise RuntimeError(
                f"InProcessChannel: no data available for tag '{tag}'. "
                "Ensure the other party sends before this party receives."
            )
        return q.popleft()

    def close(self) -> None:
        self._send.clear()
        self._recv.clear()


class InProcessChannelPair:
    @staticmethod
    def create() -> tuple[Channel, Channel]:
        a_to_b: Dict[str, deque] = {}
        b_to_a: Dict[str, deque] = {}
        ep_a = _InProcessEndpoint(send_queues=a_to_b, recv_queues=b_to_a)
        ep_b = _InProcessEndpoint(send_queues=b_to_a, recv_queues=a_to_b)
        return ep_a, ep_b


def _recv_exact(sock: socket.socket, n: int) -> bytes:

    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed before all bytes received.")
        buf.extend(chunk)
    return bytes(buf)


class SocketChannel(Channel):
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self.bytes_sent: int = 0
        self.bytes_recv: int = 0
        self.msgs_sent: int = 0
        self.msgs_recv: int = 0
        self.bytes_by_tag: Dict[str, List[int]] = {}

        import os as _os
        import time as _time

        self._rtt_one_way_s = float(_os.getenv("ENCFORMER_INJECT_RTT_MS", "0")) / 2000.0
        bw_mbps = float(_os.getenv("ENCFORMER_INJECT_BW_MBPS", "0"))
        self._bw_bytes_per_s = (bw_mbps * 1e6 / 8) if bw_mbps > 0 else 0.0
        self._time = _time

    def _tag_bucket(self, tag: str) -> List[int]:
        b = self.bytes_by_tag.get(tag)
        if b is None:
            b = [0, 0, 0, 0]
            self.bytes_by_tag[tag] = b
        return b

    def send(self, tag: str, data: np.ndarray) -> None:
        tag_b = tag.encode("utf-8")
        dtype_b = str(data.dtype).encode("utf-8")
        header = struct.pack(">I", len(tag_b)) + tag_b
        header += struct.pack(">I", len(dtype_b)) + dtype_b
        header += struct.pack(">I", data.ndim)
        for s in data.shape:
            header += struct.pack(">Q", s)
        payload = data.tobytes()
        wire = header + payload

        if self._rtt_one_way_s > 0:
            self._time.sleep(self._rtt_one_way_s)
        self._sock.sendall(wire)

        if self._bw_bytes_per_s > 0:
            self._time.sleep(len(wire) / self._bw_bytes_per_s)
        self.bytes_sent += len(wire)
        self.msgs_sent += 1
        bucket = self._tag_bucket(tag)
        bucket[0] += len(wire)
        bucket[2] += 1

    def recv(self, tag: str) -> np.ndarray:
        bytes_read = 0
        raw_tag_len = _recv_exact(self._sock, 4)
        bytes_read += 4
        tag_len = struct.unpack(">I", raw_tag_len)[0]
        recv_tag_b = _recv_exact(self._sock, tag_len)
        bytes_read += tag_len
        recv_tag = recv_tag_b.decode("utf-8")
        if recv_tag != tag:
            raise ValueError(f"SocketChannel: expected tag '{tag}', got '{recv_tag}'")
        raw_dtype_len = _recv_exact(self._sock, 4)
        bytes_read += 4
        dtype_len = struct.unpack(">I", raw_dtype_len)[0]
        dtype_str = _recv_exact(self._sock, dtype_len).decode("utf-8")
        bytes_read += dtype_len
        raw_ndim = _recv_exact(self._sock, 4)
        bytes_read += 4
        ndim = struct.unpack(">I", raw_ndim)[0]
        shape = tuple(struct.unpack(">Q", _recv_exact(self._sock, 8))[0] for _ in range(ndim))
        bytes_read += ndim * 8
        dt = np.dtype(dtype_str)
        nbytes = int(np.prod(shape)) * dt.itemsize if shape else dt.itemsize
        payload = _recv_exact(self._sock, nbytes)
        bytes_read += nbytes
        self.bytes_recv += bytes_read
        self.msgs_recv += 1
        bucket = self._tag_bucket(tag)
        bucket[1] += bytes_read
        bucket[3] += 1
        return np.frombuffer(payload, dtype=dt).reshape(shape).copy()

    @property
    def stats(self) -> dict:

        return {
            "bytes_sent": self.bytes_sent,
            "bytes_recv": self.bytes_recv,
            "bytes_total": self.bytes_sent + self.bytes_recv,
            "msgs_sent": self.msgs_sent,
            "msgs_recv": self.msgs_recv,
            "bytes_by_tag": {
                k: {"sent": v[0], "recv": v[1], "msgs_sent": v[2], "msgs_recv": v[3]}
                for k, v in self.bytes_by_tag.items()
            },
        }

    def close(self) -> None:
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()
