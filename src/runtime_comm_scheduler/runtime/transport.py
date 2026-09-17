"""Small TCP NDJSON transport for the coordinator and rank runtimes."""

from __future__ import annotations

import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

from .coordinator import CoordinatorError, CoordinatorState, Outbound
from .protocol import decode_message, encode_message, hello_message, event_message


@dataclass
class _Inbound:
    endpoint: int
    message: dict[str, Any] | None
    error: BaseException | None = None


class CoordinatorServer:
    _MAX_CHECK_INTERVAL_S = 0.05

    def __init__(self, coordinator: CoordinatorState, host: str, port: int) -> None:
        self.coordinator = coordinator
        self.host = host
        self.port = port
        self._listener: socket.socket | None = None
        self._inbound: queue.Queue[_Inbound] = queue.Queue()
        self._writers: dict[int, queue.Queue[dict[str, Any] | None]] = {}
        self._connections: dict[int, socket.socket] = {}
        self._finish_sent: dict[int, threading.Event] = {}
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(len(self.coordinator.endpoints))
        listener.settimeout(0.2)
        self._listener = listener
        self.port = int(listener.getsockname()[1])
        self._threads = [
            threading.Thread(target=self._accept_loop, name="runtime-accept", daemon=True),
            threading.Thread(target=self._event_loop, name="runtime-coordinator", daemon=True),
        ]
        for thread in tuple(self._threads):
            thread.start()

    def wait_ready(self, timeout: float = 20.0) -> None:
        if not self._ready.wait(timeout):
            raise TimeoutError("runtime coordinator did not receive all endpoints")

    def close(self, timeout: float = 2.0) -> None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        if self.coordinator.finished:
            deadline = time.monotonic() + timeout
            with self._lock:
                finish_events = tuple(self._finish_sent.values())
            for event in finish_events:
                remaining = max(0.0, deadline - time.monotonic())
                if remaining == 0:
                    break
                event.wait(remaining)
        self._stop.set()
        with self._lock:
            for writer in self._writers.values():
                writer.put(None)
            for conn in self._connections.values():
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                conn.close()
        for thread in self._threads:
            thread.join(timeout=2)

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            thread = threading.Thread(target=self._connection_loop, args=(conn,), daemon=True)
            thread.start()
            self._threads.append(thread)

    def _connection_loop(self, conn: socket.socket) -> None:
        endpoint: int | None = None
        try:
            reader = conn.makefile("rb")
            first = decode_message(reader.readline())
            if first.get("kind") != "HELLO":
                raise CoordinatorError("first control message must be HELLO")
            endpoint = int(first["endpoint"])
            if endpoint not in self.coordinator.endpoint:
                raise CoordinatorError(f"unknown endpoint {endpoint}")
            with self._lock:
                if endpoint in self._connections:
                    raise CoordinatorError(f"duplicate connection for endpoint {endpoint}")
                self._connections[endpoint] = conn
                self._writers[endpoint] = queue.Queue()
                self._finish_sent[endpoint] = threading.Event()
                writer = threading.Thread(target=self._writer_loop, args=(endpoint, conn, self._writers[endpoint]), daemon=True)
                writer.start()
                self._threads.append(writer)
            self._inbound.put(_Inbound(endpoint, first))
            for line in reader:
                self._inbound.put(_Inbound(endpoint, decode_message(line)))
        except BaseException as exc:  # noqa: BLE001
            self._inbound.put(_Inbound(endpoint if endpoint is not None else -1, None, exc))
        finally:
            if endpoint is not None and not self._stop.is_set() and not self.coordinator.done:
                self._inbound.put(_Inbound(endpoint, None, ConnectionError("control connection closed")))
            try:
                conn.close()
            except OSError:
                pass

    def _writer_loop(self, endpoint: int, conn: socket.socket, messages: queue.Queue[dict[str, Any] | None]) -> None:
        while True:
            message = messages.get()
            if message is None:
                return
            try:
                conn.sendall(encode_message(message))
                if message.get("kind") == "FINISHED":
                    with self._lock:
                        finish_sent = self._finish_sent.get(endpoint)
                    if finish_sent is not None:
                        finish_sent.set()
            except OSError:
                self._inbound.put(_Inbound(endpoint, None, ConnectionError("control send failed")))
                return

    def _event_loop(self) -> None:
        while not self._stop.is_set():
            if self.coordinator.done:
                self._stop.wait()
                continue
            try:
                deadline = self.coordinator.next_deadline
                timeout = self._MAX_CHECK_INTERVAL_S
                if deadline is not None:
                    timeout = min(timeout, max(0.0, deadline - time.monotonic()))
                inbound = self._inbound.get(timeout=timeout)
            except queue.Empty:
                self._dispatch(self.coordinator.tick(time.monotonic()))
                continue
            if inbound.error is not None:
                try:
                    self._dispatch(self.coordinator.fail("transport", endpoint=inbound.endpoint, error=str(inbound.error)))
                except CoordinatorError:
                    pass
                continue
            assert inbound.message is not None
            message = inbound.message
            if message.get("kind") == "HELLO":
                if len(self._connections) == len(self.coordinator.endpoints):
                    self._ready.set()
                    self._dispatch([Outbound(rank, "READY", {"epoch": self.coordinator.epoch}) for rank in self.coordinator.endpoints])
                continue
            try:
                if message.get("epoch") != self.coordinator.epoch:
                    raise CoordinatorError("old or future epoch")
                out = self.coordinator.apply(inbound.endpoint, message["kind"], int(message["event_seq"]), message["payload"], time.monotonic())
                self._dispatch(out)
            except BaseException as exc:  # noqa: BLE001
                try:
                    self._dispatch(self.coordinator.fail("protocol", endpoint=inbound.endpoint, error=str(exc)))
                except CoordinatorError:
                    pass

    def _dispatch(self, messages: list[Outbound]) -> None:
        with self._lock:
            for item in messages:
                writer = self._writers.get(item.endpoint)
                if writer is None:
                    continue
                writer.put({"protocol": 1, "kind": item.kind, "epoch": self.coordinator.epoch, "endpoint": item.endpoint, "payload": item.payload})


class ControlClient:
    def __init__(self, endpoint: int, epoch: int, host: str, port: int, timeout: float = 20.0) -> None:
        self.endpoint = endpoint
        self.epoch = epoch
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: socket.socket | None = None
        self._writer: Any = None
        self._incoming: queue.Queue[dict[str, Any] | BaseException] = queue.Queue()
        self._event_seq = 1
        self._delivery_seq = 1
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None

    def connect(self) -> None:
        deadline = time.monotonic() + self.timeout
        last_error: BaseException | None = None
        while True:
            try:
                sock = socket.create_connection(
                    (self.host, self.port),
                    timeout=min(0.2, max(0.01, deadline - time.monotonic())),
                )
                break
            except OSError as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise ConnectionError(f"could not connect to coordinator: {last_error}") from exc
                time.sleep(0.01)
        sock.settimeout(None)
        self._socket = sock
        self._writer = sock.makefile("wb")
        self._writer.write(encode_message(hello_message(self.epoch, self.endpoint)))
        self._writer.flush()
        self._reader_thread = threading.Thread(target=self._reader_loop, name=f"runtime-reader-{self.endpoint}", daemon=True)
        self._reader_thread.start()

    def send(self, kind: str, payload: dict[str, Any]) -> None:
        with self._lock:
            if self._writer is None:
                raise RuntimeError("control client is not connected")
            message = event_message(kind, self.epoch, self.endpoint, self._event_seq, payload)
            self._event_seq += 1
            self._writer.write(encode_message(message))
            self._writer.flush()

    def receive(self, timeout: float | None = None) -> dict[str, Any]:
        item = self._incoming.get(timeout=timeout)
        if isinstance(item, BaseException):
            raise item
        if item.get("epoch") != self.epoch or item.get("endpoint") != self.endpoint:
            raise RuntimeError("control message identity mismatch")
        if item.get("kind") in {"GRANT", "FINISHED", "FAILED"}:
            delivery = item.get("payload", {}).get("delivery_seq")
            if delivery is not None:
                if int(delivery) != self._delivery_seq:
                    raise RuntimeError(f"delivery sequence error: expected {self._delivery_seq}, got {delivery}")
                self._delivery_seq += 1
        return item

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except OSError:
                pass
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass

    def _reader_loop(self) -> None:
        assert self._socket is not None
        try:
            reader = self._socket.makefile("rb")
            for line in reader:
                self._incoming.put(decode_message(line))
            self._incoming.put(ConnectionError("control connection closed"))
        except BaseException as exc:  # noqa: BLE001
            self._incoming.put(exc)
