"""Deterministic unit tests for engine.live_feed.physx_reader.

physx_reader is a bidirectional NDJSON TCP bridge that now RECONNECTS with
backoff and accepts an optional on_state callback. These tests exercise it
against a REAL localhost loopback socket fed canned NDJSON — never the real
PhysX stream — using short socket timeouts and hard threading.Event guards so
the suite can never hang.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from engine.live_feed import physx_reader

# Hard ceiling for any single reader-thread to live. If a test's stop logic
# regresses, the join times out and the test fails loudly instead of hanging CI.
_JOIN_TIMEOUT = 5.0


class _Stopper:
    """Callable stop flag with a settable event, usable as physx_reader's stopped()."""

    def __init__(self) -> None:
        self._ev = threading.Event()

    def __call__(self) -> bool:
        return self._ev.is_set()

    def stop(self) -> None:
        self._ev.set()


def _listen():
    """Open a localhost listener; return (server_socket, host, port)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(_JOIN_TIMEOUT)
    host, port = srv.getsockname()
    return srv, host, port


def _run_reader(host, port, on_frame, stopped, **kw):
    """Start physx_reader in a daemon thread and return the Thread."""
    t = threading.Thread(
        target=physx_reader,
        args=(host, port, on_frame, stopped),
        kwargs=kw,
        daemon=True,
    )
    t.start()
    return t


def test_normal_frame_round_trips_as_dict():
    srv, host, port = _listen()
    frames: list[dict] = []
    got = threading.Event()

    def on_frame(d):
        frames.append(d)
        got.set()

    stopper = _Stopper()
    t = _run_reader(host, port, on_frame, stopper)
    try:
        conn, _ = srv.accept()
        payload = {"t_sim": 1.5, "agents": {"amr_00": [1.0, 2.0, 0.0]}}
        conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        assert got.wait(_JOIN_TIMEOUT), "frame never delivered"
        assert frames[0] == payload
        assert isinstance(frames[0], dict)
    finally:
        stopper.stop()
        conn.close()
        srv.close()
        t.join(_JOIN_TIMEOUT)
        assert not t.is_alive()


def test_frame_split_across_two_recv_chunks_parsed_once():
    srv, host, port = _listen()
    frames: list[dict] = []
    got = threading.Event()

    def on_frame(d):
        frames.append(d)
        got.set()

    stopper = _Stopper()
    t = _run_reader(host, port, on_frame, stopper)
    try:
        conn, _ = srv.accept()
        payload = {"t_sim": 9.0, "agents": {"amr_01": [3.0, 4.0, 1.57]}}
        line = (json.dumps(payload) + "\n").encode("utf-8")
        half = len(line) // 2
        conn.sendall(line[:half])
        time.sleep(0.2)            # force a separate recv() for the tail
        conn.sendall(line[half:])
        assert got.wait(_JOIN_TIMEOUT), "split frame never parsed"
        assert frames == [payload]  # parsed exactly once, not twice
    finally:
        stopper.stop()
        conn.close()
        srv.close()
        t.join(_JOIN_TIMEOUT)
        assert not t.is_alive()


def test_malformed_json_line_skipped_without_killing_reader():
    srv, host, port = _listen()
    frames: list[dict] = []
    good = threading.Event()

    def on_frame(d):
        frames.append(d)
        good.set()

    stopper = _Stopper()
    t = _run_reader(host, port, on_frame, stopper)
    try:
        conn, _ = srv.accept()
        conn.sendall(b"{not valid json}\n")     # malformed -> must be skipped
        good_payload = {"t_sim": 2.0, "agents": {}}
        conn.sendall((json.dumps(good_payload) + "\n").encode("utf-8"))
        assert good.wait(_JOIN_TIMEOUT), "reader died on malformed line"
        assert frames == [good_payload]
    finally:
        stopper.stop()
        conn.close()
        srv.close()
        t.join(_JOIN_TIMEOUT)
        assert not t.is_alive()


def test_control_queue_message_sent_upstream_as_one_ndjson_line():
    import queue

    srv, host, port = _listen()
    control_q: queue.Queue = queue.Queue()
    control_q.put({"cmd": "pause", "value": 1})

    stopper = _Stopper()
    t = _run_reader(host, port, lambda d: None, stopper, control_q=control_q)
    try:
        conn, _ = srv.accept()
        conn.settimeout(_JOIN_TIMEOUT)
        # Read the upstream control line off the peer socket.
        buf = b""
        deadline = time.monotonic() + _JOIN_TIMEOUT
        while b"\n" not in buf and time.monotonic() < deadline:
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk
        assert b"\n" in buf, "no control line received upstream"
        line, _ = buf.split(b"\n", 1)
        assert json.loads(line) == {"cmd": "pause", "value": 1}
    finally:
        stopper.stop()
        conn.close()
        srv.close()
        t.join(_JOIN_TIMEOUT)
        assert not t.is_alive()


def test_reader_stops_promptly_when_stopped_returns_true():
    srv, host, port = _listen()
    stopper = _Stopper()
    t = _run_reader(host, port, lambda d: None, stopper)
    try:
        conn, _ = srv.accept()
        # Connected and idle. Now request stop; the inner loop polls stopped()
        # each ~1s recv timeout, so it must exit well within the join window.
        stopper.stop()
        t.join(_JOIN_TIMEOUT)
        assert not t.is_alive(), "reader did not stop promptly"
    finally:
        stopper.stop()
        try:
            conn.close()
        except Exception:
            pass
        srv.close()
        t.join(_JOIN_TIMEOUT)


def test_on_state_callback_reports_connected():
    srv, host, port = _listen()
    states: list[str] = []
    connected = threading.Event()

    def on_state(s):
        states.append(s)
        if s == "connected":
            connected.set()

    stopper = _Stopper()
    t = threading.Thread(
        target=physx_reader,
        args=(host, port, lambda d: None, stopper),
        kwargs={"on_state": on_state},
        daemon=True,
    )
    t.start()
    try:
        conn, _ = srv.accept()
        assert connected.wait(_JOIN_TIMEOUT), "never reported connected"
        assert "connecting" in states
        assert "connected" in states
    finally:
        stopper.stop()
        conn.close()
        srv.close()
        t.join(_JOIN_TIMEOUT)
        assert not t.is_alive()
        # On clean shutdown the terminal state is reported.
        assert states[-1] == "disconnected"
