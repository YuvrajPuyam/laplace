"""engine/live_feed.py — the engine's bridge to the PhysX pose stream + a wall clock.

`physx_reader` is a bidirectional NDJSON TCP client (run in a thread) that connects to the
PhysX stream (renderer/physx_stream — local mock, or Isaac on Gilbreth over an SSH tunnel),
calls on_frame(dict) for each pose frame, forwards operator control upstream, and RECONNECTS
with backoff if the stream drops. `SystemClock` is the monotonic clock for the playout relay.

Pairs with engine/playout.py (jitter buffer + relay) and the /twin/live WebSocket. The live
motion model itself is renderer.physx_stream.Fleet (mock or PhysX) — this module only
transports its frames.
"""

from __future__ import annotations

import json as _json
import socket as _socket
import time


def physx_reader(host: str, port: int, on_frame, stopped, control_q=None,
                 on_state=None) -> None:
    """Bidirectional NDJSON TCP bridge to the PhysX stream (renderer/physx_stream), run in a
    thread: connect, call on_frame(dict) for each inbound pose frame, and (if control_q is
    given) drain it and send each control dict back to the stream as one NDJSON line.

    Runs until ``stopped()``. RECONNECTS with capped backoff if the stream is not up yet or
    the connection drops mid-feed (e.g. an SSH-tunnel blip) — a dropped tunnel must NOT
    silently kill the live feed, which was the prior failure mode. ``on_state(s)``, if given,
    is called with 'connecting' | 'connected' | 'reconnecting' | 'disconnected' so the engine
    can surface real per-hop liveness to the viewer instead of a frozen scene."""
    def _state(s: str) -> None:
        if on_state is not None:
            try:
                on_state(s)
            except Exception:                                   # a status callback must never kill the feed
                pass

    backoff = 0.5
    while not stopped():
        _state("connecting")
        try:
            with _socket.create_connection((host, port), timeout=10) as s:
                s.settimeout(1.0)
                _state("connected")
                backoff = 0.5                                   # healthy connect resets the backoff
                buf = b""
                while not stopped():
                    if control_q is not None:                   # forward operator control upstream
                        while True:
                            try:
                                msg = control_q.get_nowait()
                            except Exception:
                                break
                            s.sendall((_json.dumps(msg) + "\n").encode("utf-8"))  # OSError -> reconnect
                    try:
                        chunk = s.recv(8192)
                    except _socket.timeout:
                        continue
                    if not chunk:
                        break                                   # peer closed -> reconnect
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            on_frame(_json.loads(line))
                        except ValueError:
                            continue
        except OSError:
            pass                                                # connect/read/send failed -> back off + retry
        if stopped():
            break
        _state("reconnecting")
        slept = 0.0                                             # poll stopped() THROUGHOUT the backoff
        while slept < backoff and not stopped():
            time.sleep(0.1)
            slept += 0.1
        backoff = min(backoff * 2, 5.0)
    _state("disconnected")


class SystemClock:
    """Real wall clock for engine/playout.PlayoutController (monotonic)."""

    def now(self) -> float:
        return time.monotonic()
