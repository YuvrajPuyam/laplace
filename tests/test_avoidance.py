"""Tests for renderer/avoidance.py - the ORCA local-avoidance layer (Spike A).

Pure CPU; no Isaac, no GPU. These tests are the trustworthiness anchor for the
avoidance math: if they pass, the on-cluster physics run is driving a verified
controller, not an unchecked one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from renderer.avoidance import collision_free_velocities, compute_new_velocity


def _min_separation(pos: np.ndarray) -> float:
    """Smallest centre-to-centre distance among all robot pairs."""
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def _simulate(start, goals, *, radius=0.5, max_speed=1.2, dt=0.05, steps=600,
              time_horizon=3.0, goal_tol=0.2, neighbor_dist=None):
    """Drive every agent toward its goal under ORCA. Returns (final_pos, min_sep_seen,
    initial_mean_goal_dist, final_mean_goal_dist)."""
    pos = np.asarray(start, dtype=float).reshape(-1, 2)
    goals = np.asarray(goals, dtype=float).reshape(-1, 2)
    vel = np.zeros_like(pos)
    init_mean = float(np.linalg.norm(goals - pos, axis=1).mean())
    min_sep = np.inf

    for _ in range(steps):
        to_goal = goals - pos
        d = np.linalg.norm(to_goal, axis=1, keepdims=True)
        pref = np.where(d > goal_tol, to_goal / np.clip(d, 1e-9, None) * max_speed, 0.0)
        vel = collision_free_velocities(
            pos, vel, pref, radius=radius, max_speed=max_speed,
            time_horizon=time_horizon, dt=dt, neighbor_dist=neighbor_dist)
        pos = pos + vel * dt
        min_sep = min(min_sep, _min_separation(pos))
        if np.all(np.linalg.norm(goals - pos, axis=1) < goal_tol):
            break

    final_mean = float(np.linalg.norm(goals - pos, axis=1).mean())
    return pos, float(min_sep), init_mean, final_mean


# -- unit behaviour --------------------------------------------------------------
def test_no_neighbors_returns_preferred():
    v = compute_new_velocity((0.0, 0.0), (0.0, 0.0), (1.0, 0.0), [],
                             radius=0.5, max_speed=1.5, time_horizon=2.0, dt=1 / 60)
    assert v == pytest.approx((1.0, 0.0))


def test_speed_cap_enforced():
    # Preferred velocity exceeds max_speed -> clamped to the speed disc.
    out = collision_free_velocities([[0, 0]], [[0, 0]], [[10.0, 0.0]],
                                    radius=0.5, max_speed=1.5)
    assert np.linalg.norm(out[0]) == pytest.approx(1.5, abs=1e-6)


def test_deterministic():
    pos = [[0, 0], [3, 0.1], [1.5, 2.0]]
    vel = [[1, 0], [-1, 0], [0, -1]]
    pref = [[1.2, 0], [-1.2, 0], [0, -1.2]]
    a = collision_free_velocities(pos, vel, pref, radius=0.5, max_speed=1.2)
    b = collision_free_velocities(pos, vel, pref, radius=0.5, max_speed=1.2)
    assert np.array_equal(a, b)


def test_diverging_neighbor_does_not_perturb():
    # A neighbour moving AWAY should not bend our velocity (its VO is behind us).
    out = collision_free_velocities(
        [[0, 0], [-2, 0]], [[1, 0], [-1, 0]], [[1.2, 0], [-1.2, 0]],
        radius=0.5, max_speed=1.2)
    assert out[0] == pytest.approx((1.2, 0.0), abs=1e-6)


# -- emergent avoidance ----------------------------------------------------------
def test_head_on_avoid_and_pass():
    # Near-head-on (tiny offset so it's not a degenerate perfectly-collinear case).
    start = [[-5.0, 0.0], [5.0, 0.05]]
    goals = [[5.0, 0.0], [-5.0, 0.05]]
    final, min_sep, _, final_mean = _simulate(start, goals, radius=0.5)
    assert min_sep > 0.9, f"robots collided (min sep {min_sep:.3f} m, need > 0.9)"
    assert final_mean < 0.3, f"robots did not reach goals (mean dist {final_mean:.3f})"


def test_pass_stationary_robot():
    # One robot must route around a parked one sitting on its straight path.
    start = [[-5.0, 0.0], [0.0, 0.0]]
    goals = [[5.0, 0.0], [0.0, 0.0]]
    final, min_sep, _, _ = _simulate(start, goals, radius=0.5)
    assert min_sep > 0.9, f"clipped the parked robot (min sep {min_sep:.3f} m)"
    # The mover reached the far side.
    assert final[0, 0] > 4.5, f"mover did not get past (x={final[0,0]:.2f})"


def test_circle_dense_no_collision():
    # Canonical ORCA stressor: agents on a circle heading to their antipodes -> a dense
    # central scramble. The property the local layer GUARANTEES is collision-freedom;
    # convergence through a perfectly symmetric pinch is a known ORCA limitation
    # (central deadlock), resolved in Laplace by the higher-level DES path plan, not by
    # avoidance alone. So here we assert no-collision (+ net inward progress) only.
    n = 8
    R = 4.0
    base = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ang = base + 0.03 * np.cos(3 * base)  # deterministic symmetry-break
    start = np.column_stack([R * np.cos(ang), R * np.sin(ang)])
    goals = -start  # antipodes
    _, min_sep, init_mean, final_mean = _simulate(
        start, goals, radius=0.5, max_speed=1.2, dt=0.05, steps=1000, time_horizon=3.0)
    assert min_sep > 0.85, f"circle scramble collided (min sep {min_sep:.3f} m)"
    assert final_mean < init_mean, "agents made no net progress toward goals"


def test_opposing_streams_converge():
    # The representative warehouse pattern: two opposing aisle streams pass through each
    # other (staggered so it's not perfectly head-on). ORCA resolves this via lane
    # formation -> both no-collision AND strong convergence.
    left_y = [-1.2, 0.0, 1.2]
    right_y = [-0.6, 0.6, 1.8]  # staggered half a lane
    start = [[-6.0, y] for y in left_y] + [[6.0, y] for y in right_y]
    goals = [[6.0, y] for y in left_y] + [[-6.0, y] for y in right_y]
    _, min_sep, init_mean, final_mean = _simulate(
        start, goals, radius=0.5, max_speed=1.2, dt=0.05, steps=800, time_horizon=2.0)
    assert min_sep > 0.85, f"streams collided (min sep {min_sep:.3f} m)"
    assert final_mean < 0.2 * init_mean, (
        f"streams did not pass through (mean dist {final_mean:.2f} vs {init_mean:.2f})")


def test_reciprocity_is_symmetric():
    # Two mirror-image robots should get mirror-image avoidance velocities.
    out = collision_free_velocities(
        [[-3.0, 0.0], [3.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]],
        [[1.2, 0.0], [-1.2, 0.0]], radius=0.5, max_speed=1.2, time_horizon=3.0)
    # vx mirrored (opposite), vy mirrored (opposite) under the 180-deg symmetry.
    assert out[0, 0] == pytest.approx(-out[1, 0], abs=1e-6)
    assert out[0, 1] == pytest.approx(-out[1, 1], abs=1e-6)
