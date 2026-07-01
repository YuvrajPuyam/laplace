"""renderer/avoidance.py - ORCA local collision avoidance (Spike A / v2 Phase A).

Pure-Python/NumPy, dependency-free. This is [no-GPU] integrity logic: the velocity
math is unit-tested on any box (tests/test_avoidance.py), so its correctness does NOT
depend on the (GPU-only) Isaac run. It feeds the per-step velocity command in
renderer/physx_run.drive, so ~12 rigid AMRs path-follow AND locally avoid each other -
killing the "fake de-overlap" problem with real reciprocal avoidance instead.

Reference: van den Berg, Guy, Lin, Manocha, "Reciprocal n-Body Collision Avoidance,"
ISRR 2011 (the RVO2 algorithm, https://gamma.cs.unc.edu/ORCA/). This is a faithful 2D
port of RVO2's Agent::computeNewVelocity + linearProgram1/2/3. There are NO static
obstacle lines here (agents only), so the obstacle-line count is always 0.

Determinism: constraints are processed in a fixed order (no random permutation), so
identical inputs give identical velocities - the spikes are reproducible. At ~12 agents
the O(n^2) LP is trivially fast (well under a millisecond per step).

Coordinates are planar (x, y) in metres; velocities in m/s; dt in seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EPSILON = 1e-5

Vec2 = tuple[float, float]


@dataclass(frozen=True)
class Line:
    """A directed line; the feasible half-plane is to the LEFT of `direction`."""
    point: Vec2
    direction: Vec2  # unit vector


# -- 2D vector helpers (float tuples, mirroring the RVO2 reference) --------------
def _det(a: Vec2, b: Vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def _mul(a: Vec2, s: float) -> Vec2:
    return (a[0] * s, a[1] * s)


def _abs_sq(a: Vec2) -> float:
    return a[0] * a[0] + a[1] * a[1]


def _abs(a: Vec2) -> float:
    return math.sqrt(_abs_sq(a))


def _normalize(a: Vec2) -> Vec2:
    length = _abs(a)
    return (a[0] / length, a[1] / length) if length > EPSILON else (0.0, 0.0)


# -- ORCA half-plane construction (one line per neighbour) -----------------------
def orca_lines(pos: Vec2, vel: Vec2, neighbors, radius: float,
               time_horizon: float, dt: float) -> list[Line]:
    """Build one ORCA constraint line per neighbour.

    neighbors: iterable of (npos, nvel, nradius). Each line's feasible half-plane is
    the set of velocities that avoid that neighbour for `time_horizon` seconds, with
    the avoidance responsibility split 50/50 (reciprocity).
    """
    inv_tau = 1.0 / time_horizon
    inv_dt = 1.0 / dt
    lines: list[Line] = []

    for npos, nvel, nrad in neighbors:
        rel_pos = _sub(npos, pos)
        rel_vel = _sub(vel, nvel)
        dist_sq = _abs_sq(rel_pos)
        comb_r = radius + nrad
        comb_r_sq = comb_r * comb_r

        if dist_sq > comb_r_sq:
            # No collision yet: VO apex is offset by the time-horizon cutoff.
            w = _sub(rel_vel, _mul(rel_pos, inv_tau))
            w_len_sq = _abs_sq(w)
            dot1 = _dot(w, rel_pos)

            if dot1 < 0.0 and dot1 * dot1 > comb_r_sq * w_len_sq:
                # Project on the cutoff circle.
                w_len = math.sqrt(w_len_sq)
                unit_w = (w[0] / w_len, w[1] / w_len)
                direction = (unit_w[1], -unit_w[0])
                u = _mul(unit_w, comb_r * inv_tau - w_len)
            else:
                # Project on one of the cone legs.
                leg = math.sqrt(dist_sq - comb_r_sq)
                if _det(rel_pos, w) > 0.0:
                    direction = ((rel_pos[0] * leg - rel_pos[1] * comb_r) / dist_sq,
                                 (rel_pos[0] * comb_r + rel_pos[1] * leg) / dist_sq)
                else:
                    direction = (-(rel_pos[0] * leg + rel_pos[1] * comb_r) / dist_sq,
                                 -(-rel_pos[0] * comb_r + rel_pos[1] * leg) / dist_sq)
                dot2 = _dot(rel_vel, direction)
                u = _sub(_mul(direction, dot2), rel_vel)
        else:
            # Already overlapping: use the time STEP, not the horizon, to escape.
            w = _sub(rel_vel, _mul(rel_pos, inv_dt))
            w_len = _abs(w)
            unit_w = (w[0] / w_len, w[1] / w_len) if w_len > EPSILON else (0.0, 0.0)
            direction = (unit_w[1], -unit_w[0])
            u = _mul(unit_w, comb_r * inv_dt - w_len)

        # Reciprocity: take half the correction.
        lines.append(Line(_add(vel, _mul(u, 0.5)), direction))

    return lines


# -- 2D linear program (RVO2 linearProgram1/2/3) ---------------------------------
def _linear_program1(lines: list[Line], line_no: int, radius: float,
                     opt_velocity: Vec2, direction_opt: bool):
    """Optimise along line `line_no` subject to the earlier lines + the speed disc.

    Returns (ok, result). ok=False means the constraints are infeasible.
    """
    line = lines[line_no]
    dot_pd = _dot(line.point, line.direction)
    discriminant = dot_pd * dot_pd + radius * radius - _abs_sq(line.point)
    if discriminant < 0.0:
        return False, None  # max-speed disc not reachable on this line

    sqrt_disc = math.sqrt(discriminant)
    t_left = -dot_pd - sqrt_disc
    t_right = -dot_pd + sqrt_disc

    for i in range(line_no):
        denominator = _det(line.direction, lines[i].direction)
        numerator = _det(lines[i].direction, _sub(line.point, lines[i].point))
        if abs(denominator) <= EPSILON:
            # Lines (nearly) parallel.
            if numerator < 0.0:
                return False, None
            continue
        t = numerator / denominator
        if denominator >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return False, None

    if direction_opt:
        t = t_right if _dot(opt_velocity, line.direction) > 0.0 else t_left
    else:
        t = _dot(line.direction, _sub(opt_velocity, line.point))
        t = max(t_left, min(t_right, t))

    return True, _add(line.point, _mul(line.direction, t))


def _linear_program2(lines: list[Line], radius: float, opt_velocity: Vec2,
                     direction_opt: bool):
    """Find the velocity closest to opt_velocity satisfying all lines + the speed disc.

    Returns (fail_index, result). fail_index == len(lines) means full success;
    otherwise it is the index of the first line that could not be satisfied.
    """
    if direction_opt:
        result = _mul(opt_velocity, radius)
    elif _abs_sq(opt_velocity) > radius * radius:
        result = _mul(_normalize(opt_velocity), radius)
    else:
        result = opt_velocity

    for i in range(len(lines)):
        if _det(lines[i].direction, _sub(lines[i].point, result)) > 0.0:
            # result is on the wrong side of line i -> re-optimise on line i.
            temp = result
            ok, result = _linear_program1(lines, i, radius, opt_velocity, direction_opt)
            if not ok:
                return i, temp

    return len(lines), result


def _linear_program3(lines: list[Line], num_obst_lines: int, begin_line: int,
                     radius: float, result: Vec2) -> Vec2:
    """Infeasible fallback: minimise the maximum constraint violation (RVO2 lp3)."""
    distance = 0.0
    for i in range(begin_line, len(lines)):
        if _det(lines[i].direction, _sub(lines[i].point, result)) > distance:
            proj_lines = list(lines[:num_obst_lines])
            for j in range(num_obst_lines, i):
                determinant = _det(lines[i].direction, lines[j].direction)
                if abs(determinant) <= EPSILON:
                    if _dot(lines[i].direction, lines[j].direction) > 0.0:
                        continue  # same direction
                    point = _mul(_add(lines[i].point, lines[j].point), 0.5)
                else:
                    point = _add(lines[i].point, _mul(
                        lines[i].direction,
                        _det(lines[j].direction, _sub(lines[i].point, lines[j].point))
                        / determinant))
                direction = _normalize(_sub(lines[j].direction, lines[i].direction))
                proj_lines.append(Line(point, direction))

            temp = result
            opt_dir = (-lines[i].direction[1], lines[i].direction[0])
            fail_idx, res = _linear_program2(proj_lines, radius, opt_dir, True)
            # If even the projected program failed, keep the previous best.
            result = temp if fail_idx < len(proj_lines) else res
            distance = _det(lines[i].direction, _sub(lines[i].point, result))

    return result


def compute_new_velocity(pos: Vec2, vel: Vec2, pref_vel: Vec2, neighbors,
                         radius: float, max_speed: float,
                         time_horizon: float, dt: float) -> Vec2:
    """ORCA: the velocity closest to `pref_vel` that avoids all neighbours and obeys
    max_speed. `neighbors` is an iterable of (npos, nvel, nradius)."""
    lines = orca_lines(pos, vel, neighbors, radius, time_horizon, dt)
    fail_idx, result = _linear_program2(lines, max_speed, pref_vel, False)
    if fail_idx < len(lines):
        result = _linear_program3(lines, 0, fail_idx, max_speed, result)
    return result


# -- Vectorised batch API (what physx_run.drive calls) ---------------------------
def collision_free_velocities(positions, velocities, pref_velocities, *,
                              radius=0.5, max_speed=1.5, time_horizon=2.0,
                              dt=1.0 / 60.0, neighbor_dist=None):
    """Compute collision-free velocities for a whole fleet in one call.

    positions/velocities/pref_velocities: (N, 2) array-likes (metres, m/s).
    radius: scalar or (N,) per-robot radius. max_speed: scalar or (N,).
    neighbor_dist: if set, ignore neighbours farther than this (centre-to-centre) -
        a cheap cutoff; None considers all other robots.

    Returns an (N, 2) numpy array of new velocities. Pure function, deterministic.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    vel = np.asarray(velocities, dtype=float).reshape(-1, 2)
    pref = np.asarray(pref_velocities, dtype=float).reshape(-1, 2)
    n = len(pos)
    if not (len(vel) == n and len(pref) == n):
        raise ValueError("positions, velocities, pref_velocities must have equal length")

    rad = np.broadcast_to(np.asarray(radius, dtype=float), (n,))
    spd = np.broadcast_to(np.asarray(max_speed, dtype=float), (n,))
    cutoff_sq = None if neighbor_dist is None else float(neighbor_dist) ** 2

    out = np.zeros((n, 2), dtype=float)
    for i in range(n):
        pi = (float(pos[i, 0]), float(pos[i, 1]))
        neighbors = []
        for j in range(n):
            if j == i:
                continue
            if cutoff_sq is not None:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                if dx * dx + dy * dy > cutoff_sq:
                    continue
            neighbors.append(((float(pos[j, 0]), float(pos[j, 1])),
                              (float(vel[j, 0]), float(vel[j, 1])),
                              float(rad[j])))
        v = compute_new_velocity(
            pi, (float(vel[i, 0]), float(vel[i, 1])),
            (float(pref[i, 0]), float(pref[i, 1])),
            neighbors, float(rad[i]), float(spd[i]), time_horizon, dt)
        out[i, 0], out[i, 1] = v
    return out
