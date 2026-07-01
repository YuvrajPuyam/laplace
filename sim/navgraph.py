"""Navgraph generation from Contract A layout, geometry, and shortest paths.

Grid generation rules (Contract A description, made concrete here):
- Nodes 'A{a}_{p}' for aisle a in 1..aisles, position p in 0..aisle_length_m,
  at 1 m intervals along each aisle.
- Aisle edges connect consecutive positions within an aisle (length 1 m).
- Cross-aisles at each position listed in layout.grid.cross_aisles connect
  equal positions across ADJACENT aisles (A1_p—A2_p, A2_p—A3_p, ...).
- AISLE_SPACING_M: geometric distance between adjacent aisles. Used as the
  DEFAULT when layout.grid.aisle_spacing_m is omitted (3.0 m, the historic
  sim-wide convention). A config MAY override it via the optional
  layout.grid.aisle_spacing_m field (e.g. a scanned real warehouse's true pitch);
  it sets the length of cross-aisle and extra edges and the node coordinates.
  Read lazily (grid.get) and intentionally NOT default-filled, so configs that
  omit it stay hash-identical to the pre-field canonical form.
  (Flagged as a modeling decision — see sim/README.md.)
- Node geometry: node A{a}_{p} sits at (x, y) = ((a-1) * AISLE_SPACING_M, p).
- extra_edges get Euclidean length between their endpoints' coordinates.

Congestion model: each physical edge is a capacity POOL shared by both travel
directions (default capacity 2). An AMR entering a full pool queues FIFO at
its tail node and holds no capacity while queued — so the network cannot
deadlock (edges always drain). edge_overrides referencing either direction's
id apply to the pool; one_way restricts traversal to the stated direction.
"""

from __future__ import annotations

import heapq
import math

AISLE_SPACING_M = 3.0
DEFAULT_EDGE_CAPACITY = 2


class NavGraphError(ValueError):
    pass


def node_name(aisle: int, pos: int) -> str:
    return f"A{aisle}_{pos:02d}"


def parse_node(name: str) -> tuple[int, int]:
    a, p = name[1:].split("_")
    return int(a), int(p)


class NavGraph:
    """Immutable topology for one config. Built once per rollout."""

    def __init__(self, config: dict):
        grid = config["layout"]["grid"]
        n_aisles = grid["aisles"]
        length = grid["aisle_length_m"]
        cross = sorted(grid["cross_aisles"])
        # Optional real-pitch override; defaults to the 3.0 m sim convention.
        # Lazy read (not default-filled) so omitting it preserves config_hash.
        self.spacing = float(grid.get("aisle_spacing_m", AISLE_SPACING_M))
        for c in cross:
            if c > length:
                raise NavGraphError(f"cross_aisle position {c} exceeds aisle_length_m {length}")

        # --- nodes -----------------------------------------------------
        self.node_names: list[str] = []
        self.node_xy: list[tuple[float, float]] = []
        self._index: dict[str, int] = {}
        for a in range(1, n_aisles + 1):
            for p in range(length + 1):
                self._index[node_name(a, p)] = len(self.node_names)
                self.node_names.append(node_name(a, p))
                self.node_xy.append(((a - 1) * self.spacing, float(p)))
        self.n_nodes = len(self.node_names)

        # --- edge pools ------------------------------------------------
        # pool: dict(a, b, length, capacity, max_speed, one_way)
        # a, b are node indices; one_way True means traversal a->b only.
        fleet_speed = config["fleet"].get("speed_mps", 1.5)
        self.pools: list[dict] = []
        self._pool_by_pair: dict[tuple[int, int], int] = {}

        def add_pool(u: int, v: int, edge_len: float) -> None:
            key = (min(u, v), max(u, v))
            if key in self._pool_by_pair:
                raise NavGraphError(
                    f"duplicate edge {self.node_names[u]}->{self.node_names[v]}"
                )
            self._pool_by_pair[key] = len(self.pools)
            self.pools.append({
                "a": u, "b": v, "length": edge_len,
                "capacity": DEFAULT_EDGE_CAPACITY,
                "max_speed": fleet_speed, "one_way": False,
            })

        for a in range(1, n_aisles + 1):
            for p in range(length):
                add_pool(self._index[node_name(a, p)], self._index[node_name(a, p + 1)], 1.0)
        for c in cross:
            for a in range(1, n_aisles):
                add_pool(self._index[node_name(a, c)], self._index[node_name(a + 1, c)],
                         self.spacing)

        for ee in config["layout"].get("extra_edges", []):
            u = self.node_index(ee["from"])
            v = self.node_index(ee["to"])
            if (min(u, v), max(u, v)) in self._pool_by_pair:
                # extra edge coincides with an existing grid edge: no-op
                # (edge_overrides still apply to the existing pool)
                continue
            ux, uy = self.node_xy[u]
            vx, vy = self.node_xy[v]
            add_pool(u, v, math.hypot(vx - ux, vy - uy))
            if not ee.get("bidirectional", True):
                self.pools[-1]["one_way"] = True
                self.pools[-1]["a"], self.pools[-1]["b"] = u, v

        for ov in config["layout"].get("edge_overrides", []):
            frm, to = ov["edge"].split("->")
            u, v = self.node_index(frm), self.node_index(to)
            key = (min(u, v), max(u, v))
            if key not in self._pool_by_pair:
                raise NavGraphError(f"edge_override references unknown edge {ov['edge']}")
            pool = self.pools[self._pool_by_pair[key]]
            if "capacity" in ov:
                pool["capacity"] = ov["capacity"]
            if "max_speed_mps" in ov:
                pool["max_speed"] = ov["max_speed_mps"]
            if ov.get("one_way", False):
                pool["one_way"] = True
                pool["a"], pool["b"] = u, v  # one-way direction = stated from->to

        # --- adjacency: node -> list of (neighbor, pool_idx) ------------
        self.adj: list[list[tuple[int, int]]] = [[] for _ in range(self.n_nodes)]
        for pi, pool in enumerate(self.pools):
            self.adj[pool["a"]].append((pool["b"], pi))
            if not pool["one_way"]:
                self.adj[pool["b"]].append((pool["a"], pi))

        self._sp_cache: dict[int, tuple[list[float], list[int], list[int]]] = {}

    # ------------------------------------------------------------------
    def node_index(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError:
            raise NavGraphError(f"unknown node {name!r} (check aisle count / aisle length)") from None

    def edge_id(self, u: int, v: int) -> str:
        return f"{self.node_names[u]}->{self.node_names[v]}"

    def pool_index(self, u: int, v: int) -> int:
        return self._pool_by_pair[(min(u, v), max(u, v))]

    # ------------------------------------------------------------------
    def _dijkstra(self, src: int, weights: list[float] | None = None):
        """Returns (dist, prev_node, prev_pool) arrays from src.

        weights: optional per-pool cost override (defaults to pool length).
        """
        INF = math.inf
        dist = [INF] * self.n_nodes
        prev_node = [-1] * self.n_nodes
        prev_pool = [-1] * self.n_nodes
        dist[src] = 0.0
        heap = [(0.0, src)]
        adj = self.adj
        pools = self.pools
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, pi in adj[u]:
                w = weights[pi] if weights is not None else pools[pi]["length"]
                nd = d + w
                # strict improvement => deterministic ties (first-found wins,
                # and exploration order is fixed by adjacency construction)
                if nd < dist[v]:
                    dist[v] = nd
                    prev_node[v] = u
                    prev_pool[v] = pi
                    heapq.heappush(heap, (nd, v))
        return dist, prev_node, prev_pool

    def shortest_dist(self, src: int, dst: int) -> float:
        if src not in self._sp_cache:
            self._sp_cache[src] = self._dijkstra(src)
        return self._sp_cache[src][0][dst]

    def shortest_path(self, src: int, dst: int) -> list[tuple[int, int, int]]:
        """Static shortest path as a list of (from_node, to_node, pool_idx)."""
        if src not in self._sp_cache:
            self._sp_cache[src] = self._dijkstra(src)
        return self._reconstruct(src, dst, *self._sp_cache[src])

    def congestion_aware_path(self, src: int, dst: int,
                              occupancy: list[int],
                              penalty: float = 2.0) -> list[tuple[int, int, int]]:
        """Path with edge cost = length * (1 + penalty * occ/capacity) (Contract A).

        `penalty` defaults to 2.0 (the frozen-v1 constant) so existing callers and
        configs are byte-identical; the engine threads fleet.congestion_penalty
        through when set. Occupancy snapshot is taken at leg dispatch time; not
        recomputed mid-leg (v1 semantics).
        """
        weights = [
            p["length"] * (1.0 + penalty * occupancy[i] / p["capacity"])
            for i, p in enumerate(self.pools)
        ]
        res = self._dijkstra(src, weights)
        return self._reconstruct(src, dst, *res)

    def _reconstruct(self, src, dst, dist, prev_node, prev_pool):
        if dist[dst] == math.inf:
            raise NavGraphError(f"no path {self.node_names[src]} -> {self.node_names[dst]}")
        path = []
        v = dst
        while v != src:
            u = prev_node[v]
            path.append((u, v, prev_pool[v]))
            v = u
        path.reverse()
        return path
