import math

import pytest

from sim.config import apply_patch, config_hash, fill_defaults, validate_config
from sim.navgraph import AISLE_SPACING_M, NavGraph, NavGraphError


def test_grid_generation(baseline_config):
    g = NavGraph(fill_defaults(baseline_config))
    # 6 aisles x 31 positions
    assert g.n_nodes == 6 * 31
    # aisle edges: 6*30; cross-aisles at 0,15,30: 3 * 5
    assert len(g.pools) == 6 * 30 + 3 * 5
    assert g.node_xy[g.node_index("A1_00")] == (0.0, 0.0)
    assert g.node_xy[g.node_index("A3_15")] == (2 * AISLE_SPACING_M, 15.0)


def test_shortest_path_uses_cross_aisle(baseline_config):
    g = NavGraph(fill_defaults(baseline_config))
    # A1_05 -> A2_05: along aisle to cross at 0 (5m) + cross (3m) + back (5m)
    d = g.shortest_dist(g.node_index("A1_05"), g.node_index("A2_05"))
    assert d == pytest.approx(5 + AISLE_SPACING_M + 5)


def test_extra_edge_shortens_path(baseline_config, braess_patch):
    cfg = fill_defaults(apply_patch(baseline_config, braess_patch["patch"]))
    g = NavGraph(cfg)
    d = g.shortest_dist(g.node_index("A3_15"), g.node_index("A4_15"))
    assert d == pytest.approx(AISLE_SPACING_M)
    # and the override took effect on the pool
    pi = g.pool_index(g.node_index("A3_15"), g.node_index("A4_15"))
    assert g.pools[pi]["capacity"] == 1


def test_aisle_spacing_m_overrides_default(baseline_config):
    # optional real-pitch field scales node coords and cross-aisle edge LENGTH,
    # which is what drives travel time / battery / routing / congestion.
    cfg = fill_defaults(apply_patch(baseline_config,
                                    {"layout.grid.aisle_spacing_m": 9.5}))
    g = NavGraph(cfg)
    assert g.spacing == 9.5
    assert g.node_xy[g.node_index("A3_15")] == (2 * 9.5, 15.0)
    # cross-aisle edge at an actual cross-aisle position (baseline: 0, 15, 30)
    pi = g.pool_index(g.node_index("A1_00"), g.node_index("A2_00"))
    assert g.pools[pi]["length"] == pytest.approx(9.5)
    d = g.shortest_dist(g.node_index("A1_05"), g.node_index("A2_05"))
    assert d == pytest.approx(5 + 9.5 + 5)


def test_omitted_aisle_spacing_uses_3m_default(baseline_config):
    g = NavGraph(fill_defaults(baseline_config))
    assert g.spacing == AISLE_SPACING_M == 3.0


def test_aisle_spacing_not_default_filled_preserves_hash(baseline_config):
    # the field is lazily read, NOT injected by fill_defaults — so a config that
    # omits it stays hash-identical to the pre-field canonical form, while a
    # real-pitch config is a genuinely different (faithful) scenario.
    assert "aisle_spacing_m" not in fill_defaults(baseline_config)["layout"]["grid"]
    real = apply_patch(baseline_config, {"layout.grid.aisle_spacing_m": 9.5})
    validate_config(real)
    assert config_hash(real) != config_hash(baseline_config)


def test_one_way_restricts_direction(baseline_config):
    cfg = fill_defaults(apply_patch(baseline_config, {
        "layout.edge_overrides": [{"edge": "A1_00->A1_01", "one_way": True}],
    }))
    g = NavGraph(cfg)
    u, v = g.node_index("A1_00"), g.node_index("A1_01")
    assert any(n == v for n, _ in g.adj[u])
    assert not any(n == u for n, _ in g.adj[v])
    # network stays connected the long way round
    assert g.shortest_dist(v, u) < math.inf


def test_unknown_node_raises(baseline_config):
    cfg = fill_defaults(apply_patch(baseline_config, {
        "layout.extra_edges": [{"from": "A9_99", "to": "A1_00"}],
    }))
    with pytest.raises(NavGraphError):
        NavGraph(cfg)


def test_congestion_aware_avoids_occupied(baseline_config):
    # two equal-length routes A1_00 -> A1_02: the 2-hop aisle route and an
    # extra direct edge; saturating the one static routing picks must flip
    # congestion-aware routing to the other
    cfg = fill_defaults(apply_patch(baseline_config, {
        "layout.extra_edges": [{"from": "A1_00", "to": "A1_02"}],
    }))
    g = NavGraph(cfg)
    src, dst = g.node_index("A1_00"), g.node_index("A1_02")
    static = g.shortest_path(src, dst)
    occ = [0] * len(g.pools)
    for _, _, pi in static:
        occ[pi] = g.pools[pi]["capacity"]  # fully occupied
    detour = g.congestion_aware_path(src, dst, occ)
    assert detour != static
    assert sum(g.pools[pi]["length"] for _, _, pi in detour) == pytest.approx(2.0)
