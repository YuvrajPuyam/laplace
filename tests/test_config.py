import pytest

from sim.config import (
    ConfigError, apply_patch, canonical_json, config_hash, fill_defaults,
    validate_config,
)


def test_baseline_validates(baseline_config):
    validate_config(baseline_config)


def test_defaults_dont_change_hash(baseline_config):
    # baseline already states every default explicitly; stripping the
    # defaulted fields must produce the identical hash (tools.md: semantically
    # identical configs hash identically)
    import copy
    stripped = copy.deepcopy(baseline_config)
    del stripped["fleet"]["speed_mps"]
    del stripped["fleet"]["routing"]
    del stripped["demand"]["pack_assignment"]
    del stripped["layout"]["extra_edges"]
    assert config_hash(stripped) == config_hash(baseline_config)


def test_hash_is_12_hex(baseline_config):
    h = config_hash(baseline_config)
    assert len(h) == 12
    int(h, 16)


def test_canonical_json_sorted_and_compact(baseline_config):
    cj = canonical_json(baseline_config)
    assert ": " not in cj and ", " not in cj
    assert cj.index('"demand"') < cj.index('"fleet"') < cj.index('"layout"')


def test_apply_patch_replaces_wholesale(baseline_config, braess_patch):
    patched = apply_patch(baseline_config, braess_patch["patch"])
    assert patched["layout"]["extra_edges"] == braess_patch["patch"]["layout.extra_edges"]
    assert patched["layout"]["edge_overrides"] == braess_patch["patch"]["layout.edge_overrides"]
    # untouched parts intact, base not mutated
    assert patched["fleet"]["amr_count"] == 4
    assert baseline_config["layout"]["extra_edges"] == []
    validate_config(patched)
    assert config_hash(patched) != config_hash(baseline_config)


def test_patch_unknown_path_rejected(baseline_config):
    with pytest.raises(ConfigError):
        apply_patch(baseline_config, {"fleet.nonexistent.deep": 1})


def test_out_of_bounds_rejected(baseline_config):
    bad = apply_patch(baseline_config, {"fleet.amr_count": 99})
    with pytest.raises(ConfigError) as ei:
        validate_config(bad)
    assert any("amr_count" in v for v in ei.value.violations)


def test_fill_defaults_fills_nested_items(baseline_config, braess_patch):
    patched = fill_defaults(apply_patch(baseline_config, braess_patch["patch"]))
    assert patched["layout"]["extra_edges"][0]["bidirectional"] is True
    assert patched["layout"]["edge_overrides"][0]["one_way"] is False
