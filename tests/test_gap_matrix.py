"""Run every scene of the benchmark (synthetic + holdout) through the pass/fail verdict.

``scenes.all_synthetic`` / ``scenes.all_holdout`` and ``metrics.verdict`` were only used by
``bench.py`` (never by pytest), and ``metrics.normal_turn_max`` (collateral damage on faces that
must NOT change) was computed but never asserted.  Without both, removing the feathering, the
wall-slope limit or the unfold step leaves the whole suite green on a bare checkout (only the
optional real-model test, which needs STL_SMOOTHING_SAMPLE, notices).
"""

import pytest

import metrics
import scenes
from stl_smoothing.flatten import flatten
from stl_smoothing.layers import LayerGrid

G = LayerGrid(0.2)

SCENES = (
    [("syn", i, s) for i, s in enumerate(scenes.all_synthetic(small=True))]
    + [("hold", i, s) for i, s in enumerate(scenes.all_holdout(small=True))]
)

# README limit: skirt scenes leave tiny residual layer edges.  Everything else must pass cleanly.
KNOWN_RESIDUAL = {("hold", 4): 2.0}  # mm of layer edges tolerated

# Collateral damage budget (degrees the normal of a face outside the targets may turn).  Measured
# on the shipped code with generous head-room; a missing feather / wall-slope guard blows through it.
TURN_CAP = {
    ("syn", 4): 20.0,   # panel_with_dome (measured 11.6)
    ("syn", 5): 30.0,   # panel_with_dome + skirt (measured 20.0)
    ("syn", 6): 50.0,   # panel_with_dome + soft skirt (measured 37.9)
    ("syn", 7): 6.0,    # two_level_panels (measured 2.8)
    ("syn", 8): 4.5,    # ceiling_pocket (measured 2.9)
    ("syn", 9): 26.0,   # terraces_with_ramp (measured 20.9)
    ("hold", 3): 15.0,  # panel_with_dome (measured 9.0)
    ("hold", 4): 25.0,  # panel_with_dome + skirt (measured 12.6)
    ("hold", 5): 4.5,   # ceiling_pocket (measured 3.1)
    ("hold", 6): 24.0,  # terraces_with_ramp (measured 18.3)
    ("hold", 8): 19.0,  # plate_with_exact_blocks (measured 17.5)
}


@pytest.mark.parametrize("kind, idx, scene", SCENES, ids=[f"{k}{i}-{s.name}" for k, i, s in SCENES])
def test_benchmark_scene_passes_the_verdict(kind, idx, scene):
    res = flatten(scene.mesh, G)
    m = metrics.evaluate(scene, res.verts, G)
    ok, why = metrics.verdict(m)
    if (kind, idx) in KNOWN_RESIDUAL:
        assert not [w for w in why if "layer edges" not in w], why
        assert all(t["contour_after"] < KNOWN_RESIDUAL[(kind, idx)] for t in m["targets"].values())
    else:
        assert ok, why
    assert res.flipped_faces == 0 and m["flipped"] == 0 and m["xy_moved_max"] == 0.0
    cap = TURN_CAP.get((kind, idx))
    if cap is not None:
        assert m["normal_turn_max"] < cap, f"collateral deformation {m['normal_turn_max']:.1f} deg >= {cap}"


@pytest.mark.parametrize("kind, idx, scene", SCENES, ids=[f"{k}{i}-{s.name}" for k, i, s in SCENES])
def test_benchmark_scene_is_idempotent(kind, idx, scene):
    """README: 'Running it again on its own output normally changes nothing' (checked after the
    float32 round trip an STL file imposes; test_idempotent only covers one scene)."""
    import numpy as np
    from stl_smoothing.mesh import Mesh

    first = flatten(scene.mesh, G)
    v1 = first.verts.astype(np.float32).astype(np.float64)
    second = flatten(Mesh(v1, scene.mesh.faces), G)
    assert np.abs(second.verts[:, 2] - v1[:, 2]).max() < 1e-6
