"""Score a smoothing algorithm on every synthetic scene (+ the real Batwing).

Usage::

    python tests/bench.py PATH/TO/algo.py [--small] [--only NAME ...] [--sample batwing.stl]

``algo.py`` must define::

    def smooth(mesh: Mesh, grid: LayerGrid) -> np.ndarray   # (V, 3) new vertices

with identical connectivity: only z may change.  Prints one line per scene
plus details for any failure.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import metrics  # noqa: E402
import scenes  # noqa: E402
from stl_smoothing import stlio  # noqa: E402
from stl_smoothing.layers import LayerGrid  # noqa: E402
from stl_smoothing.mesh import Mesh  # noqa: E402


def load_algo(path):
    spec = importlib.util.spec_from_file_location("algo_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    return mod


def run_scene(algo, sc, grid, check_idempotent=True):
    t = time.time()
    new = np.asarray(algo.smooth(sc.mesh, grid), dtype=np.float64)
    dt = time.time() - t
    assert new.shape == sc.mesh.verts.shape, "smooth() must return (V,3) with the same vertex count"
    res = metrics.evaluate(sc, new, grid)
    ok, why = metrics.verdict(res)
    # the level must stay within about a layer of what the surface already was
    for name, mask in sc.targets.items():
        vs = np.unique(sc.mesh.faces[mask])
        before_med = float(np.median(sc.mesh.verts[vs, 2]))
        after_med = float(np.median(new[vs, 2]))
        res["targets"][name]["level_shift"] = after_med - before_med
        if abs(after_med - before_med) > grid.layer_height * 1.01:
            ok = False
            why.append(f"target {name}: level moved {after_med - before_med:+.2f} mm")
    res["seconds"] = dt
    if check_idempotent:
        m2 = Mesh(new, sc.mesh.faces)
        new2 = np.asarray(algo.smooth(m2, grid), dtype=np.float64)
        res["idempotent_dz"] = float(np.abs(new2 - new).max())
        if res["idempotent_dz"] > 1e-6:
            ok = False
            why.append(f"not idempotent: 2nd pass moved {res['idempotent_dz']:.4f} mm")
    return ok, why, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("algo")
    ap.add_argument("--small", action="store_true", help="coarse 1 mm cells (fast)")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--holdout", action="store_true", help="held-out validation scenes (graders only)")
    ap.add_argument("--sample", default=os.environ.get("STL_SAMPLE"))
    ap.add_argument("--layer", type=float, default=0.2)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    algo = load_algo(a.algo)
    grid = LayerGrid(a.layer)
    all_scenes = scenes.all_holdout(small=a.small) if a.holdout else scenes.all_synthetic(small=a.small)
    if a.holdout and a.sample and os.path.exists(a.sample):
        after = Mesh.from_triangles(stlio.read_stl(a.sample).tris)
        all_scenes.append(scenes.batwing_before(after, seed=31, ptp=0.9))
        all_scenes.append(scenes.batwing_before(after, seed=32, ptp=1.4, style="bands"))
        all_scenes.append(scenes.batwing_before(after, seed=33, ptp=1.2, style="bands", wavelengths=(40.0, 140.0)))
    elif a.sample and os.path.exists(a.sample):
        after = Mesh.from_triangles(stlio.read_stl(a.sample).tris)
        all_scenes.append(scenes.batwing_before(after, seed=5))
        all_scenes.append(scenes.batwing_before(after, seed=8, ptp=1.0, wavelengths=(15.0, 120.0), style="bands"))
        # the already-fixed model: should (almost) not change, and be idempotent
        base = scenes.batwing_before(after, seed=5)
        all_scenes.append(scenes.Scene("batwing_after_noop", after, targets={"panel": base.targets["panel"]},
                                       protected=base.protected))
    npass = 0
    rows = []
    for sc in all_scenes:
        if a.only and sc.name not in a.only:
            continue
        try:
            ok, why, res = run_scene(algo, sc, grid)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok, why, res = False, [f"EXCEPTION {e!r}"], {"seconds": 0, "max_dz": 0, "n_changed": 0, "flipped": 0, "targets": {}, "protected": {}}
        npass += ok
        tg = "; ".join(f"{k}: {v['layers_before']}->{v['layers_after']} layers, edges {v['contour_before']:.0f}->{v['contour_after']:.1f}mm"
                       for k, v in res["targets"].items())
        pr = "; ".join(f"{k} drift {v['max_drift']:.3f}" for k, v in res["protected"].items())
        print(f"[{'PASS' if ok else 'FAIL'}] {sc.name:24s} {res['seconds']:6.2f}s maxdz={res['max_dz']:.2f} changed={res['n_changed']:6d} flips={res['flipped']} turn={res.get('normal_turn_max', 0):.0f}/{res.get('normal_turn_p999', 0):.0f}deg | {tg} | {pr}")
        for w in why:
            print("        -", w)
        rows.append(ok)
    print(f"\n{npass}/{len(rows)} scenes passed")
    sys.exit(0 if npass == len(rows) else 1)


if __name__ == "__main__":
    main()
