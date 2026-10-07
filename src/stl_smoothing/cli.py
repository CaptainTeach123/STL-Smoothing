"""Command line interface: ``stl-smoothing model.stl``."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

from . import __version__
from .flatten import FlattenResult, flatten
from .layers import LayerGrid
from .mesh import MIN_WELD_TOL, Mesh, auto_weld_tol
from .params import Params
from .summary import summarize
from .stlio import StlError, read_stl, write_stl

REPORT_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
OPTIONAL_PARAMS = {"planar_ramp_deg"}  # parameters where "none" switches the rule off


class UsageError(Exception):
    """A problem with the command line itself (exit code 2)."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stl-smoothing",
        description=(
            "Remove strangely offset layer lines from generated STL models. Surfaces that are meant "
            "to be flat but wobble by a fraction of a millimetre are moved (in z only) onto a single "
            "layer boundary, so the slicer prints each of them on ONE layer."
        ),
        epilog="Heights are measured from the lowest point of the model (where a slicer puts the bed).",
    )
    p.add_argument("input", help="input STL file (binary or ASCII)")
    p.add_argument("-o", "--output", help="output STL (default: <input>_smoothed.stl)")
    g = p.add_argument_group("slicer settings")
    g.add_argument("-l", "--layer-height", type=float, default=0.2, metavar="MM", help="layer height (default 0.2)")
    g.add_argument("--first-layer", type=float, default=None, metavar="MM",
                   help="first layer height (default: same as the layer height)")
    g = p.add_argument_group("what counts as a surface that should be flat")
    g.add_argument("--max-range", type=float, default=None, metavar="MM",
                   help="widest height variation of ONE flat surface (default %g). Anything that varies more "
                        "is an intentional shape. Lower it (e.g. 1.0) to protect gently curved plates and ramps; "
                        "raise it if a very wobbly surface is left alone" % Params.max_range)
    g.add_argument("--max-slope", type=float, default=None, metavar="DEG",
                   help="a surface is 'flat' while it stays within this many degrees of horizontal "
                        "(default %g, measured on smoothed normals so facet noise does not matter)" % Params.max_slope_deg)
    g.add_argument("--min-area", type=float, default=None, metavar="MM2",
                   help="ignore flat surfaces smaller than this (default %g)" % Params.min_area)
    g.add_argument("--no-snap-exact", action="store_true",
                   help="do not nudge already-flat surfaces that sit on a slicer sampling plane onto the layer grid")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="override any advanced parameter (see Params in the source); can be repeated")
    g = p.add_argument_group("output")
    g.add_argument("--analyze", action="store_true",
                   help="only report what would change; write no STL (a --report picture is still written)")
    g.add_argument("--report", metavar="PNG", help="write a before/after picture of the changed surfaces "
                                                 "(.png, .jpg, .pdf or .svg; needs matplotlib)")
    g.add_argument("--ascii", action="store_true", help="write an ASCII STL instead of binary")
    g.add_argument("--weld-tol", type=float, default=None, metavar="MM",
                   help="merge corners closer than this (default: about 5e-7 x the largest coordinate, at least "
                        "%g; 0 = exact matches only)" % MIN_WELD_TOL)
    g.add_argument("-v", "--verbose", action="store_true", help="also list what was found but left alone, and why")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _coerce(name: str, raw: str):
    kinds = {f.name: f.type for f in fields(Params)}
    if name not in kinds:
        raise UsageError(f"--set: unknown parameter {name!r}; known: {', '.join(sorted(kinds))}")
    if raw.lower() in ("none", "null") and name in OPTIONAL_PARAMS:
        return None
    default = getattr(Params, name)
    if isinstance(default, bool):
        low = raw.lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise UsageError(f"--set {name}: expected true/false, got {raw!r}")
    try:
        return type(default)(raw)
    except (TypeError, ValueError):
        raise UsageError(f"--set {name}: cannot read {raw!r}") from None


def params_from_args(a: argparse.Namespace) -> Params:
    kw = {}
    if a.max_range is not None:
        kw["max_range"] = a.max_range
    if a.max_slope is not None:
        kw["max_slope_deg"] = a.max_slope
    if a.min_area is not None:
        kw["min_area"] = a.min_area
    if a.no_snap_exact:
        kw["snap_exact_flat"] = False
    for item in a.set:
        if "=" not in item:
            raise UsageError(f"--set expects NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        kw[k.strip()] = _coerce(k.strip(), v.strip())
    return Params().replace(**kw)


def _preflight(a: argparse.Namespace, src: Path, dst: Path) -> tuple[str, Path | None]:
    """Check the paths before any heavy work; returns (error message or "", report path)."""
    if not src.is_file():
        return f"{src}: no such file", None
    report = None
    if a.report:
        report = Path(a.report)
        if report.suffix == "":
            report = report.with_suffix(".png")
        elif report.suffix.lower() not in REPORT_SUFFIXES:
            return f"--report: unsupported file type {report.suffix!r} (use .png, .jpg, .pdf or .svg)", None
        if not (report.parent.is_dir()):
            return f"--report: the directory {report.parent} does not exist", None
    if not a.analyze:
        if dst.resolve() == src.resolve():
            return "the output would overwrite the input; choose a different -o", None
        if dst.is_dir():
            return f"-o: {dst} is a directory", None
        if not dst.parent.is_dir():
            return f"-o: the directory {dst.parent} does not exist", None
    return "", report


def run(argv: list[str] | None = None, out=print, err=None) -> int:
    """Run the tool; returns the exit code (0 ok, 1 the data could not be processed, 2 bad usage)."""
    err = err or out
    a = build_parser().parse_args(argv)
    try:
        P = params_from_args(a)
        grid = LayerGrid(a.layer_height, a.first_layer)
    except (UsageError, ValueError) as exc:
        err(f"error: {exc}")
        return 2
    src = Path(a.input)
    dst = Path(a.output) if a.output else src.with_name(src.stem + "_smoothed.stl")
    problem, report_path = _preflight(a, src, dst)
    if problem:
        err(f"error: {problem}")
        return 2

    try:
        data = read_stl(src)
    except (StlError, OSError) as exc:
        err(f"error: {exc}")
        return 1
    if len(data.tris) == 0:
        err(f"error: {src}: the file contains no triangles")
        return 1
    if not np.isfinite(data.tris).all():
        err(f"error: {src}: the file contains NaN or infinite coordinates")
        return 1
    tol = auto_weld_tol(data.tris) if a.weld_tol is None else a.weld_tol
    mesh = Mesh.from_triangles(data.tris, tol=tol if tol > 0 else None)
    del data.tris  # the float64 triangle soup is large and no longer needed
    stats = mesh.edge_manifold_stats()
    lo, hi = mesh.bbox()
    ext = hi - lo
    out(f"{src.name}: {mesh.n_faces:,} triangles, {mesh.n_verts:,} vertices, "
        f"{ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm")
    if not stats["closed"]:
        out(f"warning: the mesh is not closed ({stats['boundary_edges']} open edges, "
            f"{stats['non_manifold_edges']} non-manifold); results may be incomplete")

    res = flatten(mesh, grid, P)
    after = Mesh(res.verts, mesh.faces)
    summarize(res, mesh, after, grid, a.verbose, out)

    # The picture is a convenience: whatever goes wrong with it must not stop the STL being written.
    if report_path is not None:
        if not (res.smoothed_faces.any() or res.snapped_faces.any()):
            out("No changed surfaces: no report written.")
        else:
            try:
                _write_reports(report_path, mesh, after, grid, res, out)
            except Exception as exc:  # noqa: BLE001
                out(f"warning: could not write the report: {exc}")

    if a.analyze:
        out("Analyze only: no STL written.")
        return 0
    if not res.n_moved:
        out("Nothing was changed; no STL written.")
        return 0
    try:
        write_stl(
            dst,
            after.to_triangles(),
            binary=not a.ascii,
            header=data.header if (data.source_format == "binary" and data.header.strip(b"\x00")) else b"stl-smoothing",
            attrs=data.attrs,
            name=data.name or src.stem,
        )
    except OSError as exc:
        err(f"error: cannot write {dst}: {exc}")
        return 1
    out(f"Wrote {dst}")
    return 0


def _write_reports(base: Path, before: Mesh, after: Mesh, grid: LayerGrid, res: FlattenResult, out) -> None:
    try:
        from .report import render_comparison
    except Exception as exc:  # pragma: no cover
        out(f"warning: cannot write the report: {exc}")
        return
    # Only surfaces that visibly changed (they used to end on several layers) are highlighted.
    changed = [i for i, p in enumerate(res.plateaus) if p.kind == "smoothed" and p.layers_before > 1]
    if not changed:
        out("No surface changed layers: no report written.")
        return
    region = np.isin(res.face_plateau, changed)
    has_tops = bool((region & (res.facing == 1)).any())
    for facing, view, what in ((1, "top", "Flattened top surfaces"), (-1, "bottom", "Flattened ceilings (seen from below)")):
        mask = region & (res.facing == facing)
        if not mask.any():
            continue
        n_here = len({int(i) for i in res.face_plateau[mask]})
        target = base if facing == 1 or not has_tops else base.with_name(base.stem + "_ceilings" + base.suffix)
        try:
            info = render_comparison(before, after, grid, mask, str(target), view=view, what=what)
        except RuntimeError as exc:  # matplotlib missing
            out(f"warning: {exc}")
            return
        out(f"Wrote report {target} ({n_here} surface{'s' if n_here != 1 else ''}: "
            f"{len(info['before_layers'])} layers before, {len(info['after_layers'])} after)")


def main() -> None:  # pragma: no cover - console entry point
    # Never die on a terminal that cannot show the characters in our messages (mm², –).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    sys.exit(run(err=lambda line: print(line, file=sys.stderr)))
