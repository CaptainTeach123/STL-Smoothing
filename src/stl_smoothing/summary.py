"""Plain-text summary of what the smoother did.  Shared by the command line and the web page."""

from __future__ import annotations

import numpy as np

from .flatten import FlattenResult, Plateau
from .layers import LayerGrid
from .mesh import Mesh
from .slicing import contour_length

MAX_LINES = 12  # plateaus listed per section before the summary collapses the rest


def _fmt_plateau(p: Plateau) -> str:
    where = "top    " if p.facing == "up" else "ceiling"
    patches = f"{p.regions} patch{'es' if p.regions != 1 else ''}"
    if p.kind == "snapped":
        return f"  {where} {patches:<10} {p.area:9,.0f} mm²  flat at {p.level_before:.3f} mm -> {p.level_after:.2f} mm (on a sampling plane)"
    span = f"{p.z_low:.2f}–{p.z_high:.2f} mm"
    return (
        f"  {where} {patches:<10} {p.area:9,.0f} mm²  {span:<17} {p.layers_before} layer{'s' if p.layers_before != 1 else ' '}"
        f"  ->  {p.level_after:.2f} mm"
    )


def _list(title: str, plateaus: list[Plateau], out, verbose: bool, cli: bool = True) -> None:
    """Print a section of plateaus, collapsing a long tail so a model with thousands
    of patches does not flood the terminal."""
    out(title)
    ordered = sorted(plateaus, key=lambda p: -p.area)
    shown = ordered if verbose else ordered[:MAX_LINES]
    for p in shown:
        out(_fmt_plateau(p))
    rest = ordered[len(shown):]
    if rest:
        out(f"  ... and {len(rest):,} more ({sum(p.area for p in rest):,.0f} mm² in total"
            + ("; use -v to list all)" if cli else ")"))


def flattened_indices(res: FlattenResult) -> list[int]:
    """Plateaus that visibly changed: they used to end on several layers."""
    return [i for i, p in enumerate(res.plateaus) if p.kind == "smoothed" and p.layers_before > 1]


def layer_edge_totals(res: FlattenResult, before: Mesh, after: Mesh, grid: LayerGrid) -> tuple[float, float] | None:
    """Total length (mm) of slicer layer edges on the flattened surfaces, before and after."""
    flattened = flattened_indices(res)
    if not flattened:
        return None
    region = np.isin(res.face_plateau, flattened)
    return (
        contour_length(before, grid, region),
        contour_length(after, grid, region, after.verts),
    )


def summarize(res: FlattenResult, before: Mesh, after: Mesh, grid: LayerGrid, verbose: bool, out=print,
              cli: bool = True) -> None:
    """Write the summary line by line to ``out``.  ``cli=False`` words the hints for the web page
    (which has no ``-v`` or ``--max-range`` option)."""
    flattened = flattened_indices(res)
    levelled = [i for i, p in enumerate(res.plateaus) if p.kind == "smoothed" and p.layers_before <= 1]
    snapped = [i for i, p in enumerate(res.plateaus) if p.kind == "snapped"]
    pl = res.plateaus
    plural = lambda n: "s" if n != 1 else ""  # noqa: E731
    if flattened:
        _list(f"Flattened {len(flattened)} surface{plural(len(flattened))} "
              f"(heights above the bed, {grid.layer_height:g} mm layers):", [pl[i] for i in flattened], out, verbose, cli)
        e0, e1 = layer_edge_totals(res, before, after, grid)
        out(f"Layer edges on those surfaces: {e0:,.0f} mm -> {e1:,.0f} mm")
        if any(pl[i].z_high - pl[i].z_low > 1.0 for i in flattened):
            out("Note: a flattened surface varied by more than 1 mm. If it is really meant to be curved or "
                "sloped, run again with a smaller "
                + ("--max-range (for example --max-range 1)." if cli else
                   "“Largest wobble to flatten” (in the advanced settings), for example 1."))
    elif not (levelled or snapped):
        out("Nothing to flatten: no surface found that is meant to be flat but crosses layer boundaries.")
    if levelled:
        _list(f"Levelled {len(levelled)} nearly-flat surface{plural(len(levelled))} that grazed a slicer "
              f"sampling plane (they already printed on one layer, with a few stray layer edges):",
              [pl[i] for i in levelled], out, verbose, cli)
    if snapped:
        _list(f"Moved {len(snapped)} already-flat surface{plural(len(snapped))} off a slicer sampling plane "
              f"(by at most half a layer):", [pl[i] for i in snapped], out, verbose, cli)
    if res.already_flat and verbose:
        out("Already printing on a single layer (left unchanged):")
        for facing, lvl, area in res.already_flat:
            out(f"  {facing:<7} {area:9,.0f} mm²  at {lvl:.2f} mm")
    if res.n_moved:
        out(f"Moved {res.n_moved:,} vertices, largest move {res.max_dz:.2f} mm, in z only; "
            f"{res.flipped_faces} flipped faces, {res.degenerate_faces_added} new degenerate faces.")
    if res.skipped:
        out(f"Left alone: {len(res.skipped)} candidate{plural(len(res.skipped))}"
            + ("" if verbose or not cli else " (use -v for details)"))
        if verbose:
            for line in res.skipped:
                out(f"  {line}")
