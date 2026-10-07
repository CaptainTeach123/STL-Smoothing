"""Tunable parameters of the flattening algorithm.

Only a handful are meant for users (the first block); the rest are internal
constants that were tuned on synthetic scenes and the Batwing model and are
exposed so unusual models can be adjusted without editing code.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class Params:
    # ------------------------------------------------------------ user-facing
    # A surface counts as "meant to be flat" while it stays within this many
    # degrees of horizontal.  Measured on *smoothed* normals, so the facet noise
    # of marching-cubes / remeshed models does not matter.
    max_slope_deg: float = 5.0
    # Widest height variation (mm) that can still be ONE plateau.  Bigger
    # variations are taken to be intentional shapes (ramps, domes, steps).
    max_range: float = 2.0
    # Plateaus smaller than this (mm^2) are left alone.
    min_area: float = 40.0
    # Surfaces that already print on a single layer are left alone unless they
    # graze a slicer sampling plane; this is the clearance, as a share of the layer height.
    plane_margin: float = 0.05
    # Nudge exactly-flat plateaus that sit on / near a sampling plane onto the layer grid.
    snap_exact_flat: bool = True
    min_flat_area: float = 5.0  # mm^2; only exactly-flat patches at least this big are snapped
    # Surface patches this many degrees off horizontal are *walls*.
    wall_slope_deg: float = 12.0

    # ------------------------------------------------- robust classification
    smooth_radius: float = 1.2  # mm; reach of the normal smoothing used to classify faces
    smooth_max_rounds: int = 6
    family_slope_deg: float = 30.0  # faces steeper than this never take part in smoothing
    flat_tol: float = 1e-6  # mm; z-span of a face below which it is "exactly flat" (never less than 4 float32 ulps)
    exact_max_cut: float = 0.5  # exact patches continued smoothly by noisy faces are not intentional ...
    exact_demote_width: float = 4.0  # ... when they are thinner than this (mm, 2 x area / perimeter)
    planar_ramp_deg: float | None = 0.5  # exactly planar patch tilted at least this much = intentional ramp
    planar_tol_deg: float = 0.02
    planar_min_area: float = 20.0
    planar_min_faces: int = 2

    # ---------------------------------------------------- histogram / modes
    bin_frac: float = 0.25  # histogram bin = bin_frac * layer_height
    hist_sigma: float = 0.05  # mm, Gaussian smoothing of the height histogram
    merge_ratio: float = 0.35  # peaks whose valley is >= this share of the lower peak are one peak
    member_frac: float = 0.4
    edge_frac: float = 0.005
    edge_frac_ped: float = 0.3
    ped_scale: float = 0.15
    max_half_range: float = 1.0
    span_trim: float = 0.005
    min_prominence: float = 2.0  # peak density must beat the ramp / dome pedestal by this factor
    # A wobbly panel has MOST of its area near one height; a ramp, crown or bowl spreads its
    # area evenly over its height range.  Evenly spread surfaces wider than flat_top_range
    # (mm) are intentional shapes and are left alone.
    min_peakiness: float = 1.25
    flat_top_range: float = 1.0
    # Two patches of one plateau that meet at a wall at least this many layers high
    # (measured between their median heights) are a deliberate step: it is kept.
    step_layers: float = 1.5
    step_min_faces: int = 2

    # ------------------------------------------------- connectivity clean-up
    max_cut_frac: float = 0.5  # reject plateaus whose boundary is mostly a window cut
    same_level_width: float = 5.0  # mm; min mean width of a patch that may fade into an exactly flat plateau
    core_cut_frac: float = 0.65
    core_big_area: float = 300.0
    hole_area: float = 3000.0
    tail_ext: float = 0.8
    hole_rounds: int = 3
    spike_max: float = 1.0  # mm; a lone vertex surrounded by one plateau is pulled onto it up to this far

    # ---------------------------------------------------- moving vertices
    feather_slope_deg: float = 6.0
    feather_min: float = 0.8
    feather_max: float = 10.0
    feather_smooth_iters: int = 3
    steep_cost: float = 8.0
    unfold_keep: float = 0.25
    unfold_iters: int = 30
    damp_iters: int = 8

    def replace(self, **kw) -> "Params":
        return replace(self, **kw)

    @classmethod
    def names(cls) -> set[str]:
        return {f.name for f in fields(cls)}
