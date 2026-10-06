"""Model of how a slicer cuts a part into layers.

Layer ``j`` (1-based) spans ``(b[j-1], b[j]]`` with ``b[0] = 0`` and
``b[j] = first_layer + (j - 1) * layer_height``.  Slicers sample each layer at
its mid-height, so a top surface at height ``z`` is solid in layer ``j`` iff
``mid[j] < z``.  A surface sitting exactly on a layer boundary ``b[j]`` is
therefore half a layer away from both neighbouring sample planes, which makes
it the most robust place to put a flat top (or a flat ceiling).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LayerGrid:
    layer_height: float = 0.2
    first_layer: float | None = None  # None -> same as layer_height

    def __post_init__(self):
        if not self.layer_height > 0:
            raise ValueError("layer_height must be positive")
        if self.first_layer is not None and not self.first_layer > 0:
            raise ValueError("first_layer must be positive")

    @property
    def first(self) -> float:
        return self.layer_height if self.first_layer is None else self.first_layer

    # -------------------------------------------------------------- boundaries
    def boundary(self, j) -> np.ndarray:
        """Top height of layer ``j`` (1-based); ``boundary(0) == 0``."""
        j = np.asarray(j)
        return np.where(j <= 0, 0.0, self.first + (j - 1) * self.layer_height)

    def snap(self, z) -> np.ndarray:
        """Nearest layer boundary (never below the first layer top)."""
        z = np.asarray(z, dtype=np.float64)
        j = np.rint((z - self.first) / self.layer_height) + 1
        j = np.maximum(j, 1)
        return self.boundary(j)

    # ------------------------------------------------------------ slicing view
    def layer_number(self, z) -> np.ndarray:
        """1-based number of the last layer a top surface at ``z`` is solid in.

        Returns 0 when the surface is too low to print at all.
        """
        z = np.asarray(z, dtype=np.float64)
        j = np.ceil((z - self.first) / self.layer_height + 0.5)
        j = np.where(z > self.first / 2, np.maximum(j, 1), 0)
        return j.astype(np.int64)

    def layer_end(self, z) -> np.ndarray:
        """Height at which the layer a top surface at ``z`` ends."""
        return self.boundary(self.layer_number(z))

    def within_one_layer(self, zmin, zmax, margin: float = 0.05) -> bool:
        """True if every height in ``[zmin, zmax]`` prints on the same layer.

        ``margin`` (a fraction of the layer height) keeps the heights clear of the
        slicer's sampling planes: a surface that grazes a plane may print on either
        layer, so it counts as *not* safely inside one.
        """
        m = margin * self.layer_height + 1e-6  # inclusive of the margin itself
        return bool(self.layer_number(zmin - m) == self.layer_number(zmax + m))

    def layer_count_between(self, z0: float, z1: float) -> int:
        """How many distinct layers separate surfaces at ``z0`` and ``z1``."""
        return int(abs(self.layer_number(z1) - self.layer_number(z0)))
