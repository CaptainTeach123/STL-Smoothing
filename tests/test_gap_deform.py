"""Unit tests for the wall-protection stages of ``_deform``.

None of the synthetic scenes ever folds a wall (the number of faces that flip before the unfold
step is 0 on all of them; the real model has 1-2), so ``unfold`` and ``repair_flips`` are only
reachable from tests that call them directly.
"""

import types

import numpy as np
import pytest

from stl_smoothing._deform import repair_flips, unfold
from stl_smoothing.mesh import Mesh
from stl_smoothing.params import Params

P = Params()


def wall():
    """A vertical wall (y = 0): bottom row v0, v1 at z=0, top row v2, v3 at z=1, normal -y."""
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1.0]])
    f = np.array([[0, 1, 3], [0, 3, 2]])
    m = Mesh(v, f)
    assert (m.face_normals_areas()[0][:, 1] == -1).all()
    A = types.SimpleNamespace(valid=np.ones(2, bool), soft={1: np.zeros(2, bool), -1: np.zeros(2, bool)})
    return m, A


def test_unfold_pushes_the_free_end_of_a_wall_to_keep_its_order():
    m, A = wall()
    z0 = m.verts[:, 2].copy()
    z = z0.copy()
    z[2:] = -0.5  # the (pinned) top row has been pulled below the free bottom row
    out = unfold(m, A, z0, z, np.array([False, False, True, True]), P)
    assert (out[2:] == -0.5).all()  # pinned vertices never move
    # the free bottom row follows: the top stays above it by 25 % of the original wall height (a literal:
    # the test must not read the very parameter whose value it is checking)
    assert (out[2:] - out[:2] >= 0.25 * 1.0 - 1e-9).all()
    nv = m.verts.copy()
    nv[:, 2] = out
    assert not m.flipped_faces(nv).any()


def test_repair_flips_damps_the_move_instead_of_cancelling_it():
    m, A = wall()
    z0 = m.verts[:, 2].copy()
    new = z0.copy()
    new[2:] = -0.5  # top row dragged through the bottom row: both faces flip
    nv = m.verts.copy()
    nv[:, 2] = new
    assert m.flipped_faces(nv).all()
    out = repair_flips(m, new, z0, new - z0, P)
    nv[:, 2] = out
    assert not m.flipped_faces(nv).any()
    assert (out[2:] < z0[2:]).all()  # part of the move survives (halved), it is not simply reverted
