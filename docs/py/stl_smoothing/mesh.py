"""Indexed triangle mesh with the topology helpers the smoother needs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree


MIN_WELD_TOL = 5e-5  # mm


def auto_weld_tol(tris: np.ndarray) -> float:
    """Weld tolerance that follows the float32 resolution of the coordinates.

    Exporters sometimes leave copies of a shared corner that differ in the last
    bit or two; at |coordinate| of a few hundred mm that is more than a fixed
    0.05 micron.
    """
    return max(MIN_WELD_TOL, 5e-7 * float(np.abs(tris).max()))


@dataclass
class Mesh:
    """Welded triangle mesh.  ``faces`` index into ``verts``.

    The smoother only ever moves vertices in z; the face array is shared
    between the input and the output so triangle count, order and winding are
    preserved exactly.
    """

    verts: np.ndarray  # (V, 3) float64
    faces: np.ndarray  # (F, 3) int64

    # ------------------------------------------------------------------ build
    @classmethod
    def from_triangles(cls, tris: np.ndarray, tol: float | None = None) -> "Mesh":
        """Weld a triangle soup.

        Corners are first merged when their float32 values are bit-identical (what
        every STL exporter produces for shared corners).  With ``tol`` (mm), corners
        closer than that are merged as well: some exporters leave copies of a shared
        corner that differ in the last bit, which would otherwise crack the surface
        wherever the smoother moves one copy but not the other.
        """
        pts = np.asarray(tris, dtype=np.float64).reshape(-1, 3)
        if len(pts) == 0:
            return cls(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        key = pts.astype(np.float32) + np.float32(0.0)  # fold -0.0 into 0.0
        k = (np.ascontiguousarray(key).view(np.int32).astype(np.int64) + (1 << 31)).astype(np.uint64)  # unsigned order == signed order
        # two 1-D uniques (x,y packed, then rank(x,y) with z) instead of np.unique(axis=0): same order, 3-4x faster
        s32 = np.uint64(32)
        _, ixy = np.unique((k[:, 0] << s32) | k[:, 1], return_inverse=True)
        _, first, inv = np.unique((ixy.reshape(-1).astype(np.uint64) << s32) | k[:, 2], return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        verts = pts[first]
        if tol and tol > 0 and len(verts) > 1:
            pairs = cKDTree(verts).query_pairs(float(tol), output_type="ndarray")
            if len(pairs):
                n = len(verts)
                g = sparse.coo_matrix(
                    (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
                )
                ncomp, lab = csgraph.connected_components(g, directed=False)
                rep = np.full(ncomp, n, dtype=np.int64)
                np.minimum.at(rep, lab, np.arange(n))  # lowest index of each cluster represents it
                keep = np.unique(rep)
                remap = np.empty(ncomp, dtype=np.int64)
                remap[np.argsort(rep)] = np.arange(ncomp)
                new_index = remap[lab]
                verts = verts[rep[np.argsort(rep)]]
                inv = new_index[inv]
        faces = inv.reshape(-1, 3).astype(np.int64)
        return cls(verts, faces)

    def to_triangles(self) -> np.ndarray:
        return self.verts[self.faces]

    def with_z(self, z: np.ndarray) -> "Mesh":
        v = self.verts.copy()
        v[:, 2] = z
        return Mesh(v, self.faces)

    def copy(self) -> "Mesh":
        return Mesh(self.verts.copy(), self.faces.copy())

    # --------------------------------------------------------------- geometry
    @property
    def n_verts(self) -> int:
        return len(self.verts)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def face_cross(self, verts: np.ndarray | None = None) -> np.ndarray:
        """Unnormalised face normals (length = 2 x area)."""
        v = self.verts if verts is None else verts
        t = v[self.faces]
        return np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])

    def face_normals_areas(self, verts: np.ndarray | None = None):
        c = self.face_cross(verts)
        ln = np.linalg.norm(c, axis=1)
        n = np.divide(c, ln[:, None], out=np.zeros_like(c), where=ln[:, None] > 0)
        return n, 0.5 * ln

    def face_centroids(self, verts: np.ndarray | None = None) -> np.ndarray:
        v = self.verts if verts is None else verts
        return v[self.faces].mean(axis=1)

    def bbox(self):
        return self.verts.min(axis=0), self.verts.max(axis=0)

    # --------------------------------------------------------------- topology
    def half_edge_keys(self) -> np.ndarray:
        """Sorted-pair key for each of the 3F half edges, shape (3F,).

        Half edge ``3*f + k`` runs from corner ``k`` to corner ``k+1`` of face f.
        """
        f = self.faces
        a = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
        b = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        return lo * np.int64(self.n_verts) + hi

    def edges(self):
        """Unique undirected edges ``(E, 2)`` and how many faces use each."""
        keys = self.half_edge_keys()
        uk, counts = np.unique(keys, return_counts=True)
        n = np.int64(self.n_verts)
        return np.stack([uk // n, uk % n], axis=1), counts

    def face_adjacency(self) -> np.ndarray:
        """Pairs ``(F_i, F_j)`` of faces sharing an edge, shape (P, 2).

        Faces on a non-manifold edge (3+ faces) are chained pairwise.
        """
        keys = self.half_edge_keys()
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        same = sk[1:] == sk[:-1]
        fa = (order[:-1] % self.n_faces)[same]
        fb = (order[1:] % self.n_faces)[same]
        return np.stack([fa, fb], axis=1)

    def vertex_graph(self, weighted: bool = True) -> sparse.csr_matrix:
        """Symmetric vertex adjacency; edge weights are 3-D edge lengths."""
        e, _ = self.edges()
        if weighted:
            w = np.linalg.norm(self.verts[e[:, 0]] - self.verts[e[:, 1]], axis=1)
            w = np.maximum(w, 1e-12)  # csgraph treats explicit zeros as no edge
        else:
            w = np.ones(len(e))
        n = self.n_verts
        g = sparse.coo_matrix(
            (np.concatenate([w, w]), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
            shape=(n, n),
        )
        return g.tocsr()

    def face_components(self, mask: np.ndarray) -> tuple[int, np.ndarray]:
        """Connected components of the faces selected by ``mask``.

        Components connect through shared edges.  Returns ``(count, labels)``
        where ``labels[f]`` is the component of face f, or ``-1`` outside the mask.
        """
        mask = np.asarray(mask, dtype=bool)
        adj = self.face_adjacency()
        keep = mask[adj[:, 0]] & mask[adj[:, 1]]
        adj = adj[keep]
        n = self.n_faces
        g = sparse.coo_matrix(
            (np.ones(len(adj), dtype=np.int8), (adj[:, 0], adj[:, 1])), shape=(n, n)
        )
        ncomp, lab = csgraph.connected_components(g, directed=False)
        labels = np.where(mask, lab, -1)
        # Re-number so labels are 0..k-1 over masked faces only.
        used = np.unique(labels[mask])
        remap = -np.ones(ncomp, dtype=np.int64)
        remap[used] = np.arange(len(used))
        labels = np.where(mask, remap[np.maximum(labels, 0)], -1)
        return len(used), labels

    # ------------------------------------------------------------- validation
    def edge_manifold_stats(self) -> dict:
        _, counts = self.edges()
        return {
            "edges": int(len(counts)),
            "boundary_edges": int((counts == 1).sum()),
            "non_manifold_edges": int((counts > 2).sum()),
            "closed": bool((counts == 2).all()),
        }

    def flipped_faces(self, new_verts: np.ndarray, min_area: float = 1e-9) -> np.ndarray:
        """Boolean mask of faces whose orientation inverts when moved to ``new_verts``."""
        c0 = self.face_cross(self.verts)
        c1 = self.face_cross(new_verts)
        dot = (c0 * c1).sum(axis=1)
        big = 0.5 * np.linalg.norm(c0, axis=1) > min_area
        return big & (dot < 0)

    def degenerate_faces(self, verts: np.ndarray | None = None, tol: float = 1e-12) -> np.ndarray:
        _, area = self.face_normals_areas(verts)
        return area <= tol
