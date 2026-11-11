# seamsh - Copyright (C) <2010-2020>
# <Universite catholique de Louvain (UCL), Belgium
#
# List of the contributors to the development of seamsh: see AUTHORS file.
# Description and complete License: see LICENSE file.
#
# This program (seamsh) is free software:
# you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program (see COPYING file).  If not,
# see <http://www.gnu.org/licenses/>.

"""Internals shared by the seamsh modules: reprojection of points and the binding of the C library
(`seamshlib/`, built as `libseamsh.so` next to this file and loaded with ctypes)."""

import ctypes
import os

import numpy as np
from osgeo import osr

# Coordinate transformations are costly to build: they are kept, by pair of projection objects.
_transform_cache = {}


def project_points(x, pfrom: osr.SpatialReference, pto: osr.SpatialReference) -> np.ndarray:
    """Points `x` [n, >= axes of pfrom], given in projection `pfrom`, in projection `pto`
    [n, axes of pto]. When both projections are the same, only the extra columns are dropped."""
    x = np.asarray(x)[:, : pfrom.GetAxesCount()]
    if not pfrom.IsSame(pto):
        trans = _transform_cache.get((pfrom, pto))
        if trans is None:
            trans = osr.CoordinateTransformation(pfrom, pto)
            _transform_cache[(pfrom, pto)] = trans
        x = trans.TransformPoints(x)
        x = np.asarray(x)[:, : pto.GetAxesCount()]
    return x


lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)), "libseamsh.so"))

lib.polymesh_new.restype = ctypes.c_void_p
lib.polymesh_new.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.polymesh_add_points.restype = None
lib.polymesh_add_points.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
lib.polymesh_n_faces.restype = ctypes.c_int
lib.polymesh_n_faces.argtypes = [ctypes.c_void_p]
lib.polymesh_faces.restype = None
lib.polymesh_faces.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.gen_boundaries_from_points.restype = ctypes.c_int
lib.gen_boundaries_from_points.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
    ctypes.POINTER(ctypes.c_int),
]
lib.libcfree.restype = None
lib.libcfree.argtypes = [ctypes.c_void_p]

_COARSENING_ERRORS = {1: "out of memory", 2: "inconsistent triangulation"}


def np2c(a, dtype=np.float64) -> ctypes.c_void_p:
    """A pointer to the data of array `a` as a C-contiguous array of `dtype` (a itself if it already
    is one, so that the C code can modify it in place; a copy otherwise). The array is kept alive
    as long as the pointer."""
    tmp = np.require(a, dtype, "C")
    r = ctypes.c_void_p(tmp.ctypes.data)
    r.tmp = tmp
    return r


def gen_boundaries(
    x: np.ndarray, tag_lists: np.ndarray, triangles: np.ndarray, seeds, sizes: np.ndarray
) -> tuple:
    """Coarsened boundaries of the domain around seed triangles: see `seamshlib/seamsh.c`.

    Args:
        x: the points sampled on the boundary curves [n, 2].
        tag_lists: the tags of the points, as built by `geometry._point_tag_lists`.
        triangles: a Delaunay triangulation of the points [m, 3].
        seeds: the indices of triangles inside the domain (at least one).
        sizes: the target size at each point [n].
    Returns:
        The boundary points [k, 2], and the boundary curves as a flat int array: for each curve
        its tag, the indices of its points, then -1.
    """
    # The C code works in place on its arguments: give it copies.
    x = np.array(x, dtype=np.float64, order="C")
    tag_lists = np.array(tag_lists, dtype=np.int32, order="C")
    triangles = np.array(triangles, dtype=np.int32, order="C")
    sizes = np.array(sizes, dtype=np.float64, order="C")
    seeds = np.atleast_1d(np.array(seeds, dtype=np.int32, order="C"))
    n_lines, n_points = ctypes.c_int(), ctypes.c_int()
    p_points = ctypes.POINTER(ctypes.c_double)()
    p_lines = ctypes.POINTER(ctypes.c_int)()
    error = lib.gen_boundaries_from_points(
        x.shape[0],
        np2c(x),
        np2c(tag_lists, np.int32),
        triangles.shape[0],
        np2c(triangles, np.int32),
        seeds.shape[0],
        np2c(seeds, np.int32),
        np2c(sizes),
        ctypes.byref(p_points),
        ctypes.byref(n_points),
        ctypes.byref(p_lines),
        ctypes.byref(n_lines),
    )
    if error != 0:
        reason = _COARSENING_ERRORS.get(error, f"error {error}")
        raise RuntimeError(f"Boundary coarsening failed: {reason}")
    # The results are allocated by the C code: copy them and give the memory back.
    points = np.empty((0, 2))
    lines = np.empty((0,), np.int32)
    if n_points.value > 0:
        points = np.ctypeslib.as_array(p_points, (n_points.value, 2)).copy()
    if n_lines.value > 0:
        lines = np.ctypeslib.as_array(p_lines, (n_lines.value,)).copy()
    lib.libcfree(p_points)
    lib.libcfree(p_lines)
    return points, lines


class PolyMesh:
    """A Delaunay triangulation to which points can be added (gmsh's `polymesh.c`), used for the
    background mesh that carries the mesh size."""

    def __init__(self, bbmin, bbmax):
        """
        Args:
            bbmin, bbmax: lower and upper corners of a box that holds all the points to come.
        """
        self._ptr = ctypes.c_void_p(lib.polymesh_new(np2c(bbmin), np2c(bbmax)))
        self._n = 0

    def add_points(self, v: np.ndarray) -> None:
        """Adds points [n, 2]; they are numbered in the order they are added."""
        tags = np.arange(self._n, self._n + v.shape[0], dtype=np.int32)
        self._n += v.shape[0]
        lib.polymesh_add_points(self._ptr, v.shape[0], np2c(v), np2c(tags, np.int32))

    def get_triangles(self) -> np.ndarray:
        """The triangles [m, 3], as indices of the points."""
        n = lib.polymesh_n_faces(self._ptr)
        tri = np.empty((n, 3), np.int32)
        lib.polymesh_faces(self._ptr, np2c(tri, np.int32))
        return tri
