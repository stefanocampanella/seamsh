# SPDX-FileCopyrightText: 2010-2020 Universite catholique de Louvain (UCL), Belgium
# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# Moved out of upstream's seamsh/_tools.py (LGPL, see LICENSE.txt), and rewritten.

"""Binding of seamsh's C library (`seamshlib/`, built as `libseamsh.so` next to this file), with ctypes.

The argument and result types of every C function are declared once, here; arrays are passed as numpy arrays
(`numpy.ctypeslib.ndpointer`), which ctypes checks and keeps alive for the length of the call. Two things are
wrapped:

- `coarsen_boundary_points`: the boundary coarsening (`gen_boundaries_from_points` in `seamshlib/seamsh.c`);
- `PolyMesh`: gmsh's Delaunay kernel (`seamshlib/polymesh.c`), which holds the background mesh of the mesh size.
"""

import ctypes
import dataclasses
import weakref
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer
from numpy.typing import ArrayLike

_LIBRARY = Path(__file__).resolve().parent / "libseamsh.so"
try:
  _lib = ctypes.CDLL(str(_LIBRARY))
except OSError as error:
  raise ImportError(
    f"seamsh's C library {_LIBRARY} cannot be loaded ({error}): install seamsh, or build the library in the source "
    "tree with `python setup.py build_ext --inplace`"
  ) from error

# Arrays of the C side: C-contiguous, of the C type, and writable (the coarsening works in place on its inputs).
_doubles = ndpointer(np.float64, flags=("C_CONTIGUOUS", "WRITEABLE"))
_ints = ndpointer(np.int32, flags=("C_CONTIGUOUS", "WRITEABLE"))

_lib.gen_boundaries_from_points.restype = ctypes.c_int
_lib.gen_boundaries_from_points.argtypes = [
  ctypes.c_int,  # number of points
  _doubles,  # points [n, 2]
  _ints,  # tag lists [2 n + ...]
  ctypes.c_int,  # number of triangles
  _ints,  # triangles [m, 3]
  ctypes.c_int,  # number of seed triangles
  _ints,  # seed triangles
  _doubles,  # size at each point [n]
  ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # out: boundary points, allocated by the library
  ctypes.POINTER(ctypes.c_int),  # out: number of boundary points
  ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # out: boundary curves, allocated by the library
  ctypes.POINTER(ctypes.c_int),  # out: length of the curves array
  ctypes.POINTER(ctypes.c_int),  # out: whether the coarsening reached the requested size
]
_lib.libcfree.restype = None
_lib.libcfree.argtypes = [ctypes.c_void_p]

_lib.polymesh_new.restype = ctypes.c_void_p
_lib.polymesh_new.argtypes = [_doubles, _doubles]
_lib.polymesh_add_points.restype = None
_lib.polymesh_add_points.argtypes = [ctypes.c_void_p, ctypes.c_int, _doubles, _ints]
_lib.polymesh_n_faces.restype = ctypes.c_int
_lib.polymesh_n_faces.argtypes = [ctypes.c_void_p]
_lib.polymesh_faces.restype = None
_lib.polymesh_faces.argtypes = [ctypes.c_void_p, _ints]
_lib.polymesh_delete.restype = None
_lib.polymesh_delete.argtypes = [ctypes.c_void_p]

_COARSENING_ERRORS = {1: "out of memory", 2: "inconsistent triangulation"}


@dataclasses.dataclass(frozen=True)
class CoarseningResult:
  """The output of `coarsen_boundary_points`."""

  points: np.ndarray
  """The boundary points [k, 2]."""
  curves: np.ndarray
  """The boundary curves, as a flat int array: for each curve its tag, the indices of its points, then -1."""
  converged: bool
  """Whether the coarsening reached the requested size everywhere (it stops after a number of sweeps)."""


def coarsen_boundary_points(
  points: ArrayLike, tag_lists: ArrayLike, triangles: ArrayLike, seeds: ArrayLike, sizes: ArrayLike
) -> CoarseningResult:
  """Coarsened boundaries of the domain around seed triangles: see `seamshlib/seamsh.c`.

  Args:
    points: the points sampled on the boundary curves [n, 2].
    tag_lists: the tags of the points, as built by `geometry._point_tag_lists`.
    triangles: a Delaunay triangulation of the points [m, 3].
    seeds: the indices of triangles inside the domain (at least one).
    sizes: the target size at each point [n].
  Raises:
    RuntimeError: if the library fails (out of memory, or a triangulation that is not one).
  """
  # The library works in place on its arguments: give it copies, of its types.
  points = np.array(points, dtype=np.float64, order="C")
  tag_lists = np.array(tag_lists, dtype=np.int32, order="C")
  triangles = np.array(triangles, dtype=np.int32, order="C")
  sizes = np.array(sizes, dtype=np.float64, order="C")
  seeds = np.atleast_1d(np.array(seeds, dtype=np.int32, order="C"))
  out_points = ctypes.POINTER(ctypes.c_double)()
  out_curves = ctypes.POINTER(ctypes.c_int)()
  n_points, n_curves, converged = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
  error = _lib.gen_boundaries_from_points(
    points.shape[0],
    points,
    tag_lists,
    triangles.shape[0],
    triangles,
    seeds.shape[0],
    seeds,
    sizes,
    ctypes.byref(out_points),
    ctypes.byref(n_points),
    ctypes.byref(out_curves),
    ctypes.byref(n_curves),
    ctypes.byref(converged),
  )
  # The outputs are allocated by the library (NULL on error): copy them, then give the memory back.
  try:
    if error != 0:
      raise RuntimeError(f"Boundary coarsening failed: {_COARSENING_ERRORS.get(error, f'error {error}')}")
    result_points = np.empty((0, 2))
    result_curves = np.empty((0,), np.int32)
    if n_points.value > 0:
      result_points = np.ctypeslib.as_array(out_points, (n_points.value, 2)).copy()
    if n_curves.value > 0:
      result_curves = np.ctypeslib.as_array(out_curves, (n_curves.value,)).copy()
  finally:
    _lib.libcfree(ctypes.cast(out_points, ctypes.c_void_p))
    _lib.libcfree(ctypes.cast(out_curves, ctypes.c_void_p))
  return CoarseningResult(result_points, result_curves, bool(converged.value))


def _delete_polymesh(address: int) -> None:
  # Looked up when called, not when the finalizer is made (the tests watch it).
  _lib.polymesh_delete(address)


class PolyMesh:
  """A Delaunay triangulation to which points can be added (gmsh's PolyMesh), used for the background mesh that
  carries the mesh size. Its memory, allocated by the library, is given back when the object is collected, or by
  `close`."""

  def __init__(self, bbmin: ArrayLike, bbmax: ArrayLike):
    """
    Args:
      bbmin, bbmax: lower and upper corners of a box that holds all the points to come.
    """
    self._address: int = _lib.polymesh_new(
      np.array(bbmin, dtype=np.float64, order="C"), np.array(bbmax, dtype=np.float64, order="C")
    )
    if not self._address:
      raise MemoryError("Cannot allocate the background triangulation")
    self._n_points = 0
    self._finalizer = weakref.finalize(self, _delete_polymesh, self._address)

  def add_points(self, points: ArrayLike) -> None:
    """Adds points [n, 2]; they are numbered in the order they are added."""
    points = np.array(points, dtype=np.float64, order="C")
    numbers = np.arange(self._n_points, self._n_points + points.shape[0], dtype=np.int32)
    self._n_points += points.shape[0]
    _lib.polymesh_add_points(self._address, points.shape[0], points, numbers)

  def get_triangles(self) -> np.ndarray:
    """The triangles [m, 3], as numbers of the points."""
    triangles = np.empty((_lib.polymesh_n_faces(self._address), 3), np.int32)
    _lib.polymesh_faces(self._address, triangles)
    return triangles

  def close(self) -> None:
    """Gives the memory of the triangulation back now (the object cannot be used afterwards)."""
    self._finalizer()
