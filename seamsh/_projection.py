# SPDX-FileCopyrightText: 2010-2020 Universite catholique de Louvain (UCL), Belgium
# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# Moved out of upstream's seamsh/_tools.py (LGPL, see LICENSE.txt), and rewritten.

"""Reprojection of points between coordinate systems (osr), with a cache of the transformations."""

from collections import OrderedDict

import numpy as np
from numpy.typing import ArrayLike
from osgeo import osr

_CACHE_SIZE = 64

# Coordinate transformations are costly to build: the last ones used are kept, by the definitions of the two
# projections (not by the objects: equal projections built twice share one entry). Least recently used first.
_by_definition: "OrderedDict[tuple, osr.CoordinateTransformation | None]" = OrderedDict()

# Exporting the definitions (WKT) costs as much as transforming a few points, and the same pair of objects is used
# over and over (curve sampling): the transformation of a pair of objects is also kept by the identity of the
# objects, checked against their axis order, the one setting that is commonly changed on an object in use. An object
# whose definition is replaced (ImportFrom...) after use must not be reused: make a new one. The objects are kept in
# the entry, so that their ids cannot be reused while it exists.
_by_identity: "OrderedDict[tuple[int, int], tuple]" = OrderedDict()


def _definition(projection: osr.SpatialReference) -> tuple:
  """What defines a projection for a transformation: its WKT, and the order of its axes in the data, which is a
  setting of the object that the WKT does not hold."""
  return projection.ExportToWkt(), tuple(projection.GetDataAxisToSRSAxisMapping())


def _remember(cache: OrderedDict, key, value) -> None:
  cache[key] = value
  cache.move_to_end(key)
  if len(cache) > _CACHE_SIZE:
    cache.popitem(last=False)


def transformation(source: osr.SpatialReference, target: osr.SpatialReference) -> osr.CoordinateTransformation | None:
  """The transformation from `source` to `target`, or None if they are the same projection."""
  axes = (tuple(source.GetDataAxisToSRSAxisMapping()), tuple(target.GetDataAxisToSRSAxisMapping()))
  hit = _by_identity.get((id(source), id(target)))
  if hit is not None and hit[0] is source and hit[1] is target and hit[2] == axes:
    _by_identity.move_to_end((id(source), id(target)))
    return hit[3]
  if source.IsSame(target):
    result = None
  else:
    key = (_definition(source), _definition(target))
    if key in _by_definition:
      result = _by_definition[key]
      _by_definition.move_to_end(key)
    else:
      result = osr.CoordinateTransformation(source, target)
      _remember(_by_definition, key, result)
  _remember(_by_identity, (id(source), id(target)), (source, target, axes, result))
  return result


def project_points(points: ArrayLike, source: osr.SpatialReference, target: osr.SpatialReference) -> np.ndarray:
  """`points` [n, >= axes of source], given in projection `source`, in projection `target` [n, axes of target].
  When both projections are the same, only the extra columns are dropped."""
  points = np.asarray(points)[:, : source.GetAxesCount()]
  transform = transformation(source, target)
  if transform is not None:
    points = np.asarray(transform.TransformPoints(points))[:, : target.GetAxesCount()]
  return points
