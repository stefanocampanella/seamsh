# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: GPL-3.0-or-later

"""Quality of the triangles of the current gmsh model, as moby-dick's `graph-statistics` measures it.

gamma is twice the radius of the inscribed circle over the radius of the circumscribed circle: 1 for an equilateral
triangle, towards 0 for a flat one. It is gmsh's "gamma" quality and moby-dick's `radius_ratio`. The targets are
those agreed for moby-dick's meshes (12 Nov 2026), for every level of a hierarchy from 25 to 400 km.
"""

import dataclasses

import gmsh
import numpy as np

# Targets: worst gamma, largest share of triangles with gamma below 0.8, smallest angle (degrees), and smallest
# share of nodes with 6 neighbours (meshes of 25 to 100 km).
MIN_GAMMA = 0.4
MAX_POOR_SHARE = 0.01
MIN_ANGLE = 15.0
MIN_DEGREE_6_SHARE = 0.8


@dataclasses.dataclass
class Quality:
  nodes: int
  triangles: int
  gamma: np.ndarray
  """gamma of each triangle."""
  min_angle: np.ndarray
  """Smallest angle of each triangle (degrees)."""
  max_angle: np.ndarray
  degree: np.ndarray
  """Number of neighbours of each node of a triangle."""
  on_boundary: np.ndarray
  """Whether each triangle has a vertex on the boundary (an edge of one triangle only)."""

  @property
  def worst_gamma(self) -> float:
    return float(self.gamma.min())

  @property
  def poor_share(self) -> float:
    return float((self.gamma < 0.8).mean())

  @property
  def smallest_angle(self) -> float:
    return float(self.min_angle.min())

  @property
  def degree_6_share(self) -> float:
    return float((self.degree == 6).mean())

  def failures(self, degree: bool = True) -> list[str]:
    """The targets that are missed (degree: also the share of nodes with 6 neighbours)."""
    missed = []
    if self.worst_gamma < MIN_GAMMA:
      missed.append(f"worst gamma {self.worst_gamma:.3f} < {MIN_GAMMA}")
    if self.poor_share >= MAX_POOR_SHARE:
      missed.append(f"{self.poor_share:.2%} of triangles with gamma < 0.8")
    if self.smallest_angle < MIN_ANGLE:
      missed.append(f"smallest angle {self.smallest_angle:.1f} < {MIN_ANGLE}")
    if degree and self.degree_6_share < MIN_DEGREE_6_SHARE:
      missed.append(f"{self.degree_6_share:.1%} of nodes with 6 neighbours")
    return missed

  def summary(self) -> str:
    poor = self.gamma < 0.8
    at_coast = (poor & self.on_boundary).sum() / max(poor.sum(), 1)
    return (
      f"{self.nodes} nodes, {self.triangles} triangles; worst gamma {self.worst_gamma:.3f}, "
      f"< 0.8 {self.poor_share:.2%} ({at_coast:.0%} at the boundary), < 0.5 {(self.gamma < 0.5).mean():.3%}; "
      f"angles {self.smallest_angle:.1f}-{self.max_angle.max():.1f}; degree 6 {self.degree_6_share:.1%}"
    )


def triangle_quality(x: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """gamma, smallest and largest angle (degrees) of triangles [m, 3] of points x [n, 2 or 3]."""
  a, b, c = (x[triangles[:, i]] for i in range(3))
  la = np.linalg.norm(b - c, axis=1)
  lb = np.linalg.norm(c - a, axis=1)
  lc = np.linalg.norm(a - b, axis=1)
  s = (la + lb + lc) / 2
  area = np.sqrt(np.maximum(s * (s - la) * (s - lb) * (s - lc), 0))
  with np.errstate(divide="ignore", invalid="ignore"):
    inradius = area / s
    circumradius = la * lb * lc / (4 * area)
    gamma = np.nan_to_num(2 * inradius / circumradius)
    cosines = np.stack(
      [
        (lb**2 + lc**2 - la**2) / (2 * lb * lc),
        (lc**2 + la**2 - lb**2) / (2 * lc * la),
        (la**2 + lb**2 - lc**2) / (2 * la * lb),
      ],
      axis=1,
    )
  angles = np.degrees(np.arccos(np.clip(np.nan_to_num(cosines, nan=1.0), -1, 1)))
  return gamma, angles.min(axis=1), angles.max(axis=1)


def mesh_quality() -> Quality:
  """The quality of the triangles of gmsh's current model."""
  tags, coordinates, _ = gmsh.model.mesh.getNodes()
  index = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
  index[tags.astype(np.int64)] = np.arange(len(tags))
  _, nodes = gmsh.model.mesh.getElementsByType(2)
  triangles = index[nodes.astype(np.int64)].reshape(-1, 3)
  x = coordinates.reshape(-1, 3)
  gamma, min_angle, max_angle = triangle_quality(x, triangles)
  edges = np.sort(np.r_[triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=1)
  unique, counts = np.unique(edges, axis=0, return_counts=True)
  used = np.unique(triangles)
  degree = np.bincount(unique.ravel(), minlength=len(x))[used]
  boundary = np.zeros(len(x), bool)
  boundary[unique[counts == 1].ravel()] = True
  return Quality(
    nodes=len(used),
    triangles=len(triangles),
    gamma=gamma,
    min_angle=min_angle,
    max_angle=max_angle,
    degree=degree,
    on_boundary=boundary[triangles].any(axis=1),
  )
