# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Disconnected Domains
====================
Boundary coarsening keeps the water that can be reached from a point without
going through a passage narrower than the mesh size. A basin behind a narrow
strait is therefore lost, unless a point is given in that basin too. This
example builds two basins joined by a narrow channel and meshes them as two
separate pieces.
"""

import gmsh
import numpy as np
from osgeo import osr

import seamsh

gmsh.initialize()

# %%
# The domain: two square basins of 40 km, joined by a channel 20 km long and
# 2 km wide. The boundary is one closed polyline with a point every 2 km.

domain_srs = osr.SpatialReference()
domain_srs.ImportFromEPSG(32631)

corners = np.array(
  [
    [0, 0],
    [40, 0],
    [40, 19],
    [60, 19],
    [60, 0],
    [100, 0],
    [100, 40],
    [60, 40],
    [60, 21],
    [40, 21],
    [40, 40],
    [0, 40],
    [0, 0],
  ]
)
points = [corners[0]]
for a, b in zip(corners[:-1], corners[1:]):
  n = int(np.abs(b - a).max() / 2)
  points.extend(a + (b - a) * i / n for i in range(1, n + 1))

domain = seamsh.geometry.Domain(domain_srs)
domain.add_boundary_curve(np.array(points) * 1e3, "coast", domain_srs)


# %%
# A uniform mesh size of 5 km: larger than the width of the channel.


def mesh_size(x, projection):
  return np.full(x.shape[0], 5e3)


# %%
# With one point, in the western basin, the coarsened domain is that basin
# only: the channel and the eastern basin are dropped.

west_only = seamsh.geometry.coarsen_boundaries(domain, (20e3, 20e3), domain_srs, mesh_size)

# %%
# With one point in each basin, both are kept. The channel, which the mesh
# cannot resolve, stays closed: the domain is made of two regions.

both = seamsh.geometry.coarsen_boundaries(domain, [(20e3, 20e3), (80e3, 20e3)], domain_srs, mesh_size)

# %%
# Each region is meshed on its own: the mesh is in two pieces. Note that the
# boundary curves only tell where the shore is, not on which side the water
# is: a point given on land would give a mesh of that land.

seamsh.gmsh.mesh(both, mesh_size)
gmsh.write("disconnected.msh")

# %%
# The mesh, drawn from the nodes and triangles of the gmsh model (see the first
# example), over the original boundary (grey).

import matplotlib.pyplot as plt

node_tags, xyz, _ = gmsh.model.mesh.getNodes()
xyz = xyz.reshape(-1, 3)
index = np.zeros(node_tags.max() + 1, int)
index[node_tags] = np.arange(node_tags.size)
_, triangle_nodes = gmsh.model.mesh.getElementsByType(2)
triangles = index[triangle_nodes.reshape(-1, 3)]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(corners[:, 0], corners[:, 1], color="0.6", linewidth=2)
ax.triplot(xyz[:, 0] / 1e3, xyz[:, 1] / 1e3, triangles, color="k", linewidth=0.5)
ax.set_aspect("equal")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
plt.show()

# %%
# gmsh is finalized when it is no longer needed.

gmsh.finalize()
