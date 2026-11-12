# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: GPL-3.0-or-later

# %%
# Disconnected Domains
# ====================
# Boundary coarsening keeps the water that can be reached from a point without
# going through a passage narrower than the mesh size. A basin behind a narrow
# strait is therefore lost, unless a point is given in that basin too. This
# example builds two basins joined by a narrow channel and meshes them as two
# separate pieces.

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

both = seamsh.geometry.coarsen_boundaries(
    domain, [(20e3, 20e3), (80e3, 20e3)], domain_srs, mesh_size
)

# %%
# Each region is meshed on its own: the mesh is in two pieces. Note that the
# boundary curves only tell where the shore is, not on which side the water
# is: a point given on land would give a mesh of that land.

seamsh.gmsh.mesh(both, mesh_size)
gmsh.write("disconnected.msh")
