"""
Boundary Coarsening
===================
This example revisits the previous example (GIS Data) with an extra step to
reduce the resolution of the input coastlines.

This approach has two main advantages:

- Thanks to the fitting of the input data resolution to the final mesh
  element size, the quality of the elements near the boundary is improved.
- The input curves do not have to form topologically valid closed loops.
  The only requirement is that the external boundary should be closed
  but the lines defining the boundaries can intersect each others.

But:

- Interior curves are not handled, they can be present as long as they
  do not intersect the domain boundaries (external and islands) but they
  won't be coarsened.
"""

# %%
# This example starts as the previous one.

import os
import tarfile
import urllib.request

if not os.path.isdir("data"):
  url = "https://nextcloud.cism.ucl.ac.be/s/e7daeWsMpormjpz/download/data-test-1.tar.gz"
  urllib.request.urlretrieve(url, "data-test-1.tar.gz")
  f = tarfile.open("data-test-1.tar.gz", "r:*")
  f.extractall()

import gmsh
import numpy as np
from osgeo import osr

import seamsh
import seamsh.geometry
from seamsh.geometry import CurveType

# %%
# seamsh drives gmsh, which has to be initialized first.

gmsh.initialize()

lonlat = osr.SpatialReference()
lonlat.ImportFromProj4("+proj=latlong +ellps=WGS84 +unit=degrees")
utm32 = osr.SpatialReference()
utm32.ImportFromProj4("+proj=utm +ellps=WGS84 +zone=32")
utm31 = osr.SpatialReference()
utm31.ImportFromProj4("+proj=utm +ellps=WGS84 +zone=31")

domain = seamsh.geometry.Domain(utm31)
domain.add_boundary_curves_shp("data/data_no_duplicate.shp", "physical", CurveType.POLYLINE)

# %%
# Clip the domain by a circle in UTM (zone32) coordinates

alphas = np.linspace(0, 2 * np.pi, 200)
circle_points = np.array([[272e3, 4763e3]]) + 50e3 * np.column_stack([np.sin(alphas), np.cos(alphas)])
domain.add_boundary_curve(circle_points, "open", utm32)

dist_coast = seamsh.field.Distance(domain, 100, ["coast", "island"])
dist_porquerolles = seamsh.field.Distance(domain, 20, ["porquerolles"])


def mesh_size(x, projection):
  s_coast = np.clip((dist_coast(x, projection) - 400) * 0.5, 200, 5000)
  s_porq = np.clip((dist_porquerolles(x, projection) - 200) * 0.5, 50, 5000)
  s_dist = np.minimum(s_coast, s_porq)
  return s_dist


# %%
# The geometry.coarsen_boundaries function requires in input the coordinates
# of one (any) point which is inside the domain.

coarse = seamsh.geometry.coarsen_boundaries(domain, (6.21, 42.95), lonlat, mesh_size)

# %%
# The resulting coarsened domain can be meshed normally.

seamsh.gmsh.mesh(coarse, mesh_size)
gmsh.write("coarse_boundary_mesh.msh")

# %%
# The mesh, drawn from the nodes and triangles of the gmsh model (see the first
# example), in UTM zone 31 coordinates. The domain is clipped by the circle.

import matplotlib.pyplot as plt

node_tags, xyz, _ = gmsh.model.mesh.getNodes()
xyz = xyz.reshape(-1, 3)
index = np.zeros(node_tags.max() + 1, int)
index[node_tags] = np.arange(node_tags.size)
_, triangle_nodes = gmsh.model.mesh.getElementsByType(2)
triangles = index[triangle_nodes.reshape(-1, 3)]

fig, ax = plt.subplots(figsize=(7, 7))
ax.triplot(xyz[:, 0] / 1e3, xyz[:, 1] / 1e3, triangles, color="k", linewidth=0.1)
ax.set_aspect("equal")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
plt.show()

# %%
# gmsh is finalized when it is no longer needed.

gmsh.finalize()
