"""
Advanced Gmsh / Quad Mesh
=========================
Seamsh uses gmsh internally. The whole gmsh python api is exposed through the
seamsh.gmsh.gmsh module (the repetition is not a typo). This example is a
variation on the previous one using the gmsh api to generate a quadrangular
instead of triangular mesh.
"""

# %%
# We start as usual.

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

domain_srs = osr.SpatialReference()
domain_srs.ImportFromProj4("+proj=utm +ellps=WGS84 +zone=31")
domain = seamsh.geometry.Domain(domain_srs)
domain.add_boundary_curves_shp("data/data_no_duplicate.shp", "physical", CurveType.POLYLINE)

dist_coast = seamsh.field.Distance(domain, 100, ["coast", "island"])
dist_porquerolles = seamsh.field.Distance(domain, 20, ["porquerolles"])


def mesh_size(x, projection):
  s_coast = np.clip((dist_coast(x, projection) - 400) * 0.5, 200, 5000)
  s_porq = np.clip((dist_porquerolles(x, projection) - 200) * 0.5, 50, 5000)
  s_dist = np.minimum(s_coast, s_porq)
  return s_dist


coarse = seamsh.geometry.coarsen_boundaries(domain, (8e5, 4.68e6), domain_srs, mesh_size)

# %%
# The gmsh options are set. The full list of gmsh mesh options can be found in
# `gmsh's documentation <https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options-list>`_.

gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
gmsh.option.setNumber("Mesh.Algorithm", 8)

# %%
# Eventually the mesh is generated.

seamsh.gmsh.mesh(coarse, mesh_size)
gmsh.write("quad_mesh.msh")

# %%
# The mesh, drawn from the nodes and elements of the gmsh model (see the first
# example). Quadrangles are gmsh elements of type 3; the few triangles that the
# recombination leaves are of type 2.

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

node_tags, xyz, _ = gmsh.model.mesh.getNodes()
xyz = xyz.reshape(-1, 3)
index = np.zeros(node_tags.max() + 1, int)
index[node_tags] = np.arange(node_tags.size)
polygons = []
for element_type, n_nodes in ((2, 3), (3, 4)):
  _, element_nodes = gmsh.model.mesh.getElementsByType(element_type)
  polygons.extend(xyz[index[element_nodes.reshape(-1, n_nodes)], :2] / 1e3)

fig, ax = plt.subplots(figsize=(7, 7))
ax.add_collection(PolyCollection(polygons, facecolor="none", edgecolor="k", linewidth=0.1))
ax.autoscale()
ax.set_aspect("equal")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
plt.show()

# %%
# gmsh is finalized when it is no longer needed.

gmsh.finalize()
