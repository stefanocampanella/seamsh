Changes
=======

The versions labelled ``+fork`` are those of the fork https://github.com/stefanocampanella/seamsh,
which branched from upstream's 0.4.18 (https://git.immc.ucl.ac.be/jlambrechts/seamsh). Upstream's
own releases since then (0.4.19 and later, on PyPI) are not listed here: they are different code,
and upstream's 0.5 is not this 0.5.0+fork.

seamsh 0.5.0+fork
-----------------
* the version carries the local label ``+fork``, the project's links (README, package metadata,
  documentation) point to the fork, and the installation instructions install it from GitHub;
  upstream's GitLab CI and Docker images, which released to upstream's PyPI project, are removed
* convert_to_ugrid_mesh_file: the global attribute ``source`` of the file is
  "seamsh (https://github.com/stefanocampanella/seamsh)", and the placeholders ``institution``
  ("Institution Name") and ``references`` ("REF") are no longer written
* disconnected domains: coarsen_boundaries takes several points (x0 can be a sequence of points) and keeps the basin around each one, e.g. a sea behind a strait narrower than the mesh size; mesh builds one surface per region of the domain (regions are found from the nesting of the boundary loops, so a lake on an island is meshed too) and embeds interior curves and points in the region that holds them
* **meshes change**: the points of a coarsened boundary are moved towards the domain along the normals of the two segments that meet on them (the two segments before the point were used); coarsened domains have the same number of curves and points, moved by at most two twenty-fifths of the size, and their meshes have the same quality but other nodes
* the smoothing of the mesh size stops on a tolerance relative to the sizes: meshes no longer depend on the unit of length (they are unchanged for sizes above 1000 units, e.g. in metres)
* mesh leaves the gmsh options as they were, except the file version
* mesh(quad=True) raises an error with builds of gmsh in which the quadrangle algorithm crashes (those without ANN, e.g. Ubuntu 24.04's gmsh 4.12.1)
* mesh never loses a region of the domain: one smaller than the mesh size (e.g. a small sea given its own point in coarsen_boundaries, in a coarse mesh) has its boundary meshed with three segments and is one triangle; it used to be missing from the mesh
* a shapefile or geopackage that cannot be opened is reported by name
* more than two boundary curves meeting at a point, and triangulations with more than two triangles on an edge, are errors
* a warning is logged when the coarsening stops before the boundary is at the requested size
* field.Inpoly no longer needs shapely
* convert_to_ugrid_mesh_file writes the connectivity as node indices and the units of the coordinates of the projection
* seamsh.show_progress(False) turns the progress bars off
* coarsen_boundaries no longer hangs on boundaries made of straight lines, keeps both tags of points shared by two curves and leaves numpy's global random state alone
* errors of the C library are raised as Python exceptions instead of ending the process
* outlines of islands smaller than the mesh size are no longer written to the mesh file (they were boundary elements on nodes that belong to no triangle)
* the view and field used for the mesh size are removed from gmsh after meshing
* a mesh size that is zero, negative or NaN raises an error instead of looping forever, also where it is asked outside the domain: with smoothness > 0 the size is evaluated on the whole bounding box of the domain, where it must be positive too (e.g. a size from bathymetry on land); the error names a point
* a domain can be meshed several times (e.g. at several resolutions): the second call failed with "Invalid topology" where two loops touch inside their curves
* the background triangulation of the mesh size is freed after meshing (it leaked, about 11 MB per 20,000 points)
* the regions of interior curves and points are found at once, and not at all for a domain of one region; point-in-polygon tests use bounded memory whatever the size of the loops; transformations between the same projection objects are found without exporting their definitions (a coarsening and 300 km mesh of the 1:10m coastline: 7.2 s to 4.6 s, profiled)
* convert_to_ugrid_mesh_file uses the projection stored in the mesh file when projection is None
* the mesh size is scaled correctly along curves given in a projected coordinate system (in metres): curves are sampled differently there by coarsen_boundaries and field.Distance (e.g. a Distance field at 2 km sampling in UTM: 618 points instead of 622 to 627, every new one an old one; distances change by at most 0.8 km, at 1 % of points); meshes made with smoothness > 0 are unaffected
* convert_to_gis and convert_to_ugrid_mesh_file work again
* license statements made consistent (LICENSE.txt): seamsh's own files are LGPL v3 or later as before, the files from gmsh in seamshlib are GPL, so seamsh as a whole is distributed under the GPL v3 or later; the text of the LGPL and Gmsh's license (as of Gmsh 4.8.4) are included; files added in 0.5.0 are GPL v3 or later; the license is declared as an SPDX expression (pyproject.toml, setuptools >= 77)
* the source package holds the C headers (it could not be compiled) and no longer a library built in the source tree; it also holds the tests and examples (MANIFEST.in)
* new example: enclosed seas and a hierarchy of meshes (several points, two resolutions)
* tests (pytest), pyproject.toml, examples and documentation updated to the current API (mesh works on the current gmsh model, Raster takes a gdal dataset)

seamsh 0.4.18
-------------
* adjust epsilons to domain size

seamsh 0.4.17
-------------
* fix

seamsh 0.4.16
-------------
* np.frombuffer for newer numpy versions

seamsh 0.4.15
-------------
* set random seed for reproducibility
* set layer name to "mesh" when exporting to gpkg/shapefile instead of the file name to avoid incompatible special characters

seamsh 0.4.14
-------------
* fig bug with newer versions of numpy (integer casted to int32 instead of int64)

seamsh 0.4.13
-------------
* fix bug for 2d coordinates reprojection introduced in 0.4.12

seamsh 0.4.12
-------------
* Add stereographic projection example
* Distance field can use another projection than the projection of the domain
* Allows for cartesian projection (i.e. no projection)

seamsh 0.4.11
-------------
* gmsh.merge_meshes : topological reconnection instead of coordinate based

seamsh 0.4.10
-------------
* add gmsh.merge_meshes
* add transifinite curves (physical lines with prescribed number of elements)
* avoid gmsh initialization error message

seamsh 0.4.9
------------
* add transifinite curves (physical lines with prescribed number of elements)

seamsh 0.4.8
------------
* numpy 1.24 compatibility (np.bool -> bool)

seamsh 0.4.7
------------
* write WKT projection as gmsh model attribute
* default msh output version set to 4.1

seamsh 0.4.6
------------
* License files in source package

seamsh 0.4.5
------------
* Inpoly field (requires shapely)

seamsh 0.4.4
------------
* fix unrefine when no identical points
* ugrid conversion (requires xarray)
* fix domain physical name

seamsh 0.4.3
------------
* add output_srs parameter to gmsh.mesh
* add gmsh.reproject function

seamsh 0.4.2
------------
* do not crash on features without geometry in shapefiles

seamsh 0.4.1
------------
* do not call gmsh.initialize() if already initialized
* use a temporary gmsh model for meshing

seamsh 0.4.0
------------

* precompute and smooth mesh size field on a quadtree-like mesh
* build library directly from setup.py, no cmake anymore
* discretize curves in their own projection, not in the model projection
* handle intersection of more than 2 interior curves on the same point
* source distribution on pypi
