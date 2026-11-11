Changes
=======
seamsh 0.5.0
------------
* coarsen_boundaries no longer hangs on boundaries made of straight lines, keeps both tags of points shared by two curves and leaves numpy's global random state alone
* errors of the C library are raised as Python exceptions instead of ending the process
* outlines of islands smaller than the mesh size are no longer written to the mesh file (they were boundary elements on nodes that belong to no triangle)
* the view and field used for the mesh size are removed from gmsh after meshing
* a mesh size that is zero, negative or NaN raises an error instead of looping forever
* the mesh size is scaled correctly along curves given in a projected coordinate system (in metres)
* convert_to_gis and convert_to_ugrid_mesh_file work again
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
