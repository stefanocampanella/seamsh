# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

seamsh is a Python library wrapping gmsh, GDAL/OGR/OSR and scipy to generate unstructured (mainly coastal-ocean)
meshes from GIS data. This repository is the fork `stefanocampanella/seamsh` (work goes into `develop`); upstream is
Jonathan Lambrechts' GitLab project (`master`). The method is described in Lambrechts et al., "Multiscale mesh
generation on the sphere", Ocean Dynamics 58 (2008): stereographic plane, B-spline coastlines, size fields. The
boundary coarsening in `seamshlib/seamsh.c` is later than the article and is documented in the code.

## Commands

```bash
# Build the C library in place (writes seamsh/libseamsh.so, which _tools.py loads with ctypes at import time)
python setup.py build_ext --inplace

python -m pytest                                # all tests (imports seamsh from the source tree), ~10 s
python -m pytest tests/test_mesh.py -k island   # one file / one test
pre-commit run --all-files                      # ruff (check, format), codespell, clang-format on seamshlib/seamsh.c

# Docs: examples/*.py are converted to rst, then Sphinx
cd doc && python gen_examples.py && make html
```

- Runtime dependencies: `numpy`, `scipy`, `gdal>=3.1`, `gmsh`, `tqdm`; optional `xarray` and `netcdf4`
  (`convert_to_ugrid_mesh_file`). GDAL has no pip wheels (nor gmsh on Linux aarch64): use conda-forge or a container.
- GitHub CI (`.github/workflows/ci.yml`) runs on `ubuntu-latest` in a conda-forge environment made with micromamba
  (Python 3.12 and the latest GDAL, gmsh, numpy, scipy), builds the library, then runs pytest and the hooks.
- On JUPITER the login-node Python has none of these. Run everything in moby-dick's image, with the worktree first on
  the path: `apptainer exec --env PYTHONPATH=$PWD /e/project1/e-dev-2026d09-119/graphcast/moby-dick.sif <command>`
  (it has gmsh 4.12.1 from Ubuntu, GDAL, pytest, ruff, pre-commit and gcc, but no netCDF4: the UGRID test is
  skipped there).
- `import seamsh` fails if `seamsh/libseamsh.so` has not been built; rebuild after every change to `seamshlib/`.
- Metadata is in `pyproject.toml`, and `MANIFEST.in` decides what goes into the source package (setuptools does
  not find the C headers on its own: add new `seamshlib` files there). `setup.py` only defines the C library and the version (keep it in step with
  `CHANGELOG.rst`; upstream's GitLab CI overrides it from tags `v-*` / `w-*`). `.gitlab-ci.yml` is upstream's CI
  and does not run on GitHub.

## Tests

`tests/` is pytest, with no network. `tests/domains.py` holds the synthetic domains (square with an island, two
basins joined by a channel, two separate squares) and the mesh measurements; `conftest.py` initializes gmsh once and
gives each test a fresh model (`model` fixture). `test_natural_earth.py` runs the whole chain on the Natural Earth
1:110m coastline (`tests/data`). Mesh counts are asserted within ranges, because they depend on the gmsh version.
Every test has a docstring saying what it checks and why; every bug fix comes with a test that failed before.

gmsh's quadrangle algorithm (`Mesh.Algorithm` 8) ends the process with a segmentation fault in builds of gmsh
without ANN, such as Ubuntu 24.04's gmsh 4.12.1 (the one in moby-dick's image), even on a plain square; the
conda-forge builds of gmsh 4.11 to 4.15 work. `mesh(quad=True)` therefore raises an error on such builds
(`gmsh._gmsh_has_quad_algorithm`), and `test_quad_mesh` checks whichever applies.

## Architecture

Python package `seamsh/` plus a small C library `seamshlib/` compiled into `seamsh/libseamsh.so`.

- `geometry.py`: `Domain` collects boundary curves, interior curves and interior points (`_Curve`, `_Point`), each
  curve keeping its own projection and `CurveType`. `Domain._build_topology()` merges coincident points, cuts curves
  where they touch, chains boundary curves into closed loops (`_chain_into_loops`) and sorts the loops into regions
  by nesting (`_nest_loops`: even depth = outer boundary of a region, odd depth = hole). `coarsen_boundaries` builds
  a new, coarser `Domain`.
- `gmsh.py`: `mesh()` builds the gmsh geometry (one plane surface per region) and meshes it; also `reproject`,
  `convert_to_gis`, `convert_to_ugrid_mesh_file`, `merge_meshes`.
- `field.py`: mesh-size helpers `Distance`, `Raster`, `Inpoly`.
- `_tools.py`: `project_points` (reprojection, with a cache of transformations) and the ctypes binding
  (`gen_boundaries`, `PolyMesh`): argument types are declared there, once.
- `seamshlib/seamsh.c`: boundary coarsening (`gen_boundaries_from_points`); the file starts with a description of
  the algorithm and the data layout. `polymesh.c` (from gmsh), `robustPredicates.c` (Shewchuk), `qsort.h` and
  `vector.h` are third-party: do not reformat or refactor them.

Things that need several files to see:

- **Size callbacks** are the central contract: `f(x[n, 2], projection) -> [n]`, positive, called with points in the
  domain's projection, including points of its bounding box outside the domain (background grid of `_mesh_bgrid`).
  A `Domain` can be meshed several times: `_build_topology` starts again from the curves as given. `field.Distance`/`Raster`/`Inpoly` are such callables; users compose them in plain Python.
- **gmsh global state.** seamsh never calls `gmsh.initialize()`; the caller does. `mesh()` works on the *current*
  gmsh model and does not write a file (the caller calls `gmsh.write`). Curve sampling (`gmsh._curve_sample`, used by
  `coarsen_boundaries` and `field.Distance`) builds each curve in a temporary gmsh model and restores the caller's.
  `mesh()` stores the output projection as the model attribute `Projection` (`["WKT", wkt]`), which `reproject` and
  `convert_to_gis` read back. gmsh options set by `mesh()` stay set.
- **Circular import.** `geometry.py` imports `_curve_sample` from `gmsh.py` after `Domain` is defined, because
  `gmsh.py` imports `geometry`.
- **Coarsening** (`geometry.coarsen_boundaries` then C): sample every boundary curve at half the size, merge points
  and build their tag lists (`_point_tag_lists`), jitter (seeded, local generator), scipy Delaunay, then in C: flood
  fill from the triangles holding the seed points `x0`, crossing only edges longer than the local size (anything
  behind a strait narrower than the size is dropped unless it has its own seed), edge collapses, boundary extraction
  into tagged closed loops moved inwards by size/50. The C code reports errors by return code (raised as
  `RuntimeError`), never exits, and orients triangles from their neighbours (per-triangle orientation breaks on the
  slivers that straight boundaries produce).
- **Meshing** (`gmsh.mesh`): with `smoothness > 0` (default 0.3) `_mesh_bgrid` gives gmsh the size on a background
  triangulation (`PolyMesh`), refined and smoothed; with `smoothness <= 0`, `_mesh_successive` does 1D meshing then
  three adaptation passes. Physical groups are named after curve tags; the surfaces are in the group `domain`.
  Curves with no node in a surface element (outlines of islands smaller than the mesh) are taken out of their
  physical group so they are not written: gmsh refuses to delete them. A *region* smaller than the mesh size is
  never dropped (`_keep_small_regions`): its outer boundary gets three segments (transfinite curves), i.e. one
  triangle, so that meshes of the same domain at several resolutions cover the same basins.
- **Outputs are meant to be reproducible.** Downstream meshes are compared with references to within rounding:
  anything that changes sampled points, the jitter, the collapses, the inward shift or the gmsh call order changes
  meshes (same quality, other nodes). Such a change must be deliberate, and reported with before/after numbers.

## License

`LICENSE.txt` is the reference. Upstream's files are LGPL v3 or later (copyright UCLouvain; headers kept as they
are, text in `COPYING.LESSER.txt`). `seamshlib/polymesh.c` (a C port of gmsh's PolyMesh) and `robustPredicates.c`/`.h`
(Shewchuk's predicates as modified in gmsh) come from gmsh, GPL v2 or later (`seamshlib/LICENSE.gmsh.txt`); so the
compiled library, and seamsh as a whole, is GPL v3 or later (`COPYING.txt`, README, `project.license` in
`pyproject.toml`). **Files added by the fork are GPL-3.0-or-later**: start each new file that can hold a comment with

    # SPDX-FileCopyrightText: <year> <author>
    # SPDX-License-Identifier: GPL-3.0-or-later

and list the others in `LICENSE.txt`. Keep the existing header on upstream's files; do not relicense them. Upstream
has not been asked which license it intends.

## Refactoring kit (untracked)

`refactoring-kit/`, when present, is local material and must not be committed (large data, Apache-2.0 files from
another project). `PR_GUIDE.md` and `NOTES_seamsh.md` give the context of the refactoring. It holds reference
outputs: `validation/reference/` made with `f1ab87d`, and `validation/reference_0.5.0/` made with version 0.5.0,
whose meshes differ from the former (the boundary points are shifted along other normals). After a change that
could affect outputs, from `refactoring-kit/`:

```bash
# Seconds: the 1:110m coastline mesh must be identical to the reference
python validation/validate_seamsh.py mesh --shapefile data/ne_110m_coastline/ne_110m_coastline.shp \
    --coarsen-size 100000 --mesh-size 300000 --reference validation/reference_0.5.0/mesh_110m_300km.msh \
    --require-identical

# ~20 s: the downstream 120 km configuration
python validation/validate_seamsh.py mesh --coarsen-size 10000 --mesh-size 120000 \
    --reference validation/reference_0.5.0/mesh_10m_120km.msh --require-identical

# Several seeds: every "ocean" probe inside, every "land" probe outside
python validation/validate_seamsh.py coarsen --coarsen-size 10000 --seed 43.5,34 --seed 46,36.5 --strict-probes
```
