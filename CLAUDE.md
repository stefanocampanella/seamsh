# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

seamsh is a Python library wrapping gmsh, GDAL/OGR/OSR and scipy to generate unstructured (mainly coastal-ocean)
meshes from GIS data. This repository is the fork `stefanocampanella/seamsh` (work goes into `develop`); upstream is
Jonathan Lambrechts' GitLab project (`master`, CHANGELOG up to 0.4.18).

## Commands

```bash
# Build the C library in place (writes seamsh/libseamsh.so, which _tools.py loads with ctypes at import time)
python setup.py build_ext --inplace

python -m pytest                        # all tests (imports seamsh from the source tree)
python -m pytest tests/test_mesh.py -k island   # one file / one test
pre-commit run --all-files              # ruff (check, format), codespell, clang-format on seamshlib/seamsh.c

# Docs: examples/*.py are converted to rst, then Sphinx
cd doc && python gen_examples.py && make html
```

- Runtime dependencies: `numpy`, `scipy`, `gdal>=3.1`, `gmsh`, `tqdm`; optional `shapely` (`field.Inpoly`) and `xarray`
  (`convert_to_ugrid_mesh_file`). GDAL has no pip wheels: use conda-forge (as `.github/workflows/ci.yml` does) or a
  container.
- On JUPITER the login-node Python has none of these. Run everything in moby-dick's image, with the worktree first on
  the path: `apptainer exec --env PYTHONPATH=$PWD /e/project1/e-dev-2026d09-119/graphcast/moby-dick.sif <command>`
  (it has gmsh, GDAL, pytest, ruff, pre-commit and gcc).
- `import seamsh` fails if `seamsh/libseamsh.so` has not been built; rebuild after every change to `seamshlib/`.
- Metadata is in `pyproject.toml`; `setup.py` only defines the C library and the version (keep it in step with
  `CHANGELOG.rst`; upstream's GitLab CI overrides it from tags `v-*` / `w-*`). `.gitlab-ci.yml` is upstream's CI
  and does not run on GitHub.

If the untracked `refactoring-kit/` directory is present, it holds the validation against reference meshes (see
below).

## Architecture

Python package `seamsh/` (~1,400 lines) plus a small C library `seamshlib/` compiled into `seamsh/libseamsh.so`.

- `_tools.py` is the shared namespace: it imports numpy, GDAL (`gdal`, `ogr`, `osr`), scipy, ctypes, etc., and the
  other modules reach them as `_tools.np`, `_tools.osr`, `_tools.c`, ... rather than importing them directly. It also
  holds the ctypes handle `lib`, `np2c` (numpy to `void*`, keeping the array alive), `PolyMesh` (wrapper of
  `polymesh.c`) and `project_points` (reprojection with a cache of `osr.CoordinateTransformation`).
- `geometry.py`: `Domain` collects boundary curves, interior curves and interior points (`_Curve`, `_Point`), each
  curve keeping its own projection and `CurveType`. `Domain._build_topology()` merges coincident points, splits
  curves at shared points, chains boundary curves into closed loops and puts the loop with the largest bounding box
  first. `coarsen_boundaries` builds a new, coarser `Domain`.
- `gmsh.py`: `mesh()` builds the gmsh geometry from the topology and meshes it; also `reproject`, `convert_to_gis`,
  `convert_to_ugrid_mesh_file`, `merge_meshes`.
- `field.py`: mesh-size helpers `Distance`, `Raster`, `Inpoly`.
- `seamshlib/seamsh.c`: boundary coarsening (`gen_boundaries_from_points`). `polymesh.c` (from gmsh) and
  `robustPredicates.c` (Shewchuk) are third-party: do not reformat or refactor them.

Things that need several files to see:

- **Size callbacks** are the central contract: `f(x[n, 2], projection) -> [n]`, called with points in the domain's
  projection. `field.Distance`/`Raster`/`Inpoly` are such callables; users compose them in plain Python.
- **gmsh global state.** seamsh never calls `gmsh.initialize()`; the caller does. `mesh()` works on the *current*
  gmsh model and does not write a file (the caller calls `gmsh.write`). `_curve_sample` (used by `coarsen_boundaries`
  and `field.Distance`) adds and removes a temporary gmsh model per curve, so the current model changes under the
  caller. `mesh()` stores the output projection as the model attribute `Projection` (`["WKT", wkt]`), which
  `reproject` and `convert_to_gis` read back.
- **Circular import.** `geometry.py` imports `_curve_sample` from `gmsh.py` in the middle of the file (after `Domain`
  is defined) because `gmsh.py` imports `geometry`.
- **Coarsening pipeline** (`geometry.coarsen_boundaries` then C): sample every boundary curve at half the size, merge
  points, jitter them (`np.random.seed(0)`, global state), scipy Delaunay, then in C: flood fill from the triangle
  containing the single seed `x0`, crossing only edges longer than the local size (so anything behind a strait
  narrower than the coarsening size is dropped), edge collapses, boundary extraction into tagged closed loops moved
  inwards by size/50. Arrays allocated in C come back through `double**`/`int**` and are freed with `lib.libcfree`.
- **Meshing** (`gmsh.mesh`): one `addPlaneSurface` for all loops (first loop is the exterior, all others holes, so
  disconnected regions are not supported). With `smoothness > 0` (default 0.3) `_mesh_bgrid` builds the size field on
  a background triangulation (`PolyMesh`), refined and smoothed, given to gmsh as a `PostView` field (the view `tri`
  stays in the model). With `smoothness <= 0`, `_mesh_successive` does 1D meshing then three adaptation passes.
  Physical groups are named after curve tags; the surface group is `domain`.
- Known breakage on `develop`: `convert_to_gis` and `convert_to_ugrid_mesh_file` call `_tools.log`, which was removed.

## License

Inconsistent: the README and `setup.py` say GPLv3, `COPYING.txt` is the GPLv3 text, file headers say LGPL v3 or
later, `LICENSE.txt` is a placeholder. Keep the existing header on source files; do not relicense.

## Refactoring kit (untracked)

`refactoring-kit/`, when present, is local material for the ongoing refactoring and must not be committed (it holds
large data and Apache-2.0 files from another project). Read `refactoring-kit/PR_GUIDE.md` (goal, working rules,
commit plan, API to keep, done criteria) and `refactoring-kit/NOTES_seamsh.md` (algorithms, measured behaviour, known
bugs with line numbers) before changing code. Validation against reference outputs of `f1ab87d`, run from
`refactoring-kit/` in an environment with seamsh installed:

```bash
# Fast loop (seconds): the 1:110m coastline mesh must be identical to the reference
python validation/validate_seamsh.py mesh --shapefile data/ne_110m_coastline/ne_110m_coastline.shp \
    --coarsen-size 100000 --mesh-size 300000 --reference validation/reference/mesh_110m_300km.msh --require-identical

# The downstream configuration (~20 s)
python validation/validate_seamsh.py mesh --coarsen-size 10000 --mesh-size 120000 \
    --reference validation/reference/mesh_10m_120km.msh --require-identical
```
