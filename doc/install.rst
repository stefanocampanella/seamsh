Installation
============
Dependencies
------------
Seamsh depends on the following Python packages:

- `gmsh <https://gmsh.info>`_ (the Python API; this fork is tested with gmsh 4.11 to 4.15)
- `gdal <https://gdal.org>`_ (3.1 or later)
- `numpy <https://numpy.org>`_, `scipy <https://scipy.org>`_ and `tqdm <https://tqdm.github.io>`_
- optionally `xarray <https://xarray.dev>`_ and `netCDF4 <https://unidata.github.io/netcdf4-python>`_,
  for ``seamsh.gmsh.convert_to_ugrid_mesh_file``

GDAL has no binary packages on PyPI, and gmsh has none for Linux on ARM: the simplest way to get
all of them is a conda-forge environment, which also provides a C compiler:

.. code-block:: bash

   conda create -n seamsh -c conda-forge python=3.12 gdal python-gmsh numpy scipy tqdm c-compiler
   conda activate seamsh

Check that gmsh is installed with the following command, which prints its version.

.. code-block:: bash

   python3 -c "import gmsh; gmsh.initialize(); print(gmsh.option.getString('General.Version'))"

Installation
------------
This fork is not published on PyPI: ``pip install seamsh`` installs upstream's releases, which are
different code (see the README). Install it from its repository instead. The package holds a small
C library, compiled when the package is built (a C compiler is needed):

.. code-block:: bash

   pip3 install git+https://github.com/stefanocampanella/seamsh@develop

or from a clone:

.. code-block:: bash

   git clone https://github.com/stefanocampanella/seamsh
   cd seamsh
   pip3 install .

To work on seamsh itself, build the library in place and run the tests from the source tree:

.. code-block:: bash

   python3 setup.py build_ext --inplace
   python3 -m pytest
