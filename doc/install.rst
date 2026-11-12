Installation
============
Dependencies
------------
Seamsh depends on the following python3 packages:

- `gmsh-dev <https://pypi.org/project/gmsh-dev>`_
- `gdal <https://pypi.org/project/gdal>`_
- `scipy <https://pypi.org/project/scipy>`_

Depending on your platform, gdal and scipy can probably be installed directly from your package manager, otherwise pip can be used (follow the links above for installation instructions).

To install gmsh-dev, follow the instructions on the gmsh-dev_ pypi page. Do not forget, to uninstall possible previous gmsh installations. Check that gmsh-dev is correctly installed with the following command.

.. code-block:: bash

   python3 -c "import gmsh;gmsh.initialize();print(gmsh.option.getString('General.Version'))"

The output should be a version number >= 3.6.1.
On some platforms, setting the python path manually after installation is necessary, e.g. :

.. code-block:: bash

   export "PYTONPATH=$HOME/.local/lib/python3.8/site-packages/gmsh-git-Linux64-sdk/lib/:$PYTHONPATH"


Installation with pip
---------------------
Pre-compiled seamsh libraries are provided for 64bit linux, windows and OSX.

.. code-block:: bash 

   pip3 install seamsh

Installation from sources
-------------------------
The package holds a small C library, compiled when the package is built (a C compiler is needed):

.. code-block:: bash

   git clone https://git.immc.ucl.ac.be/jlambrechts/seamsh
   cd seamsh
   pip3 install .

To work on seamsh itself, build the library in place and run the tests from the source tree:

.. code-block:: bash

   python3 setup.py build_ext --inplace
   python3 -m pytest
