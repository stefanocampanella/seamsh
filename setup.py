# seamsh - Copyright (C) <2010-2020>
# <Universite catholique de Louvain (UCL), Belgium
#
# List of the contributors to the development of seamsh: see AUTHORS file.
# Description and complete License: see LICENSE file.
#
# This program (seamsh) is free software:
# you can redistribute it and/or modify it under the terms of the GNU Lesser General
# Public License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program (see COPYING file).  If not,
# see <http://www.gnu.org/licenses/>.

"""Builds the C library and sets the version; the rest of the metadata is in pyproject.toml."""

import os

from setuptools import Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel  # setuptools >= 70 (we require 77)
from setuptools.command.build_ext import build_ext as _build_ext

# Keep in step with CHANGELOG.rst (doc/conf.py reads it from here). The local label "+fork" tells
# this fork's versions from upstream's releases on PyPI, which have the same numbers but other code
# (upstream's 0.5 is not this 0.5.0).
version = "0.5.0+fork"

# Not a Python extension module: a plain shared library, loaded with ctypes in seamsh/_libseamsh.py.
lib = Extension(
  "seamsh.libseamsh",
  sources=["seamshlib/seamsh.c", "seamshlib/polymesh.c", "seamshlib/robustPredicates.c"],
)


class bdist_wheel(_bdist_wheel):
  """Tags the wheel as independent of the Python version (the library does not use the C API)."""

  def get_tag(self):
    otag = _bdist_wheel.get_tag(self)
    return ("py3", "none", otag[2])


class build_ext(_build_ext):
  """Names the library `libseamsh.so` on every platform: the name seamsh/_libseamsh.py loads."""

  def get_ext_filename(self, fullname):
    return os.path.join(*fullname.split(".")) + ".so"


setup(
  version=version,
  ext_modules=[lib],
  cmdclass={"bdist_wheel": bdist_wheel, "build_ext": build_ext},
)
