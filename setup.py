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
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # setuptools < 70
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

# Keep in step with CHANGELOG.rst. Upstream's GitLab CI releases from tags "v-X.Y.Z" (PyPI) and
# "w-X.Y.Z" (TestPyPI), which override it.
version = "0.4.18"
commit_tag = os.environ.get("CI_COMMIT_TAG")
if commit_tag and (commit_tag.startswith("v-") or commit_tag.startswith("w-")):
    version = commit_tag[2:]

# Not a Python extension module: a plain shared library, loaded with ctypes in seamsh/_tools.py.
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
    """Names the library `libseamsh.so` on every platform: the name seamsh/_tools.py loads."""

    def get_ext_filename(self, fullname):
        return os.path.join(*fullname.split(".")) + ".so"


setup(
    version=version,
    ext_modules=[lib],
    cmdclass={"bdist_wheel": bdist_wheel, "build_ext": build_ext},
)
