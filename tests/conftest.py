# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: GPL-3.0-or-later

import gmsh
import pytest


@pytest.fixture(scope="session", autouse=True)
def gmsh_session():
  """seamsh expects the caller to initialize gmsh; do it once for all tests."""
  gmsh.initialize()
  gmsh.option.setNumber("General.Verbosity", 0)
  yield
  gmsh.finalize()


@pytest.fixture
def model():
  """A fresh current gmsh model, as callers of `seamsh.gmsh.mesh` create. Everything (models, views,
  fields) is cleared afterwards so that tests do not see each other's meshes."""
  gmsh.clear()
  gmsh.model.add("test")
  yield "test"
  gmsh.clear()
