# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re


# -- Project information -----------------------------------------------------

project = 'seamsh'
copyright = ('2010-2020, Université catholique de Louvain (UCLouvain); '
             '2026, Stefano Campanella (this fork)')
author = 'Jonathan Lambrechts, Stefano Campanella'

# The version is set in setup.py (and kept in step with CHANGELOG.rst).
with open(os.path.join(os.path.dirname(__file__), '..', 'setup.py')) as f:
    release = re.search(r'^version = "([^"]+)"', f.read(), re.M).group(1)
version = release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.napoleon',
        'sphinx_autodoc_typehints',
        'sphinx_gallery.gen_gallery',
        ]

# The examples (../examples/*.py) are run, and their figures shown, by sphinx-gallery: it
# writes the pages into examples/. They run in ../examples, where they download their data and
# write their meshes. Example 5 needs a file that the data archive does not have: it is shown
# but not run.
sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'examples',
    'filename_pattern': r'/(?!5-)[^/]+\.py$',
    'within_subsection_order': 'FileNameSortKey',
    'write_computation_times': False,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_special_with_doc = True
autoclass_content = 'both'
html_show_sourcelink = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

