# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present Srikanth Pagadarai

# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "Next-Gen Wireless DL Demos"
copyright = "2025, Srikanth Pagadarai"
author = "Srikanth Pagadarai"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "Next-Gen Wireless DL Demos"

# Furo theme options
html_theme_options = {
    "source_repository": "https://github.com/SrikanthPagadarai/nextgen-wireless-dl-demos",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- Extension configuration -------------------------------------------------

# Mock imports for heavy dependencies that aren't needed for documentation
autodoc_mock_imports = [
    "tensorflow",
    "tensorflow.keras",
    "tensorflow.keras.layers",
    "tensorflow.keras.Model",
    "tensorflow.nn",
    "tf",
    "sionna",
    "sionna.phy",
    "sionna.phy.ofdm",
    "sionna.phy.mapping",
    "sionna.phy.fec",
    "sionna.phy.fec.ldpc",
    "sionna.phy.channel",
    "sionna.phy.channel.tr38901",
    "sionna.phy.mimo",
    "sionna.phy.nr",
    "sionna.phy.nr.utils",
    "sionna.phy.signal",
    "sionna.phy.utils",
    "sionna.rt",
    "numpy",
    "np",
    "scipy",
    "scipy.signal",
    "matplotlib",
]

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "tensorflow": ("https://www.tensorflow.org/api_docs/python", None),
}

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
