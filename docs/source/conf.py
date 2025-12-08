# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "E2E AI Challenge Playground"
copyright = "2025, Masa"
author = "Masa"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add project root to sys.path so that modules can be imported
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../core/src"))
sys.path.insert(0, os.path.abspath("../../simulators/core/src"))
sys.path.insert(0, os.path.abspath("../../simulators/simulators.kinematic/src"))
sys.path.insert(0, os.path.abspath("../../experiment/src"))
sys.path.insert(0, os.path.abspath("../../experiment/training/src"))
sys.path.insert(0, os.path.abspath("../../ad_components/core/src"))
sys.path.insert(0, os.path.abspath("../../ad_components/planning/pure_pursuit/src"))
sys.path.insert(0, os.path.abspath("../../ad_components/planning/planning_utils/src"))
sys.path.insert(0, os.path.abspath("../../ad_components/control/pid_controller/src"))
sys.path.insert(0, os.path.abspath("../../ad_components/control/neural_controller/src"))
sys.path.insert(0, os.path.abspath("../../dashboard/src"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Support for Google/NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.githubpages",
    "myst_parser",  # Support for Markdown
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# -- Options for myst-parser -------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "tasklist",
]
