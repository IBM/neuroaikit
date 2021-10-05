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
import sys
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'NeuroAIKit'
copyright = '2021, IBM Research'
author = 'IBM Research'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
'nbsphinx','sphinx_rtd_theme',
## Configuration from: https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion/tree/master/docs/_templates
    'sphinx.ext.autodoc',  # Core Sphinx library for auto html doc generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables for modules/classes/methods etc
    'sphinx.ext.viewcode',  # Add a link to the Python source code for classes, functions etc.
    #'sphinx_autodoc_typehints', # Automatically document param types (less noise in class signature)
    'nbsphinx',  # Integrate Jupyter Notebooks and Sphinx
    'IPython.sphinxext.ipython_console_highlighting',
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
# autodoc_inherit_docstrings = True  # If no docstring, inherit from base class -- Creates too much text.
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
# add_module_names = False # Remove namespaces from class/method signatures -- useful for us: tf.identity

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 
     'Examples/Untitled*', 'Examples/prv_*', 'Examples/*-Copy*', 'prv_*', '*-Copy*'] # Exclude temporary / dummy notebooks


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Logo and css color:
html_logo = '_static/neuromorphic_150_text.png'
html_theme_options = {'logo_only': True}

def setup(app):
    app.add_css_file('custom.css')
