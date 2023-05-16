# pylint: disable-all
# pylint: skip-file
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = 'Nelder-Mead'
copyright = '2023, Hasan Oren'
author = 'Hasan Oren'
version = '1.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.graphviz',
    'sphinx.ext.imgmath',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

imgmath_image_format = 'svg'
# imgmath_latex = r"O:\MikTeX\miktex\bin\x64\latex.exe"
# imgmath_dvisvgm = r"O:\MikTeX\miktex\bin\x64\dvisvgm.exe"

templates_path = ['_templates']
exclude_patterns = []

highlight_language = 'none'
language = 'ru'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for Epub output -------------------------------------------------

epub_language = 'ru'
