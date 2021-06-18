# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import woodwork
import os
import sys
import subprocess
import shutil
from pathlib import Path
from sphinx.ext.autodoc import (Documenter, MethodDocumenter)


from sphinx.ext.autodoc import MethodDocumenter, Documenter

path = os.path.join('..', '..')
sys.path.insert(0, os.path.abspath(path))


# -- Project information -----------------------------------------------------

project = 'Woodwork'
copyright = '2020, Alteryx, Inc.'
author = 'Alteryx, Inc.'

# The short X.Y version
version = woodwork.__version__
# The full version, including alpha/beta/rc tags
release = woodwork.__version__


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The main toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "github_url": "https://github.com/alteryx/woodwork",
    "twitter_url": "https://twitter.com/AlteryxOSS",
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "images/woodwork.svg"

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "images/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'Woodworkdoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'Woodwork.tex', 'Woodwork Documentation',
     'Alteryx, Inc.', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'woodwork', 'Woodwork Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'Woodwork', 'Woodwork Documentation',
     author, 'Woodwork', 'One line description of project.',
     'Miscellaneous'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------

# If woodwork is open-sourced: replace github specific style.css
extlinks = {
    'issue': ('https://github.com/alteryx/woodwork/issues/%s', '#'),
    'pr': ('https://github.com/alteryx/woodwork/pull/%s', '#'),
    'user': ('https://github.com/%s', '@')
}

autosummary_generate = ["api_reference.rst"]
templates_path = ["_templates"]

html_show_sphinx = False
nbsphinx_execute = 'always'
nbsphinx_timeout = 600 # sphinx defaults each cell to 30 seconds so we need to override here

inheritance_graph_attrs = dict(rankdir="LR", size='"1000, 333"',
                               fontsize=30, labelfontsize=30, ratio='compress', dpi=960)

class AccessorLevelDocumenter(Documenter):
    """
    Documenter subclass for objects on accessor level (methods, attributes).

    Referenced pandas-sphinx-theme (https://github.com/pandas-dev/pandas-sphinx-theme)
    and sphinx-doc (https://github.com/sphinx-doc/sphinx/blob/8c7faed6fcbc6b7d40f497698cb80fc10aee1ab3/sphinx/ext/autodoc/__init__.py#L846)
    """
    def resolve_name(self, modname, parents, path, base):
        modname = 'woodwork'
        mod_cls = path.rstrip('.')
        mod_cls = mod_cls.split('.')
        return modname, mod_cls + [base]


class AccessorCallableDocumenter(AccessorLevelDocumenter, MethodDocumenter):
    """
    This documenter lets us removes .__call__ from the method signature for
    callable accessors like Series.plot
    """

    objtype = "accessorcallable"
    directivetype = "method"

    # lower than MethodDocumenter; otherwise the doc build prints warnings
    priority = 0.5

    def format_name(self):
        return MethodDocumenter.format_name(self).rstrip(".__call__")


class AccessorMethodDocumenter(AccessorLevelDocumenter, MethodDocumenter):
    objtype = 'accessormethod'
    directivetype = 'method'

    # lower than MethodDocumenter so this is not chosen for normal methods
    priority = 0.6


def setup(app):
    home_dir = os.environ.get('HOME', '/')
    p = Path(home_dir + "/.ipython/profile_default/startup")
    p.mkdir(parents=True, exist_ok=True)
    shutil.copy("source/set-headers.py", home_dir + "/.ipython/profile_default/startup")
    app.add_js_file('https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js')
    app.add_css_file("style.css")
    app.add_autodocumenter(AccessorCallableDocumenter)
    app.add_autodocumenter(AccessorMethodDocumenter)
