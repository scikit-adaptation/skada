#
# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause

import os
import re
import sys

import pydata_sphinx_theme  # noqa
import sphinx_gallery  # noqa
from numpydoc import docscrape, numpydoc  # noqa

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("_pygments"))


# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.graphviz",
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "sphinx_design",
]

autosummary_generate = True
numpydoc_show_class_members = False

templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
source_encoding = "utf-8-sig"
master_doc = "index"

project = "SKADA"
copyright = "2024, The SKADA team"
author = "Théo Gnassounou, Rémi Flamary, Oleksii Kachaiev"

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    open("../../skada/version.py").read(),
).group(1)
version = __version__
release = __version__

language = "English"
exclude_patterns = ["build"]

# Register the custom SKADA Pygments styles so both Sphinx's generic
# highlighter and pydata-sphinx-theme's own light/dark switcher can find
# them by name (pydata-sphinx-theme looks styles up via
# ``pygments.styles.get_all_styles()``, which only sees registered styles).
import pygments.styles as _pygments_styles  # noqa: E402

_pygments_styles.STYLES["SkadaLightStyle"] = ("skada_pygments", "skada-light", ())
_pygments_styles.STYLES["SkadaDarkStyle"] = ("skada_pygments", "skada-dark", ())
_pygments_styles._STYLE_NAME_TO_MODULE_MAP["skada-light"] = (
    "skada_pygments",
    "SkadaLightStyle",
)
_pygments_styles._STYLE_NAME_TO_MODULE_MAP["skada-dark"] = (
    "skada_pygments",
    "SkadaDarkStyle",
)

pygments_style = "skada_pygments.SkadaLightStyle"
pygments_dark_style = "skada_pygments.SkadaDarkStyle"
highlight_language = "python3"
todo_include_todos = True

# Many examples share generic section headings (e.g. "Illustration of the
# problem with no domain adaptation"); prefix autosectionlabel targets with
# the document path so they don't collide across examples.
autosectionlabel_prefix_document = True

suppress_warnings = [
    # contributing.rst/releases.rst `.. include::` CONTRIBUTING.md/RELEASES.md
    # with `:start-line:` to skip the markdown's own top-level heading (the
    # rst wrapper already provides its own title) -- myst-parser always warns
    # that the included fragment doesn't start at H1, which is expected here.
    "myst.header",
    # RELEASES.md repeats "What's Changed" / "New Contributors" headings once
    # per release entry; they all collapse to the same autosectionlabel
    # target since they're included into the single "releases" document.
    "autosectionlabel.releases",
]


# -- HTML output (PyData Sphinx Theme) ------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    # Syntax-highlighting styles (see conf.py registration above)
    "pygments_light_style": "skada-light",
    "pygments_dark_style": "skada-dark",
    # Logo: light / dark variants
    "logo": {
        "image_light": "_static/images/skada_logo_full.svg",
        "image_dark": "_static/images/skada_logo_full_white.svg",
        "alt_text": "SKADA",
    },
    # Top-navbar icon links
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/scikit-adaptation/skada",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/skada/",
            "icon": "fa-solid fa-box",
            "type": "fontawesome",
        },
    ],
    # Navigation
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    # Secondary sidebar (right column)
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "show_toc_level": 2,
    # Footer
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    # Version switcher (optional — wire up later)
    # "switcher": {
    #     "json_url": "https://scikit-adaptation.github.io/_static/switcher.json",
    #     "version_match": release,
    # },
    "use_edit_page_button": False,
}

# Navigation bar items
html_title = "SKADA"

html_logo = "_static/images/skada_logo_full.svg"

html_favicon = None  # add a .ico later if desired

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# Hide secondary sidebar on the landing page
html_sidebars = {
    "index": [],  # full-width hero on home
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"],
}


# -- Intersphinx ----------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}


# -- Sphinx Gallery -------------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": "auto_examples",
    "nested_sections": False,
    "backreferences_dir": "gen_modules/backreferences",
    "inspect_global_variables": True,
    "doc_module": ("skada", "ot", "numpy", "scipy", "pylab"),
    "matplotlib_animations": True,
    "reference_url": {"skada": None},
}


# -- LaTeX / man page stubs -----------------------------------------------

latex_documents = [
    (master_doc, "SKADA.tex", "SKADA: Scikit Adaptation", author, "manual"),
]
man_pages = [(master_doc, "skada", "SKADA: Scikit Adaptation", [author], 1)]
htmlhelp_basename = "SKADAdoc"
