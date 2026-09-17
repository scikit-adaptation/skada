"""Pygments styles using only the SKADA logo's blue/red/olive palette."""

from pygments.style import Style
from pygments.token import Comment, Error, Generic, Keyword, Name, Number, String, Token

_BLUE = "#2364aa"
_RED = "#c84630"
_OLIVE = "#6D8E64"
_TEXT = "#2b2b2b"


class SkadaLightStyle(Style):
    """A light style using only the SKADA logo's blue/red/olive palette."""

    name = "skada-light"
    background_color = "#fdf6e3"
    highlight_color = "#f7e7b4"
    line_number_color = "#9a9a9a"

    styles = {
        Token: _TEXT,
        Comment: f"italic {_OLIVE}",
        Keyword: f"bold {_BLUE}",
        Keyword.Constant: f"bold {_BLUE}",
        Keyword.Namespace: f"bold {_BLUE}",
        Name.Builtin: _BLUE,
        Name.Function: f"bold {_OLIVE}",
        Name.Class: f"bold {_OLIVE}",
        Name.Decorator: _OLIVE,
        Name.Exception: f"bold {_RED}",
        String: _RED,
        String.Doc: f"italic {_RED}",
        Number: _RED,
        Generic.Deleted: _RED,
        Generic.Inserted: _BLUE,
        Generic.Error: _RED,
        Error: f"bg:#f8d7da {_RED}",
    }


class SkadaDarkStyle(Style):
    """A dark style using only the SKADA logo's blue/red/olive palette."""

    name = "skada-dark"
    background_color = "#1e1a10"
    highlight_color = "#4a3d17"
    line_number_color = "#8a8a8a"

    _BLUE_D = "#6ea6db"
    _RED_D = "#e2917d"
    _OLIVE_D = "#9cc39a"
    _TEXT_D = "#e8e6df"

    styles = {
        Token: _TEXT_D,
        Comment: f"italic {_OLIVE_D}",
        Keyword: f"bold {_BLUE_D}",
        Keyword.Constant: f"bold {_BLUE_D}",
        Keyword.Namespace: f"bold {_BLUE_D}",
        Name.Builtin: _BLUE_D,
        Name.Function: f"bold {_OLIVE_D}",
        Name.Class: f"bold {_OLIVE_D}",
        Name.Decorator: _OLIVE_D,
        Name.Exception: f"bold {_RED_D}",
        String: _RED_D,
        String.Doc: f"italic {_RED_D}",
        Number: _RED_D,
        Error: f"bg:#4e111b {_RED_D}",
    }
