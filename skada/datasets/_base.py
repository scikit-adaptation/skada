import os
from typing import Optional, Union


_DEFAULT_HOME_FOLDER_KEY = "SKADA_DATA_FOLDER"
_DEFAULT_HOME_FOLDER = "~/skada_datasets"


def get_data_home(data_home: Optional[Union[str, os.PathLike]] = None) -> str:
    """Return the path of the `skada` data folder.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data directory is set to a folder named 'skada_datasets' in the
    user home folder.

    Alternatively, it can be set by the 'SKADA_DATA_FOLDER' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str or path-like, default=None
        The path to `skada` data folder. If `None`, the default path
        is `~/skada_datasets`.

    Returns
    -------
    data_home: str
        The path to `skada` data folder.
    """
    if data_home is None:
        data_home = os.environ.get(_DEFAULT_HOME_FOLDER_KEY, _DEFAULT_HOME_FOLDER)
    data_home = os.path.expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return data_home
