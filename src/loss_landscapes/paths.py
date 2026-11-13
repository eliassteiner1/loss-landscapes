import os
from pathlib import Path

def _find_project_root(marker: str = "pyproject.toml") -> Path:
    """ 
    helper to define project root for all other scripts. looks for the parent folder of pyproject.toml. this fails, if the repo were to be installed non-editable (to safeguard against writing data files into some .venv folders).
    """
    
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(
        f"no {marker} found above {here}, thus couldn't set project root! "
        f"make sure to use this package as an editable install from the repo."
    )

PROJ_ROOT = _find_project_root()
