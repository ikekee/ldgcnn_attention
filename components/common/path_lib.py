"""This module contains functions for operating with paths."""
from pathlib import Path


def create_path_if_not_exists(path: Path):
    """Creates a directory if it doesn't exist.

    Args:
        path: A path to folder to check and create.
    """
    if not path.exists():
        path.mkdir(parents=True)


def directory_is_not_empty(directory: Path) -> bool:
    """Checks if a directory is empty.

    Args:
        directory: A path to folder to check.

    Returns:
        True if the directory is not empty, False otherwise.
    """
    return bool(list(directory.glob("*")))
