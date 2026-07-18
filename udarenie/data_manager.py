"""
Data management: downloading and caching model/dictionary files.
"""

import os
import zipfile
from pathlib import Path
import urllib.request
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 
# Release URLs for data files
DATA_URLS = [
    "https://github.com/omogr/russian-stress-benchmark/releases/download/v1.0.0/accent_engine.zip",
    "https://github.com/omogr/russian-stress-benchmark/releases/download/v1.0.0/morph_enhancer.zip"
]

DEFAULT_CACHE_SUBDIR = Path.home() / ".cache" / "udarenie" / "data"


def get_default_data_dir() -> Path:
    """
    Return the default data directory.

    Uses ~/.cache/russian_accentor/data to avoid polluting the package directory
    and to survive package upgrades.
    """
    return DEFAULT_CACHE_SUBDIR


def _check_data_complete(data_dir: Path) -> bool:
    """Check if all required data files are present."""
    accent_engine_dir = data_dir / "accent_engine"
    morph_file = data_dir / "morph_enhancer" / "morph.pq"
    return accent_engine_dir.exists() and morph_file.exists()


def ensure_data(data_dir: Path = None, force: bool = False) -> Path:
    """
    Ensure data files are downloaded and extracted.

    Parameters
    ----------
    data_dir : Path, optional
        Directory to store data. Uses ~/.cache/russian_accentor/data if None.
    force : bool, default False
        If True, re-download and overwrite existing data (useful for updates).

    Returns
    -------
    Path
        Path to the data directory.

    Raises
    ------
    RuntimeError
        If download or extraction fails.
    """
    if data_dir is None:
        data_dir = get_default_data_dir()
    else:
        data_dir = Path(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists and is complete
    if not force and _check_data_complete(data_dir):
        logger.info("Data already present at %s", data_dir)
        return data_dir
        
    print('ensure_data', data_dir)

    # Download and extract each archive
    for url in DATA_URLS:
        zip_name = url.split("/")[-1]
        zip_path = data_dir / zip_name

        if force or not zip_path.exists():
            logger.info("Downloading %s...", zip_name)
            try:
                urllib.request.urlretrieve(url, zip_path)
                logger.info("Downloaded %s", zip_name)
            except Exception as e:
                logger.error("Failed to download %s: %s", url, e)
                raise RuntimeError("Failed to download " + url + ": " + str(e)) from e
        else:
            logger.info("Using cached %s", zip_name)

        # Extract
        logger.info("Extracting %s...", zip_name)
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(data_dir)
        except zipfile.BadZipFile as e:
            logger.error("Corrupted zip file %s: %s", zip_path, e)
            zip_path.unlink(missing_ok=True)
            raise RuntimeError("Corrupted zip file " + str(zip_path)) from e

        # Remove zip to save space
        zip_path.unlink(missing_ok=True)

    # Verify completeness
    if not _check_data_complete(data_dir):
        raise RuntimeError(
            "Data incomplete after extraction. Expected: "
            + str(data_dir / "accent_engine") + " and "
            + str(data_dir / "morph_enhancer" / "morph.pq")
        )

    logger.info("Data ready at %s", data_dir)
    return data_dir


def update_data(data_dir: Path = None) -> Path:
    """
    Force re-download data. Useful when dictionaries or models are updated.

    Parameters
    ----------
    data_dir : Path, optional
        Directory to store data. Uses default cache location if None.

    Returns
    -------
    Path
        Path to the data directory.
    """
    logger.info("Forcing data update...")
    return ensure_data(data_dir, force=True)
