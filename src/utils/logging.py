"""
Logging utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str):

    Path("logs").mkdir(exist_ok=True)

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    if logger.handlers:

        return logger

    formatter = logging.Formatter(

        "%(asctime)s | %(levelname)s | %(message)s"

    )

    fh = logging.FileHandler(

        "logs/project.log"

    )

    fh.setFormatter(formatter)

    logger.addHandler(fh)

    logger.addHandler(

        logging.StreamHandler()

    )

    return logger
