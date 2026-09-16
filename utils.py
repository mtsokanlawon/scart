"""
utils.py

Shared helper functions.
"""

import logging
import os
import sys
from datetime import datetime

from config import LOG_DIR, LOG_LEVEL


def create_logger():

    LOG_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger("SmartFollower")

    logger.setLevel(getattr(logging, LOG_LEVEL))

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        "%H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)

    console.setFormatter(formatter)

    logger.addHandler(console)

    logfile = LOG_DIR / f"{datetime.now():%Y%m%d}.log"

    filehandler = logging.FileHandler(logfile)

    filehandler.setFormatter(formatter)

    logger.addHandler(filehandler)

    return logger


def display_available():

    return bool(os.environ.get("DISPLAY"))
