"""
logger.py — Centralized structured logging.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("message")
"""
from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LEVEL = os.environ.get("ADA_LOG_LEVEL", "INFO").upper()

_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger("ada")
    root.setLevel(_LEVEL)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(f"ada.{name}")
