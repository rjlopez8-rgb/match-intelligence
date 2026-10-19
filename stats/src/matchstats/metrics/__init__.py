"""Metric plug-ins. Importing this package registers every metric in run order."""

from . import (  # noqa: F401
    control,
    momentum,
    passing,
    physical,
    possession,
    pressing,
    shooting,
    summary,
    win_probability,
)
from .base import REGISTRY, Metric, register

__all__ = ["REGISTRY", "Metric", "register"]
