"""Leakage-safe historical day-ahead dataset construction."""

from .strict_dataset import StrictBacktestBuilder, StrictDataError

__all__ = ["StrictBacktestBuilder", "StrictDataError"]
