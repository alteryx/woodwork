from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class RangeObject:
    """Class for keeping track of an range object"""

    dt: str
    idx: int
    range: int


@dataclass
class InferDebug:
    """Class for frequency inference debug object"""

    actual_range_start: str
    actual_range_end: str

    message: str = None

    estimated_freq: str = None
    estimated_range_start: str = None
    estimated_range_end: str = None

    duplicate_values: list[RangeObject] = field(default_factory=list)
    missing_values: list[RangeObject] = field(default_factory=list)
    extra_values: list[RangeObject] = field(default_factory=list)


class DataCheckMessageCode(Enum):
    """Enum for data check message code."""

    DATETIME_IS_NOT_MONOTONIC = "datetime_is_not_monotonic"

    DATETIME_FREQ_CANNOT_BE_ESTIMATED = "datetime_freq_cannot_be_estimated"
