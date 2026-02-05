"""
m2m - MIDI to MIDI conversion with adaptive segmentation and transposition.

A framework for converting MIDI files through intelligent segmentation and
optimized transposition, making piano music easier to play.
"""

from .core.models import M2MConfig, Section, SegmentationStrategy, TranspositionStrategy
from .core.pipeline import M2MPipeline
from .core.observer import Observer, Event, EventType

__version__ = "1.0.0"
__author__ = "AutoPiano Team"

__all__ = [
    "M2MConfig",
    "Section",
    "SegmentationStrategy",
    "TranspositionStrategy",
    "M2MPipeline",
    "Observer",
    "Event",
    "EventType",
]
