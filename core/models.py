"""
Data models for m2m (MIDI→MIDI) framework.

Defines core data structures used across the framework.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class SegmentationStrategy(Enum):
    """Available segmentation strategies."""

    BASIC = "basic"  # Tempo/key/gap based
    ADAPTIVE = "adaptive"  # Iterative white key rate optimization
    SSM = "ssm"  # Self-Similarity Matrix
    SSM_PRO = "ssm_pro"  # Enhanced SSM with multi-feature fusion


class TranspositionStrategy(Enum):
    """Available transposition optimization strategies for game piano."""

    WHITE_KEY_RATE = "white_key_rate"  # Maximize notes in playable range (48-83)
    GAME_PIANO = "game_piano"  # Optimized for 3-octave game pianos (21 white keys)

    # Note: Both strategies now use the same logic - maximizing playability.
    # WHITE_KEY_RATE is the general optimizer, GAME_PIANO is game-specific.
    # For game piano projects, both produce identical results.


@dataclass
class TranspositionConfig:
    """Configuration for transposition optimization."""

    strategy: TranspositionStrategy = TranspositionStrategy.WHITE_KEY_RATE
    semitone_min: int = -27
    semitone_max: int = 27
    melody_threshold: int = 60
    enable_melody_threshold: bool = True


@dataclass
class SSMProConfig:
    """
    Configuration for SSM-Pro segmentation strategy.

    SSM-Pro uses multi-feature fusion for improved segmentation:
    - Pitch features: Average pitch, variance, range
    - Chroma features: 12-dimensional pitch class profile (most important for harmony)
    - Density features: Note density, onset rate
    - Velocity features: Mean velocity, velocity variance
    """

    # Feature weights for multi-feature fusion
    # Chroma gets highest weight as it's most important for harmonic/section changes
    pitch_weight: float = 1.0
    chroma_weight: float = 1.5
    density_weight: float = 1.0
    velocity_weight: float = 0.5

    # Minimum segment duration in seconds
    min_segment_duration: float = 2.0

    def to_feature_weights_dict(self) -> dict:
        """Convert to dictionary format expected by SSMProSegmentationStrategy."""
        return {
            "pitch": self.pitch_weight,
            "chroma": self.chroma_weight,
            "density": self.density_weight,
            "velocity": self.velocity_weight,
        }


@dataclass
class Section:
    """
    Represents a time segment with its optimal transposition.

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        transpose: Optimal transposition in semitones (-27 to +27)
        white_key_rate: White key rate with optimal transposition
        white_key_rate_0: White key rate with no transposition
        note_count: Number of notes in this segment
        confidence: Confidence score (0.0 to 1.0) for this segmentation
        reasons: List of reasons why this segment was created
    """

    start: float
    end: float
    transpose: int = 0
    white_key_rate: float = 0.0
    white_key_rate_0: float = 0.0
    note_count: int = 0
    confidence: float = 1.0
    reasons: List[str] = field(default_factory=list)
    # Berserk Mode
    berserk_mode: bool = False
    max_pitch: int = 83  # B5 (Game piano max pitch)
    # Melody threshold for melody-only mode
    melody_threshold: int = 60

    @property
    def duration(self) -> float:
        """Return segment duration in seconds."""
        return self.end - self.start

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "start": self.start,
            "end": self.end,
            "transpose": self.transpose,
            "white_key_rate": self.white_key_rate,
            "white_key_rate_0": self.white_key_rate_0,
            "note_count": self.note_count,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "berserk_mode": self.berserk_mode,
            "max_pitch": self.max_pitch,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Section":
        """Create Section from dictionary."""
        return cls(**data)


@dataclass
class M2MConfig:
    """
    Main configuration for m2m framework.

    Attributes:
        segmentation_strategy: Which segmentation algorithm to use
        transposition_config: Configuration for transposition optimization
        ssm_pro_config: Configuration for SSM-Pro segmentation (only used when strategy is SSM_PRO)
        min_segment_duration: Minimum segment duration in seconds
        max_transposition_changes: Maximum number of transposition changes
        enable_undo: Whether to enable undo/redo in GUI
    """

    segmentation_strategy: SegmentationStrategy = SegmentationStrategy.ADAPTIVE
    transposition_config: TranspositionConfig = field(
        default_factory=TranspositionConfig
    )
    ssm_pro_config: SSMProConfig = field(default_factory=SSMProConfig)
    min_segment_duration: float = 2.0
    max_transposition_changes: int = 10
    enable_undo: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "segmentation_strategy": self.segmentation_strategy.value,
            "transposition_config": self.transposition_config.__dict__,
            "min_segment_duration": self.min_segment_duration,
            "max_transposition_changes": self.max_transposition_changes,
            "enable_undo": self.enable_undo,
        }
