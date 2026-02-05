"""
Transposition optimization strategies for m2m framework.

Implements pluggable transposition algorithms using Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

# Handle imports for both package and direct execution modes
try:
    # Package mode: python -m m2m.strategies.transposition
    from ..core.models import TranspositionConfig, TranspositionStrategy
    from .game_piano_constants import (
        GAME_PIANO_PITCHES,
        GAME_PIANO_MIN_PITCH,
        GAME_PIANO_MAX_PITCH,
        is_playable_pitch,
    )
except ImportError:
    # Direct mode: python app.py (from project root)
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from core.models import TranspositionConfig, TranspositionStrategy
    from strategies.game_piano_constants import (
        GAME_PIANO_PITCHES,
        GAME_PIANO_MIN_PITCH,
        GAME_PIANO_MAX_PITCH,
        is_playable_pitch,
    )


class TranspositionOptimizerABC(ABC):
    """Abstract base class for transposition optimizers."""

    @abstractmethod
    def optimize(
        self, notes: List[tuple], config: TranspositionConfig
    ) -> Tuple[int, float]:
        """
        Find optimal transposition for a set of notes.

        Args:
            notes: List of (start, end, pitch) tuples
            config: TranspositionConfig with optimization parameters

        Returns:
            Tuple of (best_transpose, white_key_rate)
        """
        pass


class WhiteKeyRateOptimizer(TranspositionOptimizerABC):
    """
    Optimizes transposition for game piano playability.

    This optimizer maximizes the number of notes that fall within the
    game piano's playable range (48-83, 21 white keys).

    Only notes that map to game piano keys (C3-B5) are counted.
    Notes outside this range are considered unplayable and reduce the rate.

    This is the definitive algorithm for game piano automation.
    """

    def optimize(
        self, notes: List[tuple], config: TranspositionConfig
    ) -> Tuple[int, float]:
        """
        Find transposition that maximizes game piano playability.

        Args:
            notes: List of (start, end, pitch) tuples
            config: TranspositionConfig

        Returns:
            Tuple of (best_transpose, best_playable_rate)
        """
        best_transpose = 0
        best_rate = -1.0

        for transpose in range(config.semitone_min, config.semitone_max + 1):
            rate = self._calculate_playable_rate(
                notes,
                transpose,
                config.melody_threshold,
                config.enable_melody_threshold,
            )

            # Choose higher rate, or prefer smaller transpose if rates are equal
            if rate > best_rate or (
                abs(rate - best_rate) < 1e-9 and abs(transpose) < abs(best_transpose)
            ):
                best_rate = rate
                best_transpose = transpose

        return best_transpose, best_rate

    @staticmethod
    def _calculate_playable_rate(
        notes: List[tuple],
        transpose: int,
        melody_threshold: int = 60,
        enable_threshold: bool = True,
    ) -> float:
        """
        Calculate playable rate for transposed notes.

        Only notes that fall within the game piano's playable range (48-83)
        are counted as playable. Notes outside this range are unplayable.

        Args:
            notes: List of (start, end, pitch) tuples
            transpose: Transposition in semitones
            melody_threshold: Minimum pitch for melody-only mode
            enable_threshold: If True, only consider notes above melody_threshold.
                            If False, consider all notes.

        Returns:
            Playable rate (0.0 to 1.0) - fraction of notes in playable range
        """
        if not notes:
            return 0.0

        playable_count = 0
        total_count = 0

        for _, _, pitch in notes:
            # Only consider notes above melody threshold if threshold is enabled
            if not enable_threshold or pitch >= melody_threshold:
                transposed_pitch = pitch + transpose

                # Check if transposed pitch is in game piano playable range
                if is_playable_pitch(transposed_pitch):
                    playable_count += 1
                total_count += 1

        return playable_count / total_count if total_count > 0 else 0.0

    # Backward compatibility aliases
    _game_piano_playable_rate = _calculate_playable_rate
    _white_key_rate = _calculate_playable_rate


class TranspositionFactory:
    """Factory for creating transposition optimizers."""

    @staticmethod
    def create(strategy: TranspositionStrategy) -> TranspositionOptimizerABC:
        """
        Create a transposition optimizer instance.

        Args:
            strategy: TranspositionStrategy enum value

        Returns:
            TranspositionOptimizerABC instance

        Raises:
            ValueError: If strategy is not supported
        """
        if strategy == TranspositionStrategy.WHITE_KEY_RATE:
            return WhiteKeyRateOptimizer()
        elif strategy == TranspositionStrategy.GAME_PIANO:
            # Import here to avoid circular dependency
            from .game_piano import GamePianoOptimizer

            return GamePianoOptimizer()
        else:
            raise ValueError(
                f"Strategy {strategy.value} is not supported for game piano. "
                f"Use 'white_key_rate' or 'game_piano'."
            )
