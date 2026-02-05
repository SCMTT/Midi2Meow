"""
Game Piano Optimizer - Specialized transposition strategy for 3-octave game pianos.

This module implements a transposition optimizer specifically designed for
game pianos with limited range (21 white keys, 3 octaves).

Key design principles:
- ONLY metric is playability (notes must map to game piano keys)
- Keyboard range is strictly limited (C3-B5, MIDI 48-83)
- No black keys (game pianos typically only have white keys)
- Based on reference implementation from midi2lrcp project

Note: This class inherits from WhiteKeyRateOptimizer and uses the same
playable rate calculation logic. The inheritance is kept for API compatibility.
"""

from typing import List, Tuple

# Handle imports for both package and direct execution modes
try:
    # Package mode: python -m m2m.strategies.game_piano
    from .transposition import WhiteKeyRateOptimizer, TranspositionConfig
except ImportError:
    # Direct mode: python app.py (from project root)
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from strategies.transposition import WhiteKeyRateOptimizer
    from core.models import TranspositionConfig


class GamePianoOptimizer(WhiteKeyRateOptimizer):
    """
    Optimizes transposition for game piano playability.

    This is the definitive optimizer for game piano automation.
    The ONLY metric is playability - notes must map to the 21 playable keys.

    Inherits core optimization logic from WhiteKeyRateOptimizer.
    Based on the reference implementation from midi2lrcp project.
    """

    def __init__(self):
        """
        Initialize game piano optimizer.

        Uses fixed keyboard range: C3-B5 (MIDI 48-83, 21 white keys).
        """
        # Call parent __init__ if needed (currently parent has no __init__)
        self.min_pitch = 48  # C3
        self.max_pitch = 83  # B5
