"""
SSM-Based Segmentation - Self-Similarity Matrix segmentation strategy.

This module implements a segmentation strategy based on Self-Similarity Matrices (SSM),
which is state-of-the-art for music structure analysis.

Key features:
- Computes SSM from MIDI note features
- Multi-scale novelty detection
- Correlation Block-Matching (CBM) algorithm
- Adaptive thresholding for boundary detection

Based on research:
- "Barwise Music Structure Analysis with the Correlation Block-Matching Segmentation Algorithm" (Marmoret et al., 2023)
- "Self-Similarity-Based and Novelty-Based loss for music structure analysis" (Peeters, 2023)
"""

from typing import List, Tuple
import numpy as np
import pretty_midi
from scipy import signal
from scipy.spatial.distance import pdist, squareform

# Handle imports for both package and direct execution modes
try:
    # Package mode: python -m m2m.strategies.ssm_segmentation
    from ..core.models import Section, SegmentationStrategy
    from .segmentation import SegmentationStrategyABC
    from .ssm_base import SSMBaseStrategy
except ImportError:
    # Direct mode: python app.py (from project root)
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from core.models import Section, SegmentationStrategy
    from strategies.segmentation import SegmentationStrategyABC
    from strategies.ssm_base import SSMBaseStrategy


class SSMSegmentationStrategy(SegmentationStrategyABC, SSMBaseStrategy):
    """
    Segmentation using Self-Similarity Matrix (SSM) approach.

    This strategy:
    1. Extracts features from MIDI (pitch contour, rhythm)
    2. Computes Self-Similarity Matrix
    3. Applies multi-scale novelty detection
    4. Detects boundaries using adaptive thresholding
    5. Refines boundaries considering minimum segment duration
    """

    def __init__(self, midi_data: pretty_midi.PrettyMIDI):
        """
        Initialize SSM segmentation strategy.

        Args:
            midi_data: PrettyMIDI object
        """
        super().__init__(midi_data)
        self.window_size = 1.0  # 1 second windows for feature extraction
        self.min_segment_duration = 2.0  # Minimum segment length

    def split(
        self, melody_notes: List[tuple] = None, min_duration: float = 2.0, **kwargs
    ) -> List[Section]:
        """
        Split MIDI using SSM-based segmentation.

        Args:
            melody_notes: Optional list of melody notes (for filtering)
            min_duration: Minimum segment duration in seconds
            **kwargs: Additional parameters

        Returns:
            List of Section objects
        """
        self.min_segment_duration = min_duration

        # Collect all notes
        notes = self._collect_notes()
        if not notes:
            return []

        # Extract features
        features = self._extract_features(notes)

        if len(features) < 2:
            # Not enough data for SSM, fall back to basic
            from .segmentation import BasicSegmentationStrategy

            basic = BasicSegmentationStrategy(self.midi_data)
            return basic.split(melody_notes=melody_notes, min_duration=min_duration)

        # Compute Self-Similarity Matrix
        ssm = self._compute_ssm(features)

        # Compute novelty curve using multi-scale approach
        novelty = self._compute_novelty_multiscale(ssm)

        # Detect boundaries using adaptive thresholding
        boundaries = self._detect_boundaries_adaptive(novelty, notes)

        # Add start and end boundaries
        total_duration = self._get_midi_duration()
        boundaries = [0.0] + boundaries + [total_duration]
        boundaries = sorted(set(boundaries))

        # Create sections
        sections = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]

            # Skip very short segments
            if end - start < min_duration:
                continue

            section = Section(
                start=start, end=end, reasons=["SSM分段", "多尺度novelty检测"]
            )
            sections.append(section)

        return sections

    def _extract_features(self, notes: List[tuple]) -> np.ndarray:
        """
        Extract features from MIDI notes for SSM computation.

        Features:
        - Pitch contour (normalized)
        - Note density
        - Velocity average

        Args:
            notes: List of (start, end, pitch) tuples

        Returns:
            Feature matrix (n_windows x n_features)
        """
        # Create time windows
        total_duration = max(note[1] for note in notes)
        n_windows = int(total_duration / self.window_size) + 1

        features = []

        for i in range(n_windows):
            window_start = i * self.window_size
            window_end = (i + 1) * self.window_size

            # Get notes in this window
            window_notes = [
                (s, e, p)
                for s, e, p in notes
                if window_start <= s < window_end or window_start <= e < window_end
            ]

            if not window_notes:
                # Empty window - use zeros
                features.append([0.0, 0.0, 0.0])
                continue

            # Extract features
            pitches = [p for _, _, p in window_notes]

            # Feature 1: Average pitch (normalized to 0-1)
            avg_pitch = np.mean(pitches) / 127.0

            # Feature 2: Note density (notes per second)
            density = len(window_notes) / self.window_size
            # Normalize (assume max 20 notes per second)
            density = min(density / 20.0, 1.0)

            # Feature 3: Pitch variance (captures pitch range)
            pitch_var = np.var(pitches) / (127.0**2)  # Normalize

            features.append([avg_pitch, density, pitch_var])

        return np.array(features)
