"""
SSM-Pro Segmentation - Enhanced SSM with Multi-Feature Fusion.

This module extends the SSM-based segmentation strategy by fusing multiple
feature types for better boundary detection.

Key improvements over standard SSM:
- Chroma features (12-dimensional pitch class profile) - captures harmonic changes
- Velocity statistics - captures dynamics changes
- Rhythmic features - captures tempo/activity changes
- Weighted feature fusion with configurable weights
- Better normalization and scaling strategies

Based on research insights from:
- "Barwise Music Structure Analysis with the Correlation Block-Matching Segmentation Algorithm" (Marmoret et al., 2023)
- "Self-Similarity-Based and Novelty-Based loss for music structure analysis" (Peeters, 2023)
"""

from typing import List, Tuple, Dict
import numpy as np
import pretty_midi
from scipy import signal
from scipy.spatial.distance import pdist, squareform

# Handle imports for both package and direct execution modes
try:
    # Package mode: python -m m2m.strategies.ssm_pro_segmentation
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


class SSMProSegmentationStrategy(SegmentationStrategyABC, SSMBaseStrategy):
    """
    Enhanced SSM segmentation with multi-feature fusion.

    This strategy extends standard SSM by:
    1. Extracting multiple complementary feature types
    2. Normalizing each feature type independently
    3. Fusing features with configurable weights
    4. Computing SSM from the fused feature representation

    Features extracted:
    - Pitch features: average pitch, pitch variance, pitch range
    - Chroma features: 12-dimensional pitch class profile
    - Density features: note density, onset rate
    - Velocity features: mean velocity, velocity variance
    """

    def __init__(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        feature_weights: Dict[str, float] = None,
    ):
        """
        Initialize SSM-Pro segmentation strategy.

        Args:
            midi_data: PrettyMIDI object
            feature_weights: Optional weights for feature fusion
                Defaults: {'pitch': 1.0, 'chroma': 1.5, 'density': 1.0, 'velocity': 0.5}
                Chroma gets higher weight as it's most important for harmony/section changes
        """
        super().__init__(midi_data)
        self.window_size = 1.0  # 1 second windows for feature extraction
        self.min_segment_duration = 2.0  # Minimum segment length

        # Default feature weights (tuned for typical pop music)
        self.feature_weights = feature_weights or {
            "pitch": 1.0,  # Melodic contour
            "chroma": 1.5,  # Harmonic content (most important for sections)
            "density": 1.0,  # Note density / activity
            "velocity": 0.5,  # Dynamics (less important for structure)
        }

    def split(
        self, melody_notes: List[tuple] = None, min_duration: float = 2.0, **kwargs
    ) -> List[Section]:
        """
        Split MIDI using SSM-Pro (multi-feature fusion).

        Args:
            melody_notes: Optional list of melody notes (for filtering)
            min_duration: Minimum segment duration in seconds
            **kwargs: Additional parameters (feature_weights for custom weights)

        Returns:
            List of Section objects
        """
        self.min_segment_duration = min_duration

        # Allow custom feature weights via kwargs
        if "feature_weights" in kwargs:
            self.feature_weights.update(kwargs["feature_weights"])

        # Collect all notes
        notes = self._collect_notes()
        if not notes:
            return []

        # Extract multi-modal features
        features = self._extract_multi_features(notes)

        if len(features) < 2:
            # Not enough data for SSM, fall back to basic
            try:
                from .segmentation import BasicSegmentationStrategy
            except ImportError:
                from strategies.segmentation import BasicSegmentationStrategy

            basic = BasicSegmentationStrategy(self.midi_data)
            return basic.split(melody_notes=melody_notes, min_duration=min_duration)

        # Compute Self-Similarity Matrix from fused features
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

            section = Section(
                start=start, end=end, reasons=["SSM-Pro分段", "多特征融合"]
            )
            sections.append(section)

        # Merge very short sections with neighbors
        sections = self._merge_short_sections(sections, min_duration)

        return sections

    def _extract_multi_features(self, notes: List[tuple]) -> np.ndarray:
        """
        Extract multiple feature types from MIDI notes.

        Returns:
            Feature matrix (n_windows x n_features) where n_features is the
            concatenation of all feature types after normalization
        """
        # Create time windows
        total_duration = max(note[1] for note in notes)
        n_windows = int(total_duration / self.window_size) + 1

        # Store different feature groups separately
        pitch_features = []
        chroma_features = []
        density_features = []
        velocity_features = []

        for i in range(n_windows):
            window_start = i * self.window_size
            window_end = (i + 1) * self.window_size

            # Get notes in this window
            window_notes = [
                (s, e, p, v)
                for s, e, p, v in self._get_notes_with_velocity()
                if window_start <= s < window_end or window_start <= e < window_end
            ]

            if not window_notes:
                # Empty window - use zeros
                pitch_features.append([0.0, 0.0, 0.0])
                chroma_features.append([0.0] * 12)
                density_features.append([0.0, 0.0])
                velocity_features.append([0.0, 0.0])
                continue

            # Extract various features
            pitch_feat = self._extract_pitch_features(window_notes)
            chroma_feat = self._extract_chroma_features(window_notes)
            density_feat = self._extract_density_features(window_notes)
            velocity_feat = self._extract_velocity_features(window_notes)

            pitch_features.append(pitch_feat)
            chroma_features.append(chroma_feat)
            density_features.append(density_feat)
            velocity_features.append(velocity_feat)

        # Normalize each feature group independently
        pitch_features = self._normalize_features(np.array(pitch_features))
        chroma_features = self._normalize_features(np.array(chroma_features))
        density_features = self._normalize_features(np.array(density_features))
        velocity_features = self._normalize_features(np.array(velocity_features))

        # Apply weights and concatenate
        weighted_features = []
        for i in range(n_windows):
            # Weight each feature group
            weighted_pitch = pitch_features[i] * self.feature_weights["pitch"]
            weighted_chroma = chroma_features[i] * self.feature_weights["chroma"]
            weighted_density = density_features[i] * self.feature_weights["density"]
            weighted_velocity = velocity_features[i] * self.feature_weights["velocity"]

            # Concatenate all weighted features
            all_features = np.concatenate(
                [weighted_pitch, weighted_chroma, weighted_density, weighted_velocity]
            )
            weighted_features.append(all_features)

        return np.array(weighted_features)

    def _get_notes_with_velocity(self) -> List[tuple]:
        """
        Collect notes with velocity information.

        Returns:
            List of (start, end, pitch, velocity) tuples
        """
        notes = []
        for instrument in self.midi_data.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                notes.append((note.start, note.end, note.pitch, note.velocity))

        # Sort by start time
        notes.sort(key=lambda x: x[0])
        return notes

    def _extract_pitch_features(self, window_notes: List[tuple]) -> List[float]:
        """
        Extract pitch-related features.

        Returns:
            [average_pitch, pitch_variance, pitch_range] (normalized)
        """
        pitches = [p for _, _, p, _ in window_notes]

        # Feature 1: Average pitch (normalized)
        avg_pitch = np.mean(pitches) / 127.0

        # Feature 2: Pitch variance (normalized)
        pitch_var = np.var(pitches) / (127.0**2)

        # Feature 3: Pitch range (normalized)
        pitch_range = (max(pitches) - min(pitches)) / 127.0

        return [avg_pitch, pitch_var, pitch_range]

    def _extract_chroma_features(self, window_notes: List[tuple]) -> List[float]:
        """
        Extract chroma (pitch class profile) features.

        Returns:
            12-dimensional chroma vector (normalized to sum to 1)
        """
        chroma = np.zeros(12)

        for _, _, pitch, _ in window_notes:
            chroma[pitch % 12] += 1

        # Normalize to sum to 1 (probability distribution)
        if chroma.sum() > 0:
            chroma = chroma / chroma.sum()

        return chroma.tolist()

    def _extract_density_features(self, window_notes: List[tuple]) -> List[float]:
        """
        Extract density-related features.

        Returns:
            [note_density, onset_rate] (normalized)
        """
        # Feature 1: Note density (notes per second)
        density = len(window_notes) / self.window_size
        # Normalize (assume max 20 notes per second)
        density = min(density / 20.0, 1.0)

        # Feature 2: Onset rate (unique start times per second)
        unique_onsets = len(set(s for s, _, _, _ in window_notes))
        onset_rate = unique_onsets / self.window_size
        # Normalize (assume max 15 onsets per second)
        onset_rate = min(onset_rate / 15.0, 1.0)

        return [density, onset_rate]

    def _extract_velocity_features(self, window_notes: List[tuple]) -> List[float]:
        """
        Extract velocity/dynamics features.

        Returns:
            [mean_velocity, velocity_variance] (normalized)
        """
        velocities = [v for _, _, _, v in window_notes]

        # Feature 1: Mean velocity (normalized)
        mean_vel = np.mean(velocities) / 127.0

        # Feature 2: Velocity variance (normalized)
        vel_var = np.var(velocities) / (127.0**2)

        return [mean_vel, vel_var]

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 1] range.

        Args:
            features: Feature matrix (n_windows x n_features)

        Returns:
            Normalized feature matrix
        """
        # Normalize each feature dimension independently
        normalized = np.zeros_like(features)

        for i in range(features.shape[1]):
            col = features[:, i]
            col_min = col.min()
            col_max = col.max()

            if col_max > col_min:
                normalized[:, i] = (col - col_min) / (col_max - col_min)
            else:
                # Constant column
                normalized[:, i] = 0.5

        return normalized


