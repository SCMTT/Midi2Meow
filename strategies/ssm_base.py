"""
SSM Base Strategy - Common utilities for SSM-based segmentation strategies.

This module provides shared methods for Self-Similarity Matrix (SSM) segmentation
to avoid code duplication between SSMSegmentationStrategy and SSMProSegmentationStrategy.

Contains:
- SSM computation
- Multi-scale novelty detection
- Adaptive boundary detection
- Minimum gap enforcement
"""

from typing import List
import numpy as np
from scipy import signal
from scipy.spatial.distance import pdist, squareform


class SSMBaseStrategy:
    """
    Base class providing common SSM-based segmentation methods.

    Inherited by:
    - SSMSegmentationStrategy
    - SSMProSegmentationStrategy
    """

    window_size: float = 1.0  # Must be set by subclass
    min_segment_duration: float = 2.0  # Must be set by subclass

    def _compute_ssm(self, features: np.ndarray) -> np.ndarray:
        """
        Compute Self-Similarity Matrix from features.

        Uses Euclidean distance normalized to [0, 1].

        Args:
            features: Feature matrix (n_windows x n_features)

        Returns:
            SSM matrix (n_windows x n_windows)
        """
        # Compute pairwise distances
        distances = pdist(features, metric="euclidean")

        # Convert to distance matrix
        distance_matrix = squareform(distances)

        # Normalize to [0, 1]
        if distance_matrix.max() > 0:
            distance_matrix = distance_matrix / distance_matrix.max()

        # Convert to similarity (1 - distance)
        ssm = 1.0 - distance_matrix

        return ssm

    def _compute_novelty_multiscale(
        self, ssm: np.ndarray, scales: List[int] = [1, 2, 4, 8]
    ) -> np.ndarray:
        """
        Compute novelty curve using multi-scale approach.

        Uses checkerboard kernel at different scales to capture
        structural changes at different time granularities.

        Args:
            ssm: Self-similarity matrix
            scales: List of kernel sizes (in windows)

        Returns:
            Novelty curve (n_windows,)
        """
        novelty_curves = []

        for scale in scales:
            # Create checkerboard kernel
            kernel_size = scale * 2 + 1
            kernel = np.ones((kernel_size, kernel_size))
            kernel[:scale, :] = -1
            kernel[scale:, :] = 1

            # Normalize kernel
            kernel = kernel / np.abs(kernel).sum()

            # Apply convolution
            novelty = signal.convolve2d(ssm, kernel, mode="same", boundary="symm")

            # Take diagonal (novelty at each time point)
            novelty_diag = np.diag(novelty)

            # Normalize to [0, 1]
            if novelty_diag.max() > novelty_diag.min():
                novelty_diag = (novelty_diag - novelty_diag.min()) / (
                    novelty_diag.max() - novelty_diag.min()
                )

            novelty_curves.append(novelty_diag)

        # Combine novelty curves (average)
        combined_novelty = np.mean(novelty_curves, axis=0)

        return combined_novelty

    def _detect_boundaries_adaptive(
        self, novelty: np.ndarray, notes: List[tuple]
    ) -> List[float]:
        """
        Detect boundaries using adaptive thresholding.

        Args:
            novelty: Novelty curve
            notes: Original notes for timing reference

        Returns:
            List of boundary times in seconds
        """
        # Adaptive threshold: use percentile of novelty values
        threshold = np.percentile(novelty, 75)

        # Find peaks above threshold
        peaks, properties = signal.find_peaks(
            novelty,
            height=threshold,
            distance=int(self.min_segment_duration / self.window_size),
        )

        # Convert peak indices to times
        boundaries = []
        for peak in peaks:
            time = peak * self.window_size
            boundaries.append(time)

        # Post-process: ensure minimum gap between boundaries
        boundaries = self._ensure_min_gap(boundaries, self.min_segment_duration)

        return boundaries

    def _ensure_min_gap(self, boundaries: List[float], min_gap: float) -> List[float]:
        """
        Ensure minimum gap between boundaries.

        If two boundaries are too close, keep the one with higher importance.

        Args:
            boundaries: List of boundary times
            min_gap: Minimum time between boundaries

        Returns:
            Filtered boundary times
        """
        if not boundaries:
            return []

        # Sort boundaries
        sorted_boundaries = sorted(boundaries)

        filtered = [sorted_boundaries[0]]

        for boundary in sorted_boundaries[1:]:
            # Check distance from last kept boundary
            if boundary - filtered[-1] >= min_gap:
                filtered.append(boundary)

        return filtered
