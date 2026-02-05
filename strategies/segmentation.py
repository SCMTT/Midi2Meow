"""
Segmentation strategies for m2m framework.

Implements pluggable segmentation algorithms using Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict
import numpy as np
import pretty_midi

# Handle imports for both package and direct execution modes
try:
    # Package mode: python -m m2m.strategies.segmentation
    from ..core.models import Section, SegmentationStrategy
except ImportError:
    # Direct mode: python app.py (from project root)
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from core.models import Section, SegmentationStrategy

# Import NOTE_MAP from references for adaptive segmentation
try:
    from references.converter import NOTE_MAP
except ImportError:
    # Fallback if references directory is not available
    # This is a simplified NOTE_MAP based on game piano constants
    NOTE_MAP = {
        48: "L1",
        50: "L2",
        52: "L3",
        53: "L4",
        55: "L5",
        57: "L6",
        59: "L7",
        60: "M1",
        62: "M2",
        64: "M3",
        65: "M4",
        67: "M5",
        69: "M6",
        71: "M7",
        72: "H1",
        74: "H2",
        76: "H3",
        77: "H4",
        79: "H5",
        81: "H6",
        83: "H7",
    }


class SegmentationStrategyABC(ABC):
    """Abstract base class for segmentation strategies."""

    def __init__(self, midi_data: pretty_midi.PrettyMIDI):
        """
        Initialize strategy with MIDI data.

        Args:
            midi_data: PrettyMIDI object to segment
        """
        self.midi_data = midi_data
        self._notes_cache = None

    @abstractmethod
    def split(self, **kwargs) -> List[Section]:
        """
        Split MIDI into sections.

        Args:
            **kwargs: Additional parameters (min_duration, melody_notes, etc.)

        Returns:
            List of Section objects
        """
        pass

    def _collect_notes(self) -> List[Tuple[float, float, int]]:
        """
        Collect all notes from MIDI data.

        Returns:
            List of (start, end, pitch) tuples
        """
        if self._notes_cache is not None:
            return self._notes_cache

        notes = []
        for instrument in self.midi_data.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                s = float(note.start)
                e = float(note.end) if note.end >= note.start else float(note.start)
                notes.append((s, e, int(note.pitch)))

        # Sort by start time, then end time
        notes.sort(key=lambda x: (x[0], x[1]))
        self._notes_cache = notes
        return notes

    def _get_midi_duration(self) -> float:
        """Get total duration of MIDI in seconds."""
        if not self.midi_data.instruments:
            return 0.0
        return max(
            max([note.end for note in instrument.notes], default=0)
            for instrument in self.midi_data.instruments
        )

    def _merge_short_sections(
        self, sections: List[Section], min_duration: float
    ) -> List[Section]:
        """Merge sections that are too short with neighbors."""
        if len(sections) <= 1:
            return sections

        merged = []
        i = 0

        while i < len(sections):
            current = sections[i]

            # If section is too short
            if current.duration < min_duration:
                # If not the last section, merge with next
                if i < len(sections) - 1:
                    next_section = sections[i + 1]
                    merged_section = Section(
                        start=current.start,
                        end=next_section.end,
                        reasons=list(set(current.reasons + next_section.reasons)),
                    )
                    # Replace next section in list with merged one
                    sections[i + 1] = merged_section
                    # Skip current, process next (now merged) in next iteration
                    i += 1

                # If it IS the last section, merge with previous (already in merged list)
                elif merged:
                    prev_section = merged[-1]
                    prev_section.end = current.end
                    prev_section.reasons = list(
                        set(prev_section.reasons + current.reasons)
                    )
                    i += 1
                else:
                    # Only one section left and it's short, keep it
                    merged.append(current)
                    i += 1
            else:
                merged.append(current)
                i += 1

        return merged


class BasicSegmentationStrategy(SegmentationStrategyABC):
    """
    Basic segmentation using tempo, key, and gap boundaries.

    Detects structural changes in the music:
    - Tempo changes
    - Time signature changes
    - Key signature changes
    - Significant gaps (>2.5s)
    """

    def split(
        self,
        melody_notes: List[Tuple[float, float, int]] | None = None,
        min_duration: float = 8.0,
        **kwargs,
    ) -> List[Section]:
        """
        Split MIDI at structural boundaries.

        Args:
            melody_notes: Optional list of melody notes (ignored in basic strategy)
            min_duration: Minimum segment duration in seconds
            **kwargs: Additional parameters

        Returns:
            List of Section objects
        """
        boundaries = self._detect_boundaries()
        total_duration = self._get_midi_duration()

        # Ensure we have start and end boundaries
        boundaries = [0.0] + boundaries + [total_duration]
        boundaries = sorted(set(boundaries))

        # Create sections
        sections = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]

            section = Section(
                start=start, end=end, reasons=self._get_boundary_reasons(start, end)
            )
            sections.append(section)

        # Merge very short sections with neighbors
        sections = self._merge_short_sections(sections, min_duration)

        return sections

    def _detect_boundaries(self) -> List[float]:
        """Detect structural boundaries in MIDI."""
        boundaries = []

        # Tempo changes
        for tempo_change in self.midi_data.get_tempo_changes():
            if tempo_change[0] > 0:  # Skip time 0
                boundaries.append(tempo_change[0])

        # Time signature changes
        for ts_change in self.midi_data.time_signature_changes:
            if ts_change.time > 0:
                boundaries.append(ts_change.time)

        # Significant gaps
        notes = self._collect_notes()
        if len(notes) > 1:
            gaps = []
            for i in range(len(notes) - 1):
                gap = notes[i + 1][0] - notes[i][1]  # Next start - current end
                if gap > 2.5:  # 2.5 second threshold
                    gaps.append((notes[i][1], gap))

            # Add boundaries at significant gaps
            for gap_start, gap_duration in gaps:
                boundaries.append(gap_start)

        return sorted(set(boundaries))

    def _get_boundary_reasons(self, start: float, end: float) -> List[str]:
        """Get reasons why this section was created."""
        reasons = []

        # Check for tempo changes
        for tempo_change in self.midi_data.get_tempo_changes():
            if abs(tempo_change[0] - start) < 0.1:
                reasons.append("速度变化")

        # Check for time signature changes
        for ts_change in self.midi_data.time_signature_changes:
            if abs(ts_change.time - start) < 0.1:
                reasons.append("拍号变化")

        # Check for gaps
        notes = self._collect_notes()
        for i in range(len(notes) - 1):
            gap_start = notes[i][1]
            gap = notes[i + 1][0] - gap_start
            if abs(gap_start - start) < 0.1 and gap > 2.5:
                reasons.append("显著空隙")

        return reasons if reasons else ["基础分段"]


# =============================================================================
# Helper functions from References (references/splitter.py)
# =============================================================================


def _collect_notes_ref(pm: pretty_midi.PrettyMIDI) -> List[Tuple[float, float, int]]:
    """Collect all notes from MIDI data (from References)."""
    out = []
    for inst in pm.instruments:
        if getattr(inst, "is_drum", False):
            continue
        for n in inst.notes:
            s = float(n.start)
            e = float(n.end) if n.end >= n.start else float(n.start)
            out.append((s, e, int(n.pitch)))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _candidate_boundaries(
    pm: pretty_midi.PrettyMIDI,
    notes: List[Tuple[float, float, int]],
    melody_only: bool = False,
    melody_threshold: int = 60,
) -> List[Tuple[float, str]]:
    """Get candidate boundaries for segmentation (from References)."""
    boundaries: List[Tuple[float, str]] = [(0.0, "开始")]
    tempos, tempo_times = pm.get_tempo_changes()
    for t in tempo_times:
        boundaries.append((float(t), "速度变化"))
    for ts in getattr(pm, "time_signature_changes", []):
        boundaries.append((float(ts.time), "拍号变化"))
    for ks in getattr(pm, "key_signature_changes", []):
        boundaries.append((float(ks.time), "调性变化"))
    gap_threshold = 2.5
    prev_end = 0.0
    for s, e, _ in notes:
        if s - prev_end > gap_threshold:
            boundaries.append((s, "显著空隙"))
        if e > prev_end:
            prev_end = e
    if prev_end > 0:
        boundaries.append((prev_end, "结束"))
    beats = list(pm.get_beats())
    if beats:
        snapped: List[Tuple[float, str]] = []
        for t, r in boundaries:
            bt = min(beats, key=lambda b: abs(b - t))
            if abs(bt - t) <= 0.3:
                snapped.append((bt, r))
            else:
                snapped.append((t, r))
        boundaries = snapped

    def best_t_in_window(ws: float, we: float) -> int | None:
        win_notes = [(s, e, p) for s, e, p in notes if s < we and e > ws]
        if melody_only:
            win_notes = [x for x in win_notes if x[2] >= melody_threshold]
        if len(win_notes) < 12:
            return None
        bt, _ = _best_transpose(
            win_notes,
            -27,
            27,
            melody_only=melody_only,
            melody_threshold=melody_threshold,
        )
        return bt

    if notes:
        start_time = notes[0][0]
        end_time = max(n[1] for n in notes)
        window = 8.0
        step = 4.0
        timeline: List[Tuple[float, int]] = []
        t = start_time + window / 2
        while t < end_time - window / 2:
            bt = best_t_in_window(t - window / 2, t + window / 2)
            if bt is not None:
                timeline.append((t, bt))
            t += step
        runs: List[Tuple[float, float, int]] = []
        for i, (tt, btt) in enumerate(timeline):
            if not runs:
                runs.append((tt, tt, btt))
            else:
                s0, e0, v0 = runs[-1]
                if btt == v0:
                    runs[-1] = (s0, tt, v0)
                else:
                    runs.append((tt, tt, btt))
        min_change_dur = 6.0
        filtered_runs: List[Tuple[float, float, int]] = []
        for r in runs:
            if r[1] - r[0] >= min_change_dur:
                filtered_runs.append(r)
        change_points = [r[0] for r in filtered_runs[1:]]
        for cp in change_points:
            if beats:
                bt = min(beats, key=lambda b: abs(b - cp))
                if abs(bt - cp) <= 0.3:
                    boundaries.append((bt, "移调变化"))
                else:
                    boundaries.append((cp, "移调变化"))
            else:
                boundaries.append((cp, "移调变化"))
    boundaries.sort(key=lambda x: x[0])
    merged: List[Tuple[float, str]] = []
    for t, r in boundaries:
        if not merged or abs(t - merged[-1][0]) > 0.5:
            merged.append((t, r))
    return merged


def _split_by_boundaries(
    notes: List[Tuple[float, float, int]],
    boundaries: List[Tuple[float, str]],
) -> List[Tuple[float, float, List[Tuple[float, float, int]], List[str]]]:
    """Split notes by boundaries (from References)."""
    if not notes:
        return []
    sections: List[Tuple[float, float, List[Tuple[float, float, int]], List[str]]] = []
    times = [t for t, _ in boundaries]
    reasons_map: Dict[float, List[str]] = {}
    for t, r in boundaries:
        reasons_map.setdefault(t, []).append(r)
    i = 0
    for bi in range(len(times) - 1):
        start = times[bi]
        end = times[bi + 1]
        seg_notes: List[Tuple[float, float, int]] = []
        while i < len(notes) and notes[i][0] < end:
            if notes[i][1] > start:
                seg_notes.append(notes[i])
            i += 1
        if seg_notes:
            sections.append((start, end, seg_notes, reasons_map.get(start, [])))
    min_duration = 8.0
    merged_sections: List[
        Tuple[float, float, List[Tuple[float, float, int]], List[str]]
    ] = []
    for sec in sections:
        if not merged_sections:
            merged_sections.append(sec)
            continue
        s, e, ns, rs = sec
        if e - s < min_duration:
            ps, pe, pns, prs = merged_sections[-1]
            merged_sections[-1] = (ps, e, pns + ns, prs)
        else:
            merged_sections.append(sec)
    return merged_sections


def _white_key_rate_for_notes(
    notes: List[Tuple[float, float, int]],
    transpose: int,
    melody_only: bool = False,
    melody_threshold: int = 60,
) -> Tuple[float, int, int]:
    """Calculate white key rate for notes (from References)."""
    playable = set(NOTE_MAP.keys())
    total = 0
    white = 0
    for _, _, pitch in notes:
        if melody_only and pitch < melody_threshold:
            continue
        total += 1
        if pitch + transpose in playable:
            white += 1
    rate = (white / total) if total > 0 else 0.0
    return rate, white, total


def _best_transpose(
    notes: List[Tuple[float, float, int]],
    semitone_min: int,
    semitone_max: int,
    melody_only: bool = False,
    melody_threshold: int = 60,
) -> Tuple[int, float]:
    """Find optimal transposition (from References)."""
    best_t = 0
    best_rate = -1.0
    for t in range(semitone_min, semitone_max + 1):
        rate, _, _ = _white_key_rate_for_notes(
            notes, t, melody_only=melody_only, melody_threshold=melody_threshold
        )
        if rate > best_rate or (abs(rate - best_rate) < 1e-9 and abs(t) < abs(best_t)):
            best_rate = rate
            best_t = t
    return best_t, best_rate


def _merge_sections_by_transpose(
    raw_sections: List[Tuple[float, float, List[Tuple[float, float, int]], List[str]]],
    semitone_min: int,
    semitone_max: int,
    melody_only: bool = False,
    melody_threshold: int = 60,
) -> List[Tuple[float, float, List[Tuple[float, float, int]], List[str]]]:
    """Merge sections by transposition (from References)."""
    locked = {
        "速度变化",
        "拍号变化",
        "调性变化",
        "显著空隙",
        "MusicFM变化",
        "MusicFM分段",
    }
    out: List[Tuple[float, float, List[Tuple[float, float, int]], List[str]]] = []
    prev_t = None
    for sec in raw_sections:
        s, e, ns, rs = sec
        t_best, _ = _best_transpose(
            ns,
            semitone_min,
            semitone_max,
            melody_only=melody_only,
            melody_threshold=melody_threshold,
        )
        if not out:
            out.append((s, e, ns, rs))
            prev_t = t_best
            continue
        ps, pe, pns, prs = out[-1]
        if prev_t == t_best and not any(r in locked for r in rs):
            out[-1] = (ps, e, pns + ns, prs)
        else:
            out.append((s, e, ns, rs))
        prev_t = t_best
    return out


def _notes_split(
    ns: List[Tuple[float, float, int]], t: float
) -> Tuple[List[Tuple[float, float, int]], List[Tuple[float, float, int]]]:
    """Split notes at a time point (from References)."""
    a = [x for x in ns if x[0] < t]
    b = [x for x in ns if x[0] >= t]
    return a, b


class AdaptiveSegmentationStrategy(SegmentationStrategyABC):
    """
    Adaptive segmentation using iterative white key rate optimization.

    This implementation is ported from references/splitter.py completely.
    It uses the same algorithm as the reference implementation.
    """

    def __init__(self, midi_data: pretty_midi.PrettyMIDI):
        super().__init__(midi_data)
        self.epsilon = 0.01  # Improvement threshold

    def split(
        self,
        melody_notes: List[Tuple[float, float, int]] | None = None,
        min_duration: float = 8.0,
        epsilon: float = 0.01,
        semitone_min: int = -27,
        semitone_max: int = 27,
        melody_only: bool = False,
        melody_threshold: int = 60,
        **kwargs,
    ) -> List[Section]:
        """
        Split MIDI using adaptive segmentation (from References).

        This is a complete port of split_midi_sections_adaptive from references/splitter.py.

        Args:
            melody_notes: Optional list of melody notes (ignored, melody filtering via params)
            min_duration: Minimum segment duration in seconds
            epsilon: White key rate improvement threshold
            semitone_min: Minimum transposition in semitones
            semitone_max: Maximum transposition in semitones
            melody_only: Whether to consider only melody notes
            melody_threshold: MIDI pitch threshold for melody notes
            **kwargs: Additional parameters

        Returns:
            List of Section objects with transposition info
        """
        if epsilon is not None:
            self.epsilon = epsilon

        # Use the exact algorithm from References
        pm = self.midi_data
        notes = _collect_notes_ref(pm)

        if not notes:
            return []

        # Step 1: Get candidate boundaries
        boundaries = _candidate_boundaries(
            pm, notes, melody_only=melody_only, melody_threshold=melody_threshold
        )

        # Step 2: Split by boundaries
        raw = _split_by_boundaries(notes, boundaries)

        # Step 3: Merge by transposition
        raw = _merge_sections_by_transpose(
            raw,
            semitone_min,
            semitone_max,
            melody_only=melody_only,
            melody_threshold=melody_threshold,
        )

        # Step 4: Adaptive iteration - find best split points
        total = len(notes)
        beats = list(pm.get_beats())

        def sec_best(
            ns: List[Tuple[float, float, int]],
        ) -> Tuple[int, float, int]:
            """Get best transposition and white key info for a section."""
            t, _ = _best_transpose(
                ns,
                semitone_min,
                semitone_max,
                melody_only=melody_only,
                melody_threshold=melody_threshold,
            )
            r, w, _ = _white_key_rate_for_notes(
                ns, t, melody_only=melody_only, melody_threshold=melody_threshold
            )
            return t, r, w

        sections = list(raw)
        improved = True

        while improved:
            improved = False
            best_gain = 0.0
            best_idx = None
            best_t = None

            for i, (s, e, ns, rs) in enumerate(sections):
                # Collect candidate split points: boundaries + beats
                cand = []
                for t, _r in boundaries:
                    if s + min_duration <= t <= e - min_duration:
                        cand.append(t)
                for b in beats:
                    if s + min_duration <= b <= e - min_duration:
                        cand.append(b)
                cand.sort()

                # Remove duplicates (within 0.25s)
                uniq = []
                for t in cand:
                    if not uniq or abs(t - uniq[-1]) > 0.25:
                        uniq.append(t)

                if not uniq:
                    continue

                # Evaluate whole section
                t_whole, r_whole, w_whole = sec_best(ns)

                # Try each candidate split point
                for t in uniq:
                    a, b = _notes_split(ns, t)
                    if len(a) < 6 or len(b) < 6:
                        continue
                    ta, ra, wa = sec_best(a)
                    tb, rb, wb = sec_best(b)

                    # Calculate gain: white key improvement normalized by total notes
                    gain = (wa + wb - w_whole) / total

                    # Skip if resulting sections would be too short
                    if (t - s) < min_duration or (e - t) < min_duration:
                        continue

                    # Track best gain
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                        best_t = t

            # Execute the best split if it exceeds epsilon
            if best_idx is not None and best_gain > self.epsilon:
                s, e, ns, rs = sections[best_idx]
                a, b = _notes_split(ns, best_t)
                sections = (
                    sections[:best_idx]
                    + [(s, best_t, a, rs)]
                    + [(best_t, e, b, rs)]
                    + sections[best_idx + 1 :]
                )
                improved = True

        # Convert to Section objects
        out: List[Section] = []
        for s, e, ns, rs in sections:
            t_best, r_best = _best_transpose(
                ns,
                semitone_min,
                semitone_max,
                melody_only=melody_only,
                melody_threshold=melody_threshold,
            )
            r0, _, _ = _white_key_rate_for_notes(
                ns, 0, melody_only=melody_only, melody_threshold=melody_threshold
            )
            section = Section(
                start=round(s, 3),
                end=round(e, 3),
                transpose=t_best,
                white_key_rate=r_best,
                reasons=rs,
            )
            out.append(section)

        return out


class SegmentationFactory:
    """Factory for creating segmentation strategies."""

    @staticmethod
    def create(
        strategy: SegmentationStrategy,
        midi_data: pretty_midi.PrettyMIDI,
        config: Optional["SSMProConfig"] = None,
    ) -> SegmentationStrategyABC:
        """
        Create a segmentation strategy instance.

        Args:
            strategy: SegmentationStrategy enum value
            midi_data: PrettyMIDI object
            config: Optional SSMProConfig for SSM-Pro strategy (ignored for other strategies)

        Returns:
            SegmentationStrategyABC instance

        Raises:
            ValueError: If strategy is not supported
        """
        if strategy == SegmentationStrategy.BASIC:
            return BasicSegmentationStrategy(midi_data)
        elif strategy == SegmentationStrategy.ADAPTIVE:
            return AdaptiveSegmentationStrategy(midi_data)
        elif strategy == SegmentationStrategy.SSM:
            # Import here to avoid circular dependency
            from .ssm_segmentation import SSMSegmentationStrategy

            return SSMSegmentationStrategy(midi_data)
        elif strategy == SegmentationStrategy.SSM_PRO:
            # Import here to avoid circular dependency
            from .ssm_pro_segmentation import SSMProSegmentationStrategy

            # Prepare feature weights if config is provided
            feature_weights = None
            if config is not None:
                feature_weights = config.to_feature_weights_dict()

            return SSMProSegmentationStrategy(
                midi_data, feature_weights=feature_weights
            )
        else:
            raise ValueError(
                f"Unknown strategy: {strategy.value}. "
                f"Supported strategies: 'basic', 'adaptive', 'ssm', 'ssm_pro'."
            )
