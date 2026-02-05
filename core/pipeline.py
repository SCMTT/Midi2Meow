"""
M2M Pipeline - Main orchestrator for MIDI→MIDI conversion.

This pipeline coordinates:
1. MIDI file loading
2. Melody extraction (optional)
3. Segmentation (pluggable strategies)
4. Per-segment transposition optimization
5. MIDI file output
"""

from typing import List, Optional, cast
import pretty_midi

# Handle imports for both package and direct execution modes
try:
    # Package mode: python -m m2m.core.pipeline
    from .models import Section, M2MConfig, SegmentationStrategy
    from .observer import (
        Observable,
        Event,
        EventType,
        ProgressTracker,
        FunctionObserver,
    )
except ImportError:
    # Direct mode: python app.py (from project root)
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from core.models import Section, M2MConfig, SegmentationStrategy
    from core.observer import (
        Observable,
        Event,
        EventType,
        ProgressTracker,
        FunctionObserver,
    )

# Import game piano constants for playable range checking
try:
    # Package mode
    from ..strategies.game_piano_constants import is_playable_pitch

    GAME_PIANO_MODE = True
except ImportError:
    try:
        # Direct mode (core already added to path in imports above)
        from strategies.game_piano_constants import is_playable_pitch

        GAME_PIANO_MODE = True
    except ImportError:
        GAME_PIANO_MODE = False
        is_playable_pitch = None


class M2MPipeline(Observable):
    """
    Main pipeline for MIDI→MIDI conversion with segmentation and transposition.

    This class orchestrates the entire conversion process and notifies observers
    of progress and events.

    Usage:
        config = M2MConfig()
        pipeline = M2MPipeline(config)
        pipeline.attach(gui_observer)

        # Process MIDI file
        sections = pipeline.process("input.mid", "output.mid")
    """

    def __init__(self, config: M2MConfig):
        """
        Initialize pipeline with configuration.

        Args:
            config: M2MConfig object with pipeline settings
        """
        super().__init__()
        self.config = config
        self.progress = ProgressTracker()
        self.progress.attach(FunctionObserver(self._forward_progress))

        # These will be initialized lazily
        self._segmentation_strategy = None
        self._transposition_optimizer = None

        # Store results
        self.midi_data: Optional[pretty_midi.PrettyMIDI] = None
        self.sections: List[Section] = []

    def _forward_progress(self, event: Event) -> None:
        """Forward progress events to our observers."""
        self.notify(event)

    def process(self, input_midi: str, output_midi: str) -> List[Section]:
        """
        Process MIDI file through the complete pipeline (Legacy wrapper).
        """
        self.analyze(input_midi)
        self.export(output_midi)
        return self.sections

    def analyze(self, input_midi: str) -> List[Section]:
        """
        Run analysis phase: Load -> Melody -> Segment -> Optimize.
        Does NOT write output file.
        """
        try:
            self.progress.start("MIDI Analysis", total_steps=4)

            # Step 1: Load MIDI
            self.progress.update(1, "Loading MIDI file...")
            self.midi_data = self._load_midi(input_midi)

            # Step 2: Skip melody extraction
            self.progress.update(2, "Skipping melody extraction")

            # Step 3: Segment
            self.progress.update(
                3, f"Segmenting ({self.config.segmentation_strategy.value})..."
            )
            self.sections = self._segment_midi()

            # Step 4: Transpose
            self.progress.update(4, "Optimizing transposition...")
            self._optimize_transposition()

            self.progress.complete("Analysis complete!")

            # Notify observers of segmentation results
            self.notify(
                Event(
                    type=EventType.SEGMENTATION_COMPLETE,
                    data={"sections": self.sections},
                    message=f"Created {len(self.sections)} segments",
                )
            )

            return self.sections

        except Exception as e:
            self.progress.error(f"Error: {str(e)}")
            raise

    def export(
        self, output_midi: str, sections: Optional[List[Section]] = None
    ) -> None:
        """
        Run export phase: Write output MIDI using provided sections.

        Args:
            output_midi: Path to output file
            sections: Optional list of sections to use. If None, uses self.sections.
        """
        try:
            self.progress.start("Exporting MIDI", total_steps=1)
            self.progress.update(1, "Writing output MIDI...")

            # Use provided sections if available (allows GUI to override with manual edits)
            sections_to_use = sections if sections is not None else self.sections

            self._write_midi(output_midi, sections_to_use)
            self.progress.complete("Export complete!")

        except Exception as e:
            self.progress.error(f"Error: {str(e)}")
            raise

    def _load_midi(self, midi_path: str) -> pretty_midi.PrettyMIDI:
        """
        Load MIDI file.

        Args:
            midi_path: Path to MIDI file

        Returns:
            PrettyMIDI object
        """
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            return midi
        except Exception as e:
            self.progress.error(f"Error: {str(e)}")
            raise

    def _segment_midi(self) -> List[Section]:
        """
        Segment MIDI into sections.

        Uses the configured segmentation strategy.

        Returns:
            List of Section objects
        """
        # Import segmentation strategy
        try:
            from ..strategies.segmentation import SegmentationFactory
        except ImportError:
            from strategies.segmentation import SegmentationFactory

        # Ensure MIDI data is loaded
        if self.midi_data is None:
            raise ValueError("MIDI data not loaded. Call _load_midi first.")

        midi_data = cast(pretty_midi.PrettyMIDI, self.midi_data)

        if self._segmentation_strategy is None:
            # Pass SSM-Pro config if using SSM-Pro strategy
            ssm_pro_config = None
            if self.config.segmentation_strategy == SegmentationStrategy.SSM_PRO:
                ssm_pro_config = self.config.ssm_pro_config

            self._segmentation_strategy = SegmentationFactory.create(
                self.config.segmentation_strategy, midi_data, config=ssm_pro_config
            )

        # Perform segmentation
        sections = self._segmentation_strategy.split(
            min_duration=self.config.min_segment_duration,
        )

        return sections

    def _optimize_transposition(self) -> None:
        """
        Optimize transposition for each section.

        Uses the configured transposition strategy.
        """
        # Import transposition optimizer
        try:
            from ..strategies.transposition import TranspositionFactory
        except ImportError:
            from strategies.transposition import TranspositionFactory

        # Ensure MIDI data is loaded
        if self.midi_data is None:
            raise ValueError("MIDI data not loaded. Call _load_midi first.")

        midi_data = cast(pretty_midi.PrettyMIDI, self.midi_data)

        if self._transposition_optimizer is None:
            self._transposition_optimizer = TranspositionFactory.create(
                self.config.transposition_config.strategy
            )

        # Collect all notes grouped by section
        all_notes = []
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                all_notes.append((note.start, note.end, note.pitch))

        # Optimize each section
        for section in self.sections:
            # Get notes in this section
            section_notes = [
                (s, e, p) for s, e, p in all_notes if section.start <= s < section.end
            ]

            if not section_notes:
                continue

            # Optimize transposition
            best_transpose, white_rate = self._transposition_optimizer.optimize(
                notes=section_notes, config=self.config.transposition_config
            )

            section.transpose = int(best_transpose)
            section.white_key_rate = white_rate
            section.note_count = len(section_notes)
            section.melody_threshold = self.config.transposition_config.melody_threshold

    def _write_midi(
        self, output_path: str, sections: Optional[List[Section]] = None
    ) -> None:
        """
        Write transposed MIDI to file.

        Args:
            output_path: Path to output MIDI file
            sections: Optional sections to use. If None, uses self.sections.
        """
        # Ensure MIDI data is loaded
        if self.midi_data is None:
            raise ValueError("MIDI data not loaded. Call _load_midi first.")

        # Use provided sections or internal state
        active_sections = sections if sections is not None else self.sections

        midi_data = cast(pretty_midi.PrettyMIDI, self.midi_data)

        # Get initial tempo from MIDI
        tempo_changes = midi_data.get_tempo_changes()
        if tempo_changes and len(tempo_changes[0]) > 0 and len(tempo_changes[1]) > 0:
            initial_tempo = tempo_changes[1][0]
        else:
            initial_tempo = 120.0

        # Create new MIDI with transposed notes
        output_midi = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)

        # Copy time signature changes
        output_midi.time_signature_changes = list(midi_data.time_signature_changes)

        # Process each instrument
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                # Drums don't get transposed
                output_midi.instruments.append(instrument)
                continue

            # Create new instrument with same program
            new_instrument = pretty_midi.Instrument(
                program=instrument.program, is_drum=False, name=instrument.name
            )

            # Apply transposition per section
            notes_discarded = 0
            notes_kept = 0

            for note in instrument.notes:
                # Find which section this note belongs to
                transposition = 0
                berserk_mode = False
                max_pitch = 83  # B5 (Game piano max pitch)
                melody_threshold = 60

                for section in active_sections:
                    if section.start <= note.start < section.end:
                        transposition = section.transpose
                        berserk_mode = section.berserk_mode
                        max_pitch = section.max_pitch
                        melody_threshold = section.melody_threshold
                        break

                # Apply transposition
                new_pitch = note.pitch + transposition

                # Apply Berserk Mode (Octave Folding for High Notes)
                if berserk_mode:
                    # Only affect notes above threshold (Right hand)
                    if note.pitch >= melody_threshold:
                        # While pitch exceeds max, fold down by octaves
                        while new_pitch > max_pitch:
                            new_pitch -= 12

                # Clamp to MIDI range
                final_pitch = max(0, min(127, new_pitch))

                # GAME PIANO MODE: Check if note is playable
                # Discard notes outside game piano range (48-83)
                if GAME_PIANO_MODE and is_playable_pitch is not None:
                    if not is_playable_pitch(final_pitch):
                        notes_discarded += 1
                        continue  # Skip this note

                # Add transposed note
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=final_pitch,
                    start=note.start,
                    end=note.end,
                )
                new_instrument.notes.append(new_note)
                notes_kept += 1

            # Log statistics for game piano mode
            if GAME_PIANO_MODE and notes_discarded > 0:
                total = notes_discarded + notes_kept
                rate = (notes_kept / total * 100) if total > 0 else 0
                print(
                    f"[Game Piano] Instrument '{instrument.name}': "
                    f"{notes_kept}/{total} notes kept ({rate:.1f}%), "
                    f"{notes_discarded} discarded (out of range)"
                )

            output_midi.instruments.append(new_instrument)

        # Write to file
        output_midi.write(output_path)
