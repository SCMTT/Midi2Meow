"""
Enhanced GUI for m2m framework with waveform visualization.

Features:
- Waveform visualization (note density histogram for now, can be upgraded to audio)
- Drag to create/move/delete segments
- Zoom and pan controls
- Undo/Redo support
- Real-time preview
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from typing import List, Optional, Callable
import copy
import numpy as np
import pretty_midi

# Handle imports for both package and direct execution modes
try:
    # Package mode: python -m m2m.gui.main
    from ..core.models import Section, M2MConfig, SegmentationStrategy
    from ..core.observer import Observer, Event, EventType
    from ..core.pipeline import M2MPipeline
except ImportError:
    # Direct mode: python app.py (from project root)
    import sys
    from pathlib import Path

    # Add project root to path for absolute imports
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from core.models import Section, M2MConfig, SegmentationStrategy
    from core.observer import Observer, Event, EventType
    from core.pipeline import M2MPipeline


class SegmentSettingsDialog(tk.Toplevel):
    """
    Dialog for configuring individual segment settings.
    """

    def __init__(
        self,
        parent,
        section: Section,
        min_transpose: int,
        max_transpose: int,
        enable_threshold: bool,
    ):
        super().__init__(parent)
        self.title("分段设置")
        self.geometry("350x320")
        self.resizable(False, False)

        self.section = section
        self.result = None

        # Center the dialog
        self.transient(parent)
        self.grab_set()

        # UI
        padding = 10
        frame = ttk.Frame(self, padding=padding)
        frame.pack(fill=tk.BOTH, expand=True)

        # Transpose
        ttk.Label(frame, text="移调 (半音):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.transpose_var = tk.IntVar(value=section.transpose)
        ttk.Spinbox(
            frame,
            from_=min_transpose,
            to=max_transpose,
            textvariable=self.transpose_var,
            width=10,
        ).grid(row=0, column=1, sticky=tk.W, pady=5)

        # Melody Threshold
        ttk.Label(frame, text="主旋律阈值:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.threshold_var = tk.IntVar(value=section.melody_threshold)
        threshold_spin = ttk.Spinbox(
            frame, from_=0, to=127, textvariable=self.threshold_var, width=10
        )
        threshold_spin.grid(row=1, column=1, sticky=tk.W, pady=5)

        if not enable_threshold:
            threshold_spin.configure(state="disabled")
            ttk.Label(frame, text="(全局未启用)", foreground="gray").grid(
                row=1, column=2, sticky=tk.W, padx=5
            )

        # Berserk Mode Separator
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(
            row=2, column=0, columnspan=3, sticky="ew", pady=15
        )

        # Berserk Mode Checkbox
        self.berserk_var = tk.BooleanVar(value=section.berserk_mode)
        ttk.Checkbutton(
            frame,
            text="开启狂暴模式 (Berserk Mode)",
            variable=self.berserk_var,
            command=self._toggle_berserk,
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(
            frame, text="强制折叠高音到范围内", font=("", 8), foreground="red"
        ).grid(row=3, column=2, sticky=tk.W)

        # Max Pitch (Highest White Key)
        ttk.Label(frame, text="最高白键音高:").grid(
            row=4, column=0, sticky=tk.W, pady=5
        )
        self.max_pitch_var = tk.IntVar(value=section.max_pitch)
        self.max_pitch_spin = ttk.Spinbox(
            frame, from_=0, to=127, textvariable=self.max_pitch_var, width=10
        )
        self.max_pitch_spin.grid(row=4, column=1, sticky=tk.W, pady=5)
        ttk.Label(
            frame, text="(默认 83 = B5 游戏钢琴最高音)", font=("", 8), foreground="gray"
        ).grid(row=4, column=2, sticky=tk.W, padx=5)

        # Initialize state
        self._toggle_berserk()

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=5, column=0, columnspan=3, pady=20)

        ttk.Button(btn_frame, text="确定", command=self._on_ok).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(
            side=tk.LEFT, padx=5
        )

    def _toggle_berserk(self):
        """Enable/disable max pitch input based on berserk mode."""
        if self.berserk_var.get():
            self.max_pitch_spin.configure(state="normal")
        else:
            self.max_pitch_spin.configure(state="disabled")

    def _on_ok(self):
        self.result = {
            "transpose": self.transpose_var.get(),
            "melody_threshold": self.threshold_var.get(),
            "berserk_mode": self.berserk_var.get(),
            "max_pitch": self.max_pitch_var.get(),
        }
        self.destroy()

    def _calculate_preview_stats(self):
        """Calculate and show preview stats in dialog title or label."""
        # This could be implemented to show real-time stats in the dialog
        pass


class WaveformCanvas(tk.Canvas):
    """
    Enhanced canvas for waveform visualization and segment manipulation.

    Features:
    - Display note density as waveform-like bars
    - Drag segment boundaries
    - Click to add new segments
    - Right-click to delete segments
    - Zoom and pan
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.sections: List[Section] = []
        self.total_duration: float = 0.0

        # MIDI data for visualization
        self.midi_data: Optional[pretty_midi.PrettyMIDI] = None
        self.note_density: Optional[np.ndarray] = None  # Density array
        self.density_bin_size: float = 0.1  # 100ms per bin

        # Display mode: True = show segments, False = show raw density only
        self.show_segments: bool = False

        # Zoom and pan state
        self.zoom_level: float = 1.0  # 1.0 = fit to width
        self.pan_offset: float = 0.0  # pixels
        self.dragging: bool = False
        self.drag_start_x: int = 0
        self.selected_segment: Optional[Section] = None
        self.drag_action: Optional[str] = None  # 'start', 'end', or None

        # Visual constants
        self.width = 820
        self.height = 240
        self.configure(width=self.width, height=self.height)

        # Bind events
        self.bind("<Configure>", self._on_resize)
        self.bind("<ButtonPress-1>", self._on_mouse_down)
        self.bind("<B1-Motion>", self._on_mouse_drag)
        self.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.bind("<Button-3>", self._on_right_click)  # Right-click to delete
        self.bind("<Motion>", self._on_mouse_move)

        # Undo/Redo stacks
        self.undo_stack: List[List[Section]] = []
        self.redo_stack: List[List[Section]] = []

        # Callbacks
        self.on_segments_changed: Optional[Callable] = None
        self.on_optimize_segment: Optional[Callable] = None
        self.on_set_transposition: Optional[Callable] = None

        # Tooltip state
        self.tooltip_text = None
        self.tooltip_rect = None
        self.hovered_section = None
        self.enable_threshold_display = True  # Updated from parent

    def _on_resize(self, event):
        """Handle canvas resize."""
        if self.width != event.width or self.height != event.height:
            self.width = event.width
            self.height = event.height
            self.redraw()

    def set_sections(
        self,
        sections: List[Section],
        duration: float,
        midi_data: Optional[pretty_midi.PrettyMIDI] = None,
    ):
        """
        Update sections to display.

        Args:
            sections: List of Section objects
            duration: Total duration in seconds
            midi_data: Optional MIDI data for visualization
        """
        # Save to undo stack before changing
        if self.sections and self.sections != sections:
            self.undo_stack.append(copy.deepcopy(self.sections))
            self.redo_stack.clear()

        self.sections = sections
        self.total_duration = duration
        self.show_segments = True  # Enable segment display

        # Extract and cache note density if MIDI data provided
        if midi_data:
            self.midi_data = midi_data
            self._calculate_note_density()

        self.redraw()

    def set_midi_only(self, midi_data: pretty_midi.PrettyMIDI):
        """
        Display raw MIDI note density without segments.

        Args:
            midi_data: PrettyMIDI object to visualize
        """
        self.midi_data = midi_data
        self.total_duration = midi_data.get_end_time()
        self.sections = []  # Clear sections
        self.show_segments = False  # Don't show segment boundaries

        # Calculate note density
        self._calculate_note_density()

        # Redraw with raw density only
        self.redraw()

    def _calculate_note_density(self):
        """
        Calculate note density histogram from MIDI data.

        Creates a time-series of note counts in bins for visualization.
        """
        if self.midi_data is None:
            return

        # Calculate number of bins
        num_bins = int(self.total_duration / self.density_bin_size) + 1
        self.note_density = np.zeros(num_bins, dtype=int)

        # Collect all notes
        for instrument in self.midi_data.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                # Find which bin this note falls into
                bin_idx = int(note.start / self.density_bin_size)
                if 0 <= bin_idx < num_bins:
                    self.note_density[bin_idx] += 1

    def _draw_note_density(self):
        """
        Draw note density as semi-transparent bars.

        Higher density = taller and darker bars.
        """
        if self.note_density is None:
            return

        # Find max density for normalization
        max_density = np.max(self.note_density)
        if max_density == 0:
            return

        # Calculate bar width
        bar_width = self.width / len(self.note_density)

        # Draw each density bar
        for i, density in enumerate(self.note_density):
            if density == 0:
                continue

            # Calculate bar height (normalize to canvas height)
            # Use taller bars (90% height) for better visibility
            bar_height = (density / max_density) * (self.height * 0.9)

            # Calculate position
            x1 = i * bar_width
            x2 = x1 + bar_width - 1  # -1 for small gap between bars
            y2 = self.height
            y1 = y2 - bar_height

            # Color based on density (darker blue for better visibility on light background)
            # Use darker colors to contrast with segment backgrounds
            intensity = min(255, 100 + int((density / max_density) * 155))
            # Dark blue: #000080 to #4040ff
            blue_val = min(255, 128 + int((density / max_density) * 127))
            color = f"#2020{blue_val:02x}"

            # Draw density bar
            self.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=color,
                outline="",
                # No stipple for bars - make them solid and clear
            )

    def redraw(self):
        """Redraw the canvas with current sections or raw density."""
        self.delete("all")

        # Draw background
        self.create_rectangle(0, 0, self.width, self.height, fill="#f8f9fa", outline="")

        # If no MIDI data loaded, show empty state
        if self.midi_data is None:
            self.create_text(
                self.width / 2,
                self.height / 2,
                text="请先选择输入 MIDI 文件",
                fill="#999999",
                font=("Arial", 14),
            )
            return

        # 1. Draw segments background FIRST (so they appear behind waveforms)
        if self.show_segments and self.sections:
            for i, section in enumerate(self.sections):
                x1 = self._time_to_x(section.start)
                x2 = self._time_to_x(section.end)

                # Color based on transposition
                transposition = section.transpose
                if transposition == 0:
                    color = "#e0e0e0"  # Grey - no transposition
                elif transposition > 0:
                    # Light Green shades for positive transposition
                    intensity = int(max(0, 240 - transposition * 10))
                    color = f"#{intensity:02x}ff{intensity:02x}"
                else:
                    # Light Red shades for negative transposition
                    intensity = int(max(0, 240 - abs(transposition) * 10))
                    color = f"#ff{intensity:02x}{intensity:02x}"

                # Highlight selected segment
                if section == self.selected_segment:
                    outline = "#ff0000"
                    width = 2
                    bg_stipple = "gray50"  # Darker for selection
                else:
                    outline = ""
                    width = 0
                    bg_stipple = "gray25"  # Lighter for normal

                # Draw segment background rectangle
                # Use stipple to make it look semi-transparent
                self.create_rectangle(
                    x1,
                    0,  # Full height
                    x2,
                    self.height,
                    fill=color,
                    outline=outline,
                    width=width,
                    tags=f"segment_{i}",
                    stipple=bg_stipple,
                )

        # 2. Draw note density visualization (OVER background)
        self._draw_note_density()

        # 3. Draw time grid
        self._draw_time_grid()

        # 4. Draw segment boundaries and info (Top Layer)
        if self.show_segments and self.sections:
            for i, section in enumerate(self.sections):
                x1 = self._time_to_x(section.start)
                x2 = self._time_to_x(section.end)

                # Draw segment info text (transposition)
                mid_x = (x1 + x2) / 2
                info_text = f"{int(section.transpose):+d}"

                # Add text background for readability
                self.create_rectangle(
                    mid_x - 15,
                    self.height / 2 - 10,
                    mid_x + 15,
                    self.height / 2 + 10,
                    fill="white",
                    outline="#dddddd",
                    tags=f"info_bg_{i}",
                )

                self.create_text(
                    mid_x,
                    self.height / 2,
                    text=info_text,
                    fill="#333333",
                    font=("Arial", 12, "bold"),
                    tags=f"info_text_{i}",
                )

            # Draw segment boundaries
            for i, section in enumerate(self.sections):
                x = self._time_to_x(section.start)

                # Highlight if hovering (check if mouse is near this boundary)
                is_hovered = False
                # We'll track hover state in mouse event handlers

                # Draw boundary line
                line_width = 2
                line_color = "#d9534f"

                self.create_line(
                    x,
                    0,
                    x,
                    self.height,
                    fill=line_color,
                    width=line_width,
                    dash=(4, 2),
                    tags=f"boundary_{section.start}",
                )

                # Draw arrow indicator at top
                self.create_polygon(
                    x,
                    0,
                    x - 6,
                    10,
                    x + 6,
                    10,
                    fill=line_color,
                    outline="",
                    tags=f"boundary_arrow_{section.start}",
                )

                # Draw time label
                minutes = int(section.start // 60)
                seconds = int(section.start % 60)
                time_label = f"{minutes:02d}:{seconds:02d}"

                # Background for label
                self.create_rectangle(
                    x - 22,
                    12,
                    x + 22,
                    26,
                    fill="white",
                    outline="#cccccc",
                    tags=f"boundary_label_bg_{section.start}",
                )

                # Label text
                self.create_text(
                    x,
                    19,
                    text=time_label,
                    fill="#333333",
                    font=("Arial", 9, "bold"),
                    tags=f"boundary_label_{section.start}",
                )

        # Draw status message
        if not self.show_segments:
            self.create_text(
                self.width / 2,
                15,
                text='原始MIDI音符密度 - 点击"处理MIDI"开始分段',
                fill="#666666",
                font=("Arial", 10),
            )

    def _draw_time_grid(self):
        """Draw time grid lines and labels."""
        # Calculate appropriate tick spacing
        duration = self.total_duration * self.zoom_level
        if duration <= 30:
            tick_interval = 5  # 5 seconds
        elif duration <= 120:
            tick_interval = 15  # 15 seconds
        else:
            tick_interval = 30  # 30 seconds

        # Draw vertical grid lines
        for t in np.arange(0, self.total_duration, tick_interval):
            x = self._time_to_x(float(t))
            self.create_line(x, 0, x, self.height, fill="#cccccc", dash=(2, 2))

            # Time label
            minutes = int(t // 60)
            seconds = int(t % 60)
            label = f"{minutes:02d}:{seconds:02d}"
            self.create_text(
                x + 2,
                self.height - 5,
                text=label,
                anchor="sw",
                fill="#666666",
                font=("Arial", 8),
            )

    def _time_to_x(self, time: float) -> float:
        """Convert time in seconds to x coordinate."""
        if self.total_duration == 0:
            return 0

        pixels_per_second = self.width / self.total_duration
        x = time * pixels_per_second * self.zoom_level + self.pan_offset
        return max(0, min(self.width, x))

    def _x_to_time(self, x: float) -> float:
        """Convert x coordinate to time in seconds."""
        pixels_per_second = self.width / self.total_duration
        time = (x - self.pan_offset) / (pixels_per_second * self.zoom_level)
        return max(0, min(self.total_duration, time))

    def _on_mouse_move(self, event):
        """Handle mouse movement for tooltips."""
        if not self.sections or self.dragging:
            self._hide_tooltip()
            return

        time = self._x_to_time(event.x)
        hovered = None

        for section in self.sections:
            if section.start <= time <= section.end:
                hovered = section
                break

        if hovered != self.hovered_section:
            self.hovered_section = hovered
            self.redraw()  # Redraw to show hover effect

        if hovered:
            self._show_tooltip(event.x, event.y, hovered)
        else:
            self._hide_tooltip()

    def _show_tooltip(self, x, y, section):
        """Show tooltip for section."""
        self._hide_tooltip()

        # Calculate global rate (all notes) for comparison
        global_rate_text = ""
        try:
            if self.midi_data:
                # Import here to avoid circular dependency if needed, or assume it's available
                try:
                    from ..strategies.transposition import WhiteKeyRateOptimizer
                except ImportError:
                    from strategies.transposition import WhiteKeyRateOptimizer

                # Get notes for this section
                # Note: This might be slightly slow for very large files, but usually fine for UI
                # Optimization: Could cache this in the section object
                section_notes = []
                for instrument in self.midi_data.instruments:
                    if instrument.is_drum:
                        continue
                    for note in instrument.notes:
                        if section.start <= note.start < section.end:
                            section_notes.append((note.start, note.end, note.pitch))

                if section_notes:
                    # Calculate rate with NO threshold (include everything)
                    global_rate = WhiteKeyRateOptimizer._white_key_rate(
                        section_notes,
                        section.transpose,
                        melody_threshold=0,
                        enable_threshold=False,
                    )
                    global_rate_text = f"\n全体白键率: {global_rate * 100:.1f}%"
        except Exception:
            # Log error but don't crash tooltip display
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"Tooltip calculation error: {e}")

        # Prepare text
        text = f"移调: {section.transpose:+d}\n主旋律白键率: {section.white_key_rate * 100:.1f}%{global_rate_text}"
        if self.enable_threshold_display:
            text += f"\n主旋律阈值: {section.melody_threshold}"

        if section.berserk_mode:
            text += f"\n(狂暴模式已开启 | Max: {section.max_pitch})"

        # Create tooltip
        # Draw background
        text_id = self.create_text(
            x + 15,
            y + 15,
            text=text,
            anchor="nw",
            font=("Arial", 9),
            fill="#333",
            tags="tooltip",
        )
        bbox = self.bbox(text_id)

        # Padding
        pad = 5
        rect_id = self.create_rectangle(
            bbox[0] - pad,
            bbox[1] - pad,
            bbox[2] + pad,
            bbox[3] + pad,
            fill="#ffffe0",
            outline="#333",
            tags="tooltip_bg",
        )

        # Raise text above rect
        self.tag_raise("tooltip")
        self.tooltip_text = text_id
        self.tooltip_rect = rect_id

    def _hide_tooltip(self):
        """Hide tooltip."""
        self.delete("tooltip")
        self.delete("tooltip_bg")
        self.tooltip_text = None
        self.tooltip_rect = None

    def _on_mouse_down(self, event):
        """Handle mouse click."""
        self.dragging = True
        self.drag_start_x = event.x
        self.drag_action = None

        # Check if clicking near a start boundary
        for section in self.sections:
            boundary_x = self._time_to_x(section.start)
            if abs(event.x - boundary_x) < 8:  # 8 pixel snap radius
                self.selected_segment = section
                self.drag_action = "start"
                return

        # Check if clicking near an end boundary
        for section in self.sections:
            boundary_x = self._time_to_x(section.end)
            if abs(event.x - boundary_x) < 8:
                self.selected_segment = section
                self.drag_action = "end"
                return

        # Check if clicking inside a segment to select it
        for section in self.sections:
            x1 = self._time_to_x(section.start)
            x2 = self._time_to_x(section.end)
            if x1 <= event.x <= x2:
                self.selected_segment = section
                self.redraw()
                return

        # Clicked in empty space - deselect
        self.selected_segment = None
        self.redraw()

    def _on_mouse_drag(self, event):
        """Handle mouse drag."""
        if not self.dragging or self.selected_segment is None or not self.drag_action:
            return

        # Calculate new time
        new_time = self._x_to_time(event.x)

        try:
            idx = self.sections.index(self.selected_segment)
        except ValueError:
            return

        if self.drag_action == "start":
            # Moving start boundary
            # Constrain to previous segment start or 0
            min_t = 0.0
            if idx > 0:
                min_t = self.sections[idx - 1].start + 0.1  # Minimum 0.1s segment

            # Constrain to current segment end
            max_t = self.selected_segment.end - 0.1

            new_time = max(min_t, min(new_time, max_t))

            # Update
            old_start = self.selected_segment.start
            self.selected_segment.start = new_time

            # If contiguous with previous, update previous end
            if idx > 0 and abs(self.sections[idx - 1].end - old_start) < 0.001:
                self.sections[idx - 1].end = new_time

        elif self.drag_action == "end":
            # Moving end boundary
            # Constrain to current segment start
            min_t = self.selected_segment.start + 0.1

            # Constrain to next segment end or total duration
            max_t = self.total_duration
            if idx < len(self.sections) - 1:
                max_t = self.sections[idx + 1].end - 0.1

            new_time = max(min_t, min(new_time, max_t))

            # Update
            old_end = self.selected_segment.end
            self.selected_segment.end = new_time

            # If contiguous with next, update next start
            if (
                idx < len(self.sections) - 1
                and abs(self.sections[idx + 1].start - old_end) < 0.001
            ):
                self.sections[idx + 1].start = new_time

        self.redraw()

    def _on_mouse_up(self, event):
        """Handle mouse release."""
        if self.dragging:
            self.dragging = False

            # Notify of change
            if self.on_segments_changed:
                self.on_segments_changed(self.sections)

    def _on_right_click(self, event):
        """Handle right-click to show context menu or add/delete segment."""
        click_time = self._x_to_time(event.x)

        # Check if clicking on a segment (not boundary)
        clicked_section = None
        for section in self.sections:
            if section.start <= click_time <= section.end:
                clicked_section = section
                break

        # Create popup menu
        menu = tk.Menu(self, tearoff=0)

        # Check if clicking on a boundary to delete it
        boundary_clicked = False
        for i, section in enumerate(self.sections):
            # Use pixel distance for easier clicking (consistent with drag)
            boundary_x = self._time_to_x(section.start)
            if abs(event.x - boundary_x) < 8:  # 8 pixel snap radius
                if 0 < i < len(self.sections) - 1:  # Can't delete first/last
                    boundary_clicked = True
                    menu.add_command(
                        label="删除此分界线 (Merge)",
                        command=lambda: self._delete_boundary(i),
                    )
                    break

        if not boundary_clicked and clicked_section:
            # Segment context menu
            menu.add_command(
                label="优化此段移调 (Auto Optimize)",
                command=lambda: self.on_optimize_segment(clicked_section)
                if self.on_optimize_segment
                else None,
            )
            menu.add_command(
                label="详细设置 (Settings)...",
                command=lambda: self.on_set_transposition(clicked_section)
                if self.on_set_transposition
                else None,
            )

            # Add merge option if not the first segment
            idx = self.sections.index(clicked_section)
            if idx > 0:
                menu.add_command(
                    label="与前一段合并 (Merge Previous)",
                    command=lambda: self._delete_boundary(idx),
                )

            menu.add_separator()

            # Add split option
            menu.add_command(
                label="在此处分割 (Split Here)",
                command=lambda: self._split_segment(click_time),
            )

        if not boundary_clicked and not clicked_section:
            # Empty space - just split if valid
            if 0 < click_time < self.total_duration:
                menu.add_command(
                    label="在此处添加分界线 (Add Boundary)",
                    command=lambda: self._split_segment(click_time),
                )

        # Show menu
        if menu.index("end") is not None:  # Only show if items exist
            menu.post(event.x_root, event.y_root)

    def _delete_boundary(self, index):
        """Delete boundary at index i (merge section i-1 and i)."""
        if not (0 < index < len(self.sections)):
            return

        # Save to undo
        self.undo_stack.append(copy.deepcopy(self.sections))
        self.redo_stack.clear()

        # Merge with neighboring segment
        prev_section = self.sections[index - 1]
        next_section = self.sections[index]  # Note: index is the start of this section

        # Remove current section and extend previous
        prev_section.end = next_section.end
        self.sections.pop(index)

        self.redraw()
        if self.on_segments_changed:
            self.on_segments_changed(self.sections)

    def _split_segment(self, split_time):
        """Split segment at given time."""
        if not (0 < split_time < self.total_duration):
            return

        # Save to undo
        self.undo_stack.append(copy.deepcopy(self.sections))
        self.redo_stack.clear()

        # Find which segment to split
        for i, section in enumerate(self.sections):
            if section.start <= split_time <= section.end:
                # Split this section
                new_section = Section(
                    start=split_time,
                    end=section.end,
                    transpose=section.transpose,
                    reasons=section.reasons + ["用户添加"],
                )
                section.end = split_time
                self.sections.insert(i + 1, new_section)
                break

        self.redraw()
        if self.on_segments_changed:
            self.on_segments_changed(self.sections)

    def undo(self):
        """Undo last action."""
        if self.undo_stack:
            self.redo_stack.append(copy.deepcopy(self.sections))
            self.sections = self.undo_stack.pop()
            self.redraw()
            if self.on_segments_changed:
                self.on_segments_changed(self.sections)

    def redo(self):
        """Redo last undone action."""
        if self.redo_stack:
            self.undo_stack.append(copy.deepcopy(self.sections))
            self.sections = self.redo_stack.pop()
            self.redraw()
            if self.on_segments_changed:
                self.on_segments_changed(self.sections)


class M2MMainWindow(tk.Tk):
    """
    Main window for m2m MIDI→MIDI application.

    Features:
    - File selection (input/output MIDI)
    - Configuration controls
    - Waveform canvas with segments
    - Process button
    - Progress display
    """

    def __init__(self):
        super().__init__()

        self.title("m2m - MIDI→MIDI 分段移调工具")
        self.geometry("1000x700")

        # State
        self.input_midi: Optional[str] = None
        self.output_midi: Optional[str] = None
        self.config: M2MConfig = M2MConfig()
        self.pipeline: Optional[M2MPipeline] = None
        self.sections: List[Section] = []

        # Setup GUI
        self._create_widgets()

        # Bind keyboard shortcuts
        self.bind("<Control-z>", lambda e: self.waveform.undo())
        self.bind("<Control-y>", lambda e: self.waveform.redo())

    def _create_widgets(self):
        """Create and layout all widgets."""

        # Top control panel
        control_frame = ttk.Frame(self, padding="10")
        control_frame.pack(fill=tk.X)

        # File selection
        ttk.Button(
            control_frame, text="选择输入 MIDI", command=self._select_input_midi
        ).grid(row=0, column=0, padx=5)
        self.input_label = ttk.Label(control_frame, text="未选择文件")
        self.input_label.grid(row=0, column=1, padx=5, sticky=tk.W)

        ttk.Button(
            control_frame, text="选择输出 MIDI", command=self._select_output_midi
        ).grid(row=0, column=2, padx=5)
        self.output_label = ttk.Label(control_frame, text="未选择文件")
        self.output_label.grid(row=0, column=3, padx=5, sticky=tk.W)

        # Configuration
        config_frame = ttk.LabelFrame(self, text="配置", padding="10")
        config_frame.pack(fill=tk.X, padx=10, pady=5)

        # Segmentation strategy
        ttk.Label(config_frame, text="分段策略:").grid(row=0, column=0, sticky=tk.W)
        self.strategy_var = tk.StringVar(value="adaptive")
        strategy_combo = ttk.Combobox(
            config_frame,
            textvariable=self.strategy_var,
            values=["basic", "adaptive", "ssm", "ssm_pro"],
            state="readonly",
            width=15,
        )
        strategy_combo.grid(row=0, column=1, padx=5, sticky=tk.W)

        # Melody threshold
        ttk.Label(config_frame, text="主旋律阈值:").grid(
            row=0, column=2, sticky=tk.W, padx=(20, 0)
        )
        self.threshold_var = tk.IntVar(value=60)
        ttk.Spinbox(
            config_frame, from_=0, to=127, textvariable=self.threshold_var, width=10
        ).grid(row=0, column=3, padx=5)

        # Melody threshold enable checkbox
        self.enable_threshold_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            config_frame, text="启用", variable=self.enable_threshold_var
        ).grid(row=0, column=4, padx=5)

        # Transposition range
        ttk.Label(config_frame, text="移调范围:").grid(
            row=1, column=0, sticky=tk.W, pady=(5, 0)
        )
        range_frame = ttk.Frame(config_frame)
        range_frame.grid(row=1, column=1, columnspan=3, sticky=tk.W, pady=(5, 0))

        self.transpose_min_var = tk.IntVar(value=-27)
        ttk.Spinbox(
            range_frame, from_=-27, to=0, textvariable=self.transpose_min_var, width=8
        ).pack(side=tk.LEFT)
        ttk.Label(range_frame, text=" 到 ").pack(side=tk.LEFT, padx=5)
        self.transpose_max_var = tk.IntVar(value=27)
        ttk.Spinbox(
            range_frame, from_=0, to=27, textvariable=self.transpose_max_var, width=8
        ).pack(side=tk.LEFT)

        # SSM-Pro advanced configuration panel
        self.ssm_pro_frame = ttk.LabelFrame(self, text="SSM-Pro 高级配置", padding="10")
        self.ssm_pro_frame.pack(fill=tk.X, padx=10, pady=5)

        # Initially hidden, only show when ssm_pro strategy is selected
        self.ssm_pro_frame.pack_forget()

        # Feature weight: Pitch
        ttk.Label(self.ssm_pro_frame, text="音高特征权重:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.ssm_pro_pitch_var = tk.DoubleVar(value=1.0)
        pitch_scale = ttk.Scale(
            self.ssm_pro_frame,
            from_=0.0,
            to=5.0,
            variable=self.ssm_pro_pitch_var,
            orient=tk.HORIZONTAL,
            length=150,
        )
        pitch_scale.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        self.ssm_pro_pitch_label = ttk.Label(self.ssm_pro_frame, text="1.0")
        self.ssm_pro_pitch_label.grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        pitch_scale.configure(
            command=lambda v: self.ssm_pro_pitch_label.config(text=f"{float(v):.1f}")
        )

        # Feature weight: Chroma
        ttk.Label(self.ssm_pro_frame, text="和声特征权重:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.ssm_pro_chroma_var = tk.DoubleVar(value=1.5)
        chroma_scale = ttk.Scale(
            self.ssm_pro_frame,
            from_=0.0,
            to=5.0,
            variable=self.ssm_pro_chroma_var,
            orient=tk.HORIZONTAL,
            length=150,
        )
        chroma_scale.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        self.ssm_pro_chroma_label = ttk.Label(self.ssm_pro_frame, text="1.5")
        self.ssm_pro_chroma_label.grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
        chroma_scale.configure(
            command=lambda v: self.ssm_pro_chroma_label.config(text=f"{float(v):.1f}")
        )

        # Feature weight: Density
        ttk.Label(self.ssm_pro_frame, text="节奏特征权重:").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        self.ssm_pro_density_var = tk.DoubleVar(value=1.0)
        density_scale = ttk.Scale(
            self.ssm_pro_frame,
            from_=0.0,
            to=5.0,
            variable=self.ssm_pro_density_var,
            orient=tk.HORIZONTAL,
            length=150,
        )
        density_scale.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        self.ssm_pro_density_label = ttk.Label(self.ssm_pro_frame, text="1.0")
        self.ssm_pro_density_label.grid(row=2, column=2, padx=5, pady=2, sticky=tk.W)
        density_scale.configure(
            command=lambda v: self.ssm_pro_density_label.config(text=f"{float(v):.1f}")
        )

        # Feature weight: Velocity
        ttk.Label(self.ssm_pro_frame, text="力度特征权重:").grid(
            row=3, column=0, sticky=tk.W, pady=2
        )
        self.ssm_pro_velocity_var = tk.DoubleVar(value=0.5)
        velocity_scale = ttk.Scale(
            self.ssm_pro_frame,
            from_=0.0,
            to=5.0,
            variable=self.ssm_pro_velocity_var,
            orient=tk.HORIZONTAL,
            length=150,
        )
        velocity_scale.grid(row=3, column=1, padx=5, pady=2, sticky=tk.W)
        self.ssm_pro_velocity_label = ttk.Label(self.ssm_pro_frame, text="0.5")
        self.ssm_pro_velocity_label.grid(row=3, column=2, padx=5, pady=2, sticky=tk.W)
        velocity_scale.configure(
            command=lambda v: self.ssm_pro_velocity_label.config(text=f"{float(v):.1f}")
        )

        # Minimum segment duration
        ttk.Label(self.ssm_pro_frame, text="最小分段时长 (秒):").grid(
            row=4, column=0, sticky=tk.W, pady=2
        )
        self.ssm_pro_min_duration_var = tk.DoubleVar(value=2.0)
        ttk.Spinbox(
            self.ssm_pro_frame,
            from_=2.0,
            to=30.0,
            increment=1.0,
            textvariable=self.ssm_pro_min_duration_var,
            width=10,
        ).grid(row=4, column=1, padx=5, pady=2, sticky=tk.W)

        # Add help text
        help_text = (
            "说明: 和声特征(Chroma)权重最高，用于捕捉和声变化。\n"
            "音高特征捕捉音高范围变化，节奏特征捕捉音符密度变化。\n"
            "力度特征捕捉动态变化（权重较低）。"
        )
        ttk.Label(
            self.ssm_pro_frame, text=help_text, font=("", 8), foreground="gray"
        ).grid(row=5, column=0, columnspan=3, pady=(5, 0), sticky=tk.W)

        # Bind strategy change to show/hide SSM-Pro panel
        self.strategy_var.trace_add("write", self._on_strategy_change)

        # Waveform canvas
        canvas_frame = ttk.LabelFrame(self, text="音符密度与分段预览", padding="10")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.waveform = WaveformCanvas(canvas_frame, bg="white")
        self.waveform.pack(fill=tk.BOTH, expand=True)
        self.waveform.on_segments_changed = self._on_segments_changed
        self.waveform.on_optimize_segment = self._optimize_segment
        self.waveform.on_set_transposition = self._set_transposition

        # Segment info
        info_frame = ttk.Frame(self)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.info_label = ttk.Label(info_frame, text="分段: 0 | 平均白键率: 0%")
        self.info_label.pack(side=tk.LEFT)

        ttk.Button(info_frame, text="撤销 (Ctrl+Z)", command=self.waveform.undo).pack(
            side=tk.RIGHT, padx=5
        )
        ttk.Button(info_frame, text="重做 (Ctrl+Y)", command=self.waveform.redo).pack(
            side=tk.RIGHT, padx=5
        )

        # Optimize All Button
        ttk.Button(
            info_frame, text="一键优化所有分段移调", command=self._optimize_all_segments
        ).pack(side=tk.RIGHT, padx=5)

        # Process button
        process_frame = ttk.Frame(self, padding="10")
        process_frame.pack(fill=tk.X)

        self.process_button = ttk.Button(
            process_frame,
            text="分析 MIDI (Analyze)",
            command=self._analyze_midi,
            state=tk.DISABLED,
        )
        self.process_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.export_button = ttk.Button(
            process_frame,
            text="导出 MIDI (Export)",
            command=self._export_midi,
            state=tk.DISABLED,
        )
        self.export_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill=tk.X, padx=10, pady=5)

        self.status_label = ttk.Label(self, text="就绪")
        self.status_label.pack(pady=5)

    def _select_input_midi(self):
        """Select input MIDI file and immediately show note density."""
        filename = filedialog.askopenfilename(
            title="选择输入 MIDI 文件",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")],
        )
        if filename:
            self.input_midi = filename
            self.input_label.config(text=filename)

            # Load and display MIDI immediately
            try:
                midi = pretty_midi.PrettyMIDI(filename)
                self.waveform.set_midi_only(midi)
                self.status_label.config(text=f"已加载: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"无法加载MIDI文件: {e}")

            self._update_process_button()

    def _select_output_midi(self):
        """Select output MIDI file."""
        filename = filedialog.asksaveasfilename(
            title="选择输出 MIDI 文件",
            defaultextension=".mid",
            filetypes=[("MIDI files", "*.mid"), ("All files", "*.*")],
        )
        if filename:
            self.output_midi = filename
            self.output_label.config(text=filename)
            self._update_process_button()

    def _on_strategy_change(self, *args):
        """Show/hide SSM-Pro panel based on selected strategy."""
        strategy = self.strategy_var.get()
        if strategy == "ssm_pro":
            # Show the SSM-Pro frame
            self.ssm_pro_frame.pack(fill=tk.X, padx=10, pady=5)
            # Update display order by repacking
            self.ssm_pro_frame.pack_info()
        else:
            self.ssm_pro_frame.pack_forget()

    def _update_process_button(self):
        """Enable process button if input file selected. Enable export if output selected + sections exist."""
        # Enable Analyze button if input is selected
        if self.input_midi:
            self.process_button.config(state=tk.NORMAL)
        else:
            self.process_button.config(state=tk.DISABLED)

        # Enable Export button if input loaded, output selected, and analysis done (sections exist)
        if self.input_midi and self.output_midi and self.sections:
            self.export_button.config(state=tk.NORMAL)
        else:
            self.export_button.config(state=tk.DISABLED)

    def _on_segments_changed(self, sections: List[Section]):
        """Handle segments changed from canvas."""
        self.sections = sections
        # Recalculate stats for all sections as boundaries might have changed
        self._recalculate_all_stats()
        self._update_info_label()
        self._update_process_button()  # Update export button state

    def _recalculate_all_stats(self):
        """Recalculate white key rate for all sections."""
        if not self.input_midi or not self.sections:
            return

        try:
            # We need to reload MIDI or keep a reference.
            # self.waveform.midi_data should have it if loaded.
            midi_data = self.waveform.midi_data
            if not midi_data:
                return

            try:
                from ..strategies.transposition import WhiteKeyRateOptimizer
            except ImportError:
                from strategies.transposition import WhiteKeyRateOptimizer

            # Collect all notes
            all_notes = []
            for instrument in midi_data.instruments:
                if instrument.is_drum:
                    continue
                for note in instrument.notes:
                    all_notes.append((note.start, note.end, note.pitch))

            # Update each section
            for section in self.sections:
                section_notes = [
                    (s, e, p)
                    for s, e, p in all_notes
                    if section.start <= s < section.end
                ]

                section.note_count = len(section_notes)
                if section.note_count > 0:
                    # Use section-specific melody threshold if available/relevant
                    # But respect global enable flag
                    threshold = section.melody_threshold

                    section.white_key_rate = WhiteKeyRateOptimizer._white_key_rate(
                        section_notes,
                        section.transpose,
                        threshold,
                        self.config.transposition_config.enable_melody_threshold,
                    )
                else:
                    section.white_key_rate = 0.0

        except Exception as e:
            print(f"Error recalculating stats: {e}")

    def _optimize_segment(self, section: Section):
        """Optimize transposition for a single section."""
        if not self.input_midi:
            return

        try:
            midi_data = self.waveform.midi_data
            if not midi_data:
                return

            # Get notes
            all_notes = []
            for instrument in midi_data.instruments:
                if instrument.is_drum:
                    continue
                for note in instrument.notes:
                    all_notes.append((note.start, note.end, note.pitch))

            section_notes = [
                (s, e, p) for s, e, p in all_notes if section.start <= s < section.end
            ]

            if not section_notes:
                messagebox.showwarning("警告", "该分段没有音符")
                return

            # Create optimizer
            try:
                from ..strategies.transposition import TranspositionFactory
            except ImportError:
                from strategies.transposition import TranspositionFactory

            # Use current config BUT use section's threshold
            self.config.transposition_config.semitone_min = self.transpose_min_var.get()
            self.config.transposition_config.semitone_max = self.transpose_max_var.get()
            # Ensure section uses its own threshold for optimization
            temp_config = copy.copy(self.config.transposition_config)
            temp_config.melody_threshold = section.melody_threshold
            temp_config.enable_melody_threshold = self.enable_threshold_var.get()

            optimizer = TranspositionFactory.create(temp_config.strategy)

            best_transpose, white_rate = optimizer.optimize(section_notes, temp_config)

            # Update section
            # Hack: Push current state to undo stack before modifying
            self.waveform.undo_stack.append(copy.deepcopy(self.waveform.sections))
            self.waveform.redo_stack.clear()

            section.transpose = int(best_transpose)
            section.white_key_rate = white_rate

            self.waveform.redraw()
            self._update_info_label()

            # Show result
            messagebox.showinfo(
                "优化完成",
                f"最佳移调: {best_transpose}\n白键率: {white_rate * 100:.1f}%",
            )

        except Exception as e:
            messagebox.showerror("错误", f"优化失败: {str(e)}")

    def _set_transposition(self, section: Section):
        """Open detailed settings for a section."""
        dialog = SegmentSettingsDialog(
            self,
            section,
            self.transpose_min_var.get(),
            self.transpose_max_var.get(),
            self.enable_threshold_var.get(),
        )
        self.wait_window(dialog)

        if dialog.result:
            # Save undo
            self.waveform.undo_stack.append(copy.deepcopy(self.waveform.sections))
            self.waveform.redo_stack.clear()

            section.transpose = dialog.result["transpose"]
            section.melody_threshold = dialog.result["melody_threshold"]
            section.berserk_mode = dialog.result["berserk_mode"]
            section.max_pitch = dialog.result["max_pitch"]

            # Recalculate rate
            self._recalculate_all_stats()

            self.waveform.redraw()
            self._update_info_label()

    def _optimize_all_segments(self):
        """Optimize all segments."""
        if not self.sections:
            return

        if not messagebox.askyesno(
            "确认", "确定要重新计算所有分段的最佳移调吗？\n这将覆盖当前的手动设置。"
        ):
            return

        # Save undo
        self.waveform.undo_stack.append(copy.deepcopy(self.waveform.sections))
        self.waveform.redo_stack.clear()

        # We can iterate and call optimize_segment logic, but optimized for batch
        try:
            midi_data = self.waveform.midi_data
            if not midi_data:
                return

            # Update config
            self.config.transposition_config.semitone_min = self.transpose_min_var.get()
            self.config.transposition_config.semitone_max = self.transpose_max_var.get()
            self.config.transposition_config.melody_threshold = self.threshold_var.get()
            self.config.transposition_config.enable_melody_threshold = (
                self.enable_threshold_var.get()
            )

            try:
                from ..strategies.transposition import TranspositionFactory
            except ImportError:
                from strategies.transposition import TranspositionFactory

            optimizer = TranspositionFactory.create(
                self.config.transposition_config.strategy
            )

            # Collect notes
            all_notes = []
            for instrument in midi_data.instruments:
                if instrument.is_drum:
                    continue
                for note in instrument.notes:
                    all_notes.append((note.start, note.end, note.pitch))

            count = 0
            for section in self.sections:
                section_notes = [
                    (s, e, p)
                    for s, e, p in all_notes
                    if section.start <= s < section.end
                ]

                if section_notes:
                    best_transpose, white_rate = optimizer.optimize(
                        section_notes, self.config.transposition_config
                    )
                    section.transpose = int(best_transpose)
                    section.white_key_rate = white_rate
                    section.note_count = len(section_notes)
                    count += 1

            self.waveform.redraw()
            self._update_info_label()
            messagebox.showinfo("完成", f"已优化 {count} 个分段")

        except Exception as e:
            messagebox.showerror("错误", f"优化失败: {str(e)}")

    def _update_info_label(self):
        """Update segment info label."""
        # Update tooltip enable state in waveform
        if hasattr(self, "waveform"):
            self.waveform.enable_threshold_display = self.enable_threshold_var.get()

        if self.sections:
            avg_rate = np.mean(
                [s.white_key_rate for s in self.sections if s.white_key_rate > 0]
            )
            self.info_label.config(
                text=f"分段: {len(self.sections)} | 平均白键率: {avg_rate * 100:.1f}%"
            )
        else:
            self.info_label.config(text="分段: 0 | 平均白键率: 0%")

    def _analyze_midi(self):
        """Process MIDI file with m2m pipeline (Analysis only)."""
        if not self.input_midi:
            messagebox.showerror("错误", "请先选择输入文件")
            return

        if self.sections:
            if not messagebox.askyesno(
                "确认", "重新分析将覆盖当前所有分段和设置。\n确定要继续吗？"
            ):
                return

        try:
            # Update config
            strategy_map = {
                "basic": SegmentationStrategy.BASIC,
                "adaptive": SegmentationStrategy.ADAPTIVE,
                "ssm": SegmentationStrategy.SSM,
                "ssm_pro": SegmentationStrategy.SSM_PRO,
            }
            self.config.segmentation_strategy = strategy_map[self.strategy_var.get()]
            self.config.transposition_config.melody_threshold = self.threshold_var.get()
            self.config.transposition_config.enable_melody_threshold = (
                self.enable_threshold_var.get()
            )
            self.config.transposition_config.semitone_min = self.transpose_min_var.get()
            self.config.transposition_config.semitone_max = self.transpose_max_var.get()

            # Update SSM-Pro config if using SSM-Pro strategy
            if self.config.segmentation_strategy == SegmentationStrategy.SSM_PRO:
                self.config.ssm_pro_config.pitch_weight = self.ssm_pro_pitch_var.get()
                self.config.ssm_pro_config.chroma_weight = self.ssm_pro_chroma_var.get()
                self.config.ssm_pro_config.density_weight = (
                    self.ssm_pro_density_var.get()
                )
                self.config.ssm_pro_config.velocity_weight = (
                    self.ssm_pro_velocity_var.get()
                )
                self.config.ssm_pro_config.min_segment_duration = (
                    self.ssm_pro_min_duration_var.get()
                )

            # Create pipeline
            self.pipeline = M2MPipeline(self.config)

            # Attach observer for progress updates
            try:
                from ..core.observer import FunctionObserver
            except ImportError:
                from core.observer import FunctionObserver

            def progress_handler(event):
                if event.type.value == "progress_update":
                    progress = event.data.get("progress", 0) * 100
                    self.progress["value"] = progress
                    self.status_label.config(text=event.message)
                    self.update()
                elif event.type.value == "progress_complete":
                    self.progress["value"] = 100
                    self.status_label.config(text=event.message)
                    self.update()
                elif event.type.value == "error":
                    messagebox.showerror("错误", event.message)

            self.pipeline.attach(FunctionObserver(progress_handler))

            # Analyze
            self.sections = self.pipeline.analyze(self.input_midi)

            # Display results on waveform
            import pretty_midi

            midi = pretty_midi.PrettyMIDI(self.input_midi)
            duration = midi.get_end_time()

            self.waveform.set_sections(self.sections, duration, midi_data=midi)
            self._update_info_label()
            self._update_process_button()

            messagebox.showinfo(
                "完成",
                "分析完成！请在下方波形图中调整分段和移调。\n调整完毕后点击“导出 MIDI”保存。",
            )

        except Exception as e:
            messagebox.showerror("错误", f"分析失败: {str(e)}")
            self.status_label.config(text="错误")

    def _export_midi(self):
        """Export current sections to MIDI file."""
        if not self.output_midi:
            self._select_output_midi()
            if not self.output_midi:  # If user cancelled
                return

        if not self.pipeline:
            messagebox.showerror("错误", "请先进行分析")
            return

        try:
            # Use the SAME pipeline instance to export
            # Pass the current sections from GUI (which might be modified)
            self.pipeline.export(self.output_midi, sections=self.sections)

            messagebox.showinfo("完成", f"导出成功！\n保存到: {self.output_midi}")

        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")

    def run(self):
        """Run the application."""
        self.mainloop()


def main():
    """Entry point for m2m GUI application."""
    app = M2MMainWindow()
    app.run()


if __name__ == "__main__":
    main()
