"""
Game Piano Constants - Configuration for 3-octave game pianos.

This module defines the note mapping for game pianos with limited range.
Based on the reference implementation from midi2lrcp project.

Game Piano Range:
- 21 white keys across 3 octaves (C3-B5)
- MIDI pitch range: 48 (C3) to 83 (B5)
- LRCP tokens: L1-L7 (low), M1-M7 (mid), H1-H7 (high)

Key Design Principles:
1. ONLY white keys are playable (game pianos typically lack black keys)
2. Range is strictly limited to 21 keys (hardware constraint)
3. Playability is the ONLY optimization metric
"""

from typing import Dict, Set

# ============================================================================
# NOTE MAPPING TABLE (MIDI pitch -> LRCP token)
# ============================================================================

# Core mapping from MIDI pitch numbers to game piano tokens
# This is the definitive reference for which notes are playable
GAME_PIANO_NOTE_MAP: Dict[int, str] = {
    # Low register (C3-B3) - MIDI 48-59
    48: "L1",  # C3
    50: "L2",  # D3
    52: "L3",  # E3
    53: "L4",  # F3
    55: "L5",  # G3
    57: "L6",  # A3
    59: "L7",  # B3
    # Middle register (C4-B4) - MIDI 60-71
    60: "M1",  # C4
    62: "M2",  # D4
    64: "M3",  # E4
    65: "M4",  # F4
    67: "M5",  # G4
    69: "M6",  # A4
    71: "M7",  # B4
    # High register (C5-B5) - MIDI 72-83
    72: "H1",  # C5
    74: "H2",  # D5
    76: "H3",  # E5
    77: "H4",  # F5
    79: "H5",  # G5
    81: "H6",  # A5
    83: "H7",  # B5
}

# ============================================================================
# PLAYABLE NOTE SET (for fast lookup)
# ============================================================================

# Set of all playable MIDI pitch numbers on game piano
GAME_PIANO_PITCHES: Set[int] = set(GAME_PIANO_NOTE_MAP.keys())

# Range boundaries (for validation)
GAME_PIANO_MIN_PITCH: int = 48  # C3
GAME_PIANO_MAX_PITCH: int = 83  # B5
GAME_PIANO_KEY_COUNT: int = 21  # Total playable keys

# ============================================================================
# TOKEN CATEGORIES (for analysis)
# ============================================================================

# Reverse mapping for token analysis
TOKEN_TO_PITCH: Dict[str, int] = {v: k for k, v in GAME_PIANO_NOTE_MAP.items()}

# Token groupings by register
LOW_REGISTER_TOKENS: Set[str] = {f"L{i}" for i in range(1, 8)}  # L1-L7
MID_REGISTER_TOKENS: Set[str] = {f"M{i}" for i in range(1, 8)}  # M1-M7
HIGH_REGISTER_TOKENS: Set[str] = {f"H{i}" for i in range(1, 8)}  # H1-H7

ALL_TOKENS: Set[str] = LOW_REGISTER_TOKENS | MID_REGISTER_TOKENS | HIGH_REGISTER_TOKENS

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def is_playable_pitch(pitch: int) -> bool:
    """
    Check if a MIDI pitch is playable on game piano.

    Args:
        pitch: MIDI pitch number (0-127)

    Returns:
        True if pitch is in game piano range (48-83 and is white key)
    """
    return pitch in GAME_PIANO_PITCHES


def pitch_to_token(pitch: int) -> str | None:
    """
    Convert MIDI pitch to game piano token.

    Args:
        pitch: MIDI pitch number

    Returns:
        LRCP token (e.g., 'M1') if playable, None otherwise
    """
    return GAME_PIANO_NOTE_MAP.get(pitch)


def token_to_pitch(token: str) -> int | None:
    """
    Convert game piano token to MIDI pitch.

    Args:
        token: LRCP token (e.g., 'M1')

    Returns:
        MIDI pitch number if valid, None otherwise
    """
    return TOKEN_TO_PITCH.get(token)


def clamp_to_playable_range(pitch: int) -> int:
    """
    Clamp pitch to playable range [48, 83].

    Note: This does NOT guarantee the clamped pitch is a white key.
    Use is_playable_pitch() to verify.

    Args:
        pitch: MIDI pitch number

    Returns:
        Pitch clamped to range [48, 83]
    """
    return max(GAME_PIANO_MIN_PITCH, min(GAME_PIANO_MAX_PITCH, pitch))


# ============================================================================
# VALIDATION
# ============================================================================


def _validate_mapping():
    """Internal validation of note mapping consistency."""
    assert len(GAME_PIANO_NOTE_MAP) == 21, "Must have exactly 21 playable keys"
    assert GAME_PIANO_MIN_PITCH == 48, "Min pitch must be C3 (48)"
    assert GAME_PIANO_MAX_PITCH == 83, "Max pitch must be B5 (83)"

    # Verify all pitches are white keys
    white_key_pitches = {0, 2, 4, 5, 7, 9, 11}  # C, D, E, F, G, A, B
    for pitch in GAME_PIANO_PITCHES:
        assert (pitch % 12) in white_key_pitches, f"Pitch {pitch} is not a white key"

    # Verify reverse mapping is complete
    assert len(TOKEN_TO_PITCH) == 21, "Reverse mapping must be complete"
    assert len(ALL_TOKENS) == 21, "Must have exactly 21 tokens"

    # Verify register groupings
    assert len(LOW_REGISTER_TOKENS) == 7, "Low register must have 7 keys"
    assert len(MID_REGISTER_TOKENS) == 7, "Mid register must have 7 keys"
    assert len(HIGH_REGISTER_TOKENS) == 7, "High register must have 7 keys"

    print("[OK] Game piano note mapping validated successfully")


# Run validation on module import
_validate_mapping()
