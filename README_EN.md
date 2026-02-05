[中文文档](README) | English

# Midi2Meow (MIDI→MIDI) Conversion Framework

## Project Overview

Midi2Meow (MIDI→MIDI) Framework is an intelligent music conversion tool designed specifically for **game pianos**. It can convert any MIDI music file into a format suitable for playing in the game "Open Space". Through intelligent segmentation and transposition optimization, it ensures that notes are playable within the limited range of 21 white keys (C3-B5).

## Core Features

### 1. Game Piano Optimization
- **Hardware Adaptation**: Specifically designed for 3-octave game pianos (21 white keys)
- **Range Limitation**: Notes mapped to MIDI 48-83 (C3-B5) range
- **White Key Optimization**: Maximize white key playability rate, ensuring all notes are playable on the game piano

### 2. Intelligent Segmentation
Multiple segmentation strategies provided:
- **Basic Segmentation**: Based on velocity, time signature, and gap boundaries
- **Adaptive Segmentation**: Iteratively optimize white key rate (first choice)
- **SSM Algorithm**: Intelligent segmentation based on self-similarity matrix 
- **SSM-Pro**: Enhanced SSM combining pitch, harmony, rhythm, and velocity features (recommended but may overfit)

### 3. Transposition Optimization
- **Game Piano Strategy**: Transposition algorithm optimized specifically for game pianos
- **White Key Rate Maximization**: Intelligently calculate the best transposition range (±27 semitones)
- **Melody Protection**: Identify and protect melody voices to avoid destroying musicality through excessive transposition

### 4. Dual Interfaces
- **GUI Mode**: Interactive interface with visualization of waveforms and segmentation results
- **CLI Mode**: Command-line interface suitable for batch processing and automation

## Technical Architecture

### Project Structure
```
m2m/
├── core/              # Core modules
│   ├── models.py      # Data models and configuration
│   ├── observer.py    # Process management
│   └── pipeline.py    # Main processing pipeline
├── strategies/        # Strategy implementations
│   ├── game_piano.py           # Game piano optimizer
│   ├── game_piano_constants.py # Game piano mapping table
│   ├── segmentation.py         # Segmentation strategy base class
│   ├── ssm_base.py             # SSM segmentation base class
│   ├── ssm_segmentation.py     # SSM segmentation implementation
│   ├── ssm_pro_segmentation.py # SSM-Pro segmentation implementation
│   └── transposition.py        # Transposition optimizer
├── gui/               # Graphical interface
│   └── main.py        # Main window implementation
├── app.py             # Application entry point
└── README_EN.md       # English documentation
```

### Core Algorithms
1. **Segmentation Detection**: Use SSM-Pro algorithm to identify musical section boundaries
2. **Transposition Calculation**: Calculate optimal transposition for each section (-27 to +27 semitones)
3. **White Key Rate Evaluation**: Evaluate white key playability ratio after transposition
4. **Note Mapping**: Map transposed notes to the game piano's 21 white keys

## Quick Start

### Install Dependencies
```bash
pip install pretty_midi numpy
```

### Use GUI Mode (Recommended)
```bash
python app.py
```

### Use CLI Mode
```bash
# Basic usage
python app.py --cli --input song.mid --output song_game.mid

# Use SSM-Pro segmentation (recommended)
python app.py --cli --input song.mid --output song_game.mid --segmentation ssm_pro

# Batch processing example
for file in *.mid; do
    python app.py --cli --input "$file" --output "game_${file}"
done
```

### Parameter Description
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Input MIDI file path | Required |
| `--output` | Output MIDI file path | Auto-generated |
| `--segmentation` | Segmentation strategy: basic/adaptive/ssm/ssm_pro | ssm_pro |
| `--ssm-pro-pitch-weight` | SSM-Pro pitch feature weight | 1.0 |
| `--ssm-pro-chroma-weight` | SSM-Pro harmony feature weight | 1.5 |
| `--ssm-pro-density-weight` | SSM-Pro rhythm feature weight | 1.0 |
| `--ssm-pro-velocity-weight` | SSM-Pro velocity feature weight | 0.5 |

## Game Piano Mapping Table

The game piano uses 21 white keys, divided into three octaves:

| Range | Range | MIDI Notes | LRCP Mark | Corresponding Keys |
|-------|-------|------------|-----------|-------------------|
| Low range | C3-B3 | 48-59 | L1-L7 | Low octave white keys |
| Mid range | C4-B4 | 60-71 | M1-M7 | Mid octave white keys |
| High range | C5-B5 | 72-83 | H1-H7 | High octave white keys |

**Key Limitations**:
- Only white keys are playable (no black keys)
- Notes must fall within the 48-83 range
- Notes beyond the range will be transposed or adjusted

## Configuration Guide

### Segmentation Strategy Comparison
| Strategy | Advantages | Applicable Scenarios |
|----------|------------|---------------------|
| basic | Fast speed, based on simple rules | Music with simple structure |
| adaptive | Iteratively optimize white key rate | Music with medium complexity |
| ssm | Based on self-similarity matrix, intelligently identify sections | Most pop music |
| ssm_pro | Multi-feature fusion, highest accuracy | Complex arrangements, classical music |

### Transposition Configuration
- `semitone_min/max`: Transposition range (default ±27 semitones)
- `melody_threshold`: Main melody detection threshold (MIDI 60 = C4)
