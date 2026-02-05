"""
M2M Framework - Unified Entry Point for GUI and CLI modes.

This script provides a single entry point for the m2m (MIDI→MIDI) framework
with both GUI and CLI interfaces.

Usage:
    python app.py                    # Launch GUI mode (default)
    python app.py --cli              # Force CLI mode
    python app.py --input input.mid --output output.mid  # CLI with arguments
    python app.py --help             # Show help

Examples:
    # GUI mode (interactive)
    python app.py

    # CLI mode with automatic processing
    python app.py --cli --input song.mid --output song_transposed.mid

    # CLI mode with custom segmentation
    python app.py --cli \\
        --input song.mid \\
        --output song_transposed.mid \\
        --segmentation adaptive
"""

import sys
import os
from pathlib import Path

# Add parent directory to sys.path to allow importing m2m as a package
# This resolves "attempted relative import beyond top-level package" when running from inside the directory
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))

import argparse
from typing import Optional, Tuple, Type


def _import_gui() -> Type:
    try:
        from m2m.gui.main import M2MMainWindow
    except ImportError:
        from gui.main import M2MMainWindow
    return M2MMainWindow


def _import_core() -> Tuple[type, type, type, type]:
    try:
        from m2m.core.models import (
            M2MConfig,
            SegmentationStrategy,
            TranspositionStrategy,
        )
        from m2m.core.pipeline import M2MPipeline
    except ImportError:
        from core.models import (
            M2MConfig,
            SegmentationStrategy,
            TranspositionStrategy,
        )
        from core.pipeline import M2MPipeline
    return M2MConfig, SegmentationStrategy, TranspositionStrategy, M2MPipeline


def launch_gui():
    """
    Launch the m2m GUI application.

    This creates the main window with waveform visualization,
    allowing users to interactively adjust segments and transpositions.
    """
    try:
        M2MMainWindow = _import_gui()
        app = M2MMainWindow()
        app.mainloop()

    except ImportError as e:
        print(f"错误: 无法导入GUI模块: {e}")
        print(
            "请确保从项目上级目录运行，或已安装必需依赖: pip install pretty_midi numpy"
        )
        sys.exit(1)
    except Exception as e:
        print(f"错误: GUI启动失败: {e}")
        sys.exit(1)


def run_cli(args: argparse.Namespace) -> int:
    """
    Run the m2m framework in CLI mode.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success, 1 = error)
    """
    try:
        M2MConfig, SegmentationStrategy, TranspositionStrategy, M2MPipeline = (
            _import_core()
        )
        from pathlib import Path

        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"错误: 输入文件不存在: {args.input}")
            return 1

        if not input_path.suffix.lower() in [".mid", ".midi"]:
            print(f"警告: 输入文件可能不是MIDI格式: {args.input}")

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            # Generate output filename
            output_path = (
                input_path.parent / f"{input_path.stem}_m2m{input_path.suffix}"
            )

        # Create configuration
        config = M2MConfig()

        # Parse segmentation strategy
        if args.segmentation:
            try:
                config.segmentation_strategy = SegmentationStrategy(args.segmentation)
            except ValueError:
                print(f"错误: 未知的分段策略: {args.segmentation}")
                print(f"可用选项: {[s.value for s in SegmentationStrategy]}")
                return 1
        else:
            # Default to game_piano for optimal game piano performance
            config.transposition_config.strategy = TranspositionStrategy.GAME_PIANO

        # Apply SSM-Pro specific configuration
        if config.segmentation_strategy == SegmentationStrategy.SSM_PRO:
            config.ssm_pro_config.pitch_weight = args.ssm_pro_pitch_weight
            config.ssm_pro_config.chroma_weight = args.ssm_pro_chroma_weight
            config.ssm_pro_config.density_weight = args.ssm_pro_density_weight
            config.ssm_pro_config.velocity_weight = args.ssm_pro_velocity_weight

        # Print configuration
        print("=" * 60)
        print("M2M MIDI→MIDI Converter - CLI Mode")
        print("=" * 60)
        print(f"输入文件: {input_path}")
        print(f"输出文件: {output_path}")
        print(f"分段策略: {config.segmentation_strategy.value}")
        print("=" * 60)

        # Create pipeline and process
        pipeline = M2MPipeline(config)

        # Attach progress observer (print to console)
        def progress_callback(event):
            if event.type.value == "progress":
                print(
                    f"[{event.data['current']}/{event.data['total']}] {event.message}"
                )
            elif event.type.value == "complete":
                print(f"✓ {event.message}")
            elif event.type.value == "error":
                print(f"✗ {event.message}")

        pipeline.attach(progress_callback)

        # Process MIDI
        print("\n开始处理...")
        sections = pipeline.process(str(input_path), str(output_path))

        # Print results
        print("\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)
        print(f"生成了 {len(sections)} 个段落:")
        print()

        for i, section in enumerate(sections, 1):
            duration_str = f"{section.duration:.1f}s"
            transpose_str = f"{int(section.transpose):+d}"
            rate_str = f"{section.white_key_rate * 100:.1f}%"
            reasons_str = ", ".join(section.reasons) if section.reasons else "无"

            print(
                f"  {i}. [{section.start:.1f}s - {section.end:.1f}s] ({duration_str})"
            )
            print(
                f"     移调: {transpose_str}  |  白键率: {rate_str}  |  原因: {reasons_str}"
            )

        print()
        print(f"输出已保存到: {output_path}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n错误: 处理失败: {e}")
        import traceback

        traceback.print_exc()
        return 1


def main():
    """Main entry point for m2m application."""
    parser = argparse.ArgumentParser(
        description="M2M - MIDI→MIDI Framework with Intelligent Segmentation and Transposition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # GUI模式 (交互式)
  python app.py

  # CLI模式 - 自动处理
  python app.py --cli --input song.mid --output song_transposed.mid

  # CLI模式 - 自定义分段策略
  python app.py --cli --input song.mid --output song_out.mid \\
      --segmentation adaptive

分段策略:
  basic       - 基础分段 (速度/拍号/空隙边界)
  adaptive    - 自适应分段 (迭代优化白键率)
  ssm         - 自相似矩阵算法 (推荐，基于最新研究)
  ssm_pro     - 增强型SSM (多特征融合，Chroma+力度+节奏，效果最佳)

推荐配置:
   python app.py --cli -i input.mid -o output.mid \\
       --segmentation ssm_pro
         """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--cli", action="store_true", help="强制使用CLI模式 (默认: GUI模式)"
    )
    mode_group.add_argument("--gui", action="store_true", help="强制使用GUI模式")

    # Input/output files
    parser.add_argument("--input", "-i", type=str, help="输入MIDI文件路径")
    parser.add_argument(
        "--output", "-o", type=str, help="输出MIDI文件路径 (默认: 输入文件名_m2m.mid)"
    )

    # Segmentation options
    parser.add_argument(
        "--segmentation",
        "-s",
        type=str,
        choices=["basic", "adaptive", "ssm", "ssm_pro"],
        help="分段策略 (默认: ssm_pro)",
    )

    # SSM-Pro specific options
    ssm_pro_group = parser.add_argument_group("SSM-Pro 高级选项")
    ssm_pro_group.add_argument(
        "--ssm-pro-pitch-weight",
        type=float,
        default=1.0,
        help="SSM-Pro音高特征权重 (默认: 1.0, 范围: 0.0-5.0)",
    )
    ssm_pro_group.add_argument(
        "--ssm-pro-chroma-weight",
        type=float,
        default=1.5,
        help="SSM-Pro和声特征权重 (默认: 1.5, 范围: 0.0-5.0，最重要)",
    )
    ssm_pro_group.add_argument(
        "--ssm-pro-density-weight",
        type=float,
        default=1.0,
        help="SSM-Pro节奏特征权重 (默认: 1.0, 范围: 0.0-5.0)",
    )
    ssm_pro_group.add_argument(
        "--ssm-pro-velocity-weight",
        type=float,
        default=0.5,
        help="SSM-Pro力度特征权重 (默认: 0.5, 范围: 0.0-5.0)",
    )

    # Other options
    parser.add_argument(
        "--version", "-v", action="version", version="M2M Framework v1.0.0"
    )

    args = parser.parse_args()

    # Determine mode
    if args.cli:
        # CLI mode
        if not args.input:
            parser.error("--input 参数在CLI模式下是必需的")
        sys.exit(run_cli(args))
    else:
        # GUI mode (default)
        if args.input or args.output:
            print("提示: 使用 --cli 参数来运行CLI模式")
            print("      直接运行 python app.py 启动GUI模式")
        launch_gui()


if __name__ == "__main__":
    main()
