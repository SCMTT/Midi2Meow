English | [English Version](README_EN.md)

# Mid2Meow (MIDI→MIDI) 转换框架

## 项目概述

Mid2Meow（MIDI→MIDI）框架是一个专门为**游戏钢琴**设计的智能音乐转换工具。它能够将任意MIDI音乐文件转换为适合在游戏《开放空间》中演奏的格式，通过智能分段和移调优化，确保音符在有限的21个白键（C3-B5）范围内可演奏。

## 核心功能

### 1. 游戏钢琴优化
- **硬件适配**：专门针对3八度游戏钢琴（21个白键）设计
- **范围限制**：音符映射到MIDI 48-83（C3-B5）范围内
- **白键优化**：最大化白键演奏率，确保所有音符在游戏钢琴上可演奏

### 2. 智能分段
提供多种分段策略：
- **基础分段**：基于速度、拍号和空隙边界
- **自适应分段**：迭代优化白键率（首选）
- **SSM算法**：基于自相似矩阵的智能分段
- **SSM-Pro**：增强版SSM，融合音高、和声、节奏、力度多特征（推荐但可能过拟合）

### 3. 移调优化
- **游戏钢琴策略**：专门为游戏钢琴优化的移调算法
- **白键率最大化**：智能计算最佳移调幅度（±27个半音）
- **旋律保护**：可识别并保护旋律声部，避免过度移调破坏音乐性

### 4. 双重界面
- **GUI模式**：交互式界面，可视化波形和分段结果
- **CLI模式**：命令行接口，适合批量处理和自动化

## 技术架构

### 项目结构
```
m2m/
├── core/              # 核心模块
│   ├── models.py      # 数据模型和配置
│   ├── observer.py    # 进程管理
│   └── pipeline.py    # 主处理流程
├── strategies/        # 策略实现
│   ├── game_piano.py           # 游戏钢琴优化器
│   ├── game_piano_constants.py # 游戏钢琴映射表
│   ├── segmentation.py         # 分段策略基类
│   ├── ssm_base.py             # SSM分段基类
│   ├── ssm_segmentation.py     # SSM分段实现
│   ├── ssm_pro_segmentation.py # SSM-Pro分段实现
│   └── transposition.py        # 移调优化器
├── gui/               # 图形界面
│   └── main.py        # 主窗口实现
├── app.py             # 应用入口点
└── README_ZH.md       # 中文文档
```

### 核心算法
1. **分段检测**：使用SSM-Pro算法识别音乐段落边界
2. **移调计算**：为每个段落计算最优移调（-27到+27半音）
3. **白键率评估**：评估移调后段落的白键演奏比例
4. **音符映射**：将移调后的音符映射到游戏钢琴的21个白键

## 快速开始

### 安装依赖
```bash
pip install pretty_midi numpy
```

### 使用GUI模式（推荐）
```bash
python app.py
```

### 使用CLI模式
```bash
# 基础用法
python app.py --cli --input song.mid --output song_game.mid

# 使用SSM-Pro分段（推荐）
python app.py --cli --input song.mid --output song_game.mid --segmentation ssm_pro

# 批量处理示例
for file in *.mid; do
    python app.py --cli --input "$file" --output "game_${file}"
done
```

### 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入MIDI文件路径 | 必需 |
| `--output` | 输出MIDI文件路径 | 自动生成 |
| `--segmentation` | 分段策略：basic/adaptive/ssm/ssm_pro | ssm_pro |
| `--ssm-pro-pitch-weight` | SSM-Pro音高特征权重 | 1.0 |
| `--ssm-pro-chroma-weight` | SSM-Pro和声特征权重 | 1.5 |
| `--ssm-pro-density-weight` | SSM-Pro节奏特征权重 | 1.0 |
| `--ssm-pro-velocity-weight` | SSM-Pro力度特征权重 | 0.5 |

## 游戏钢琴映射表

游戏钢琴使用21个白键，分为三个八度：

| 音区 | 范围 | MIDI音符 | LRCP标记 | 对应键位 |
|------|------|----------|-----------|----------|
| 低音区 | C3-B3 | 48-59 | L1-L7 | 低八度白键 |
| 中音区 | C4-B4 | 60-71 | M1-M7 | 中八度白键 |
| 高音区 | C5-B5 | 72-83 | H1-H7 | 高八度白键 |

**关键限制**：
- 仅白键可演奏（无黑键）
- 音符必须落在48-83范围内
- 超过范围的音符会被移调或调整

## 配置说明

### 分段策略比较
| 策略 | 优点 | 适用场景 |
|------|------|----------|
| basic | 速度快，基于简单规则 | 结构简单的音乐 |
| adaptive | 迭代优化白键率 | 中等复杂度音乐 |
| ssm | 基于自相似矩阵，智能识别段落 | 大部分流行音乐 |
| ssm_pro | 多特征融合，精度最高 | 复杂编曲、古典音乐 |

### 移调配置
- `semitone_min/max`：移调范围（默认±27半音）
- `melody_threshold`：主旋律判定阈值（MIDI 60 = C4）