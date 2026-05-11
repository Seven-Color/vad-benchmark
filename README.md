# VAD Benchmark

语音活动检测（Voice Activity Detection）模型对比基准测试。

## 包含模型

| 模型 | 来源 | 特点 |
|------|------|------|
| **Silero VAD** | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | PyTorch/ONNX，6000+语言，支持 8k/16kHz |
| **WebRTC VAD** | [dpirch/libfvad](https://github.com/dpirch/libfvad) | C 语言实现，WebRTC 原生引擎 |
| **Energy VAD** | 本项目 | 基于能量阈值 + 带通滤波，简单基线 |

## 快速开始

```bash
# 1. 安装依赖
pip install torch torchaudio soundfile scipy numpy webrtcvad

# 2. 生成带标注的测试数据（合成语音+噪声，10个条件）
python generate_synthetic_data.py

# 3. 运行基准测试
python benchmark.py
```

**输出：** `results.json`（详细指标）+ `roc_curves.png`（ROC曲线）

## 数据说明

测试数据通过 `generate_synthetic_data.py` 自动生成，共 **6个基础样本 × 10个条件 = 60个文件**：

- **1 个 clean 条件**：纯语音，无噪声
- **3 种噪声类型** × **3 个 SNR 级别** = 9 个 noisy 条件
  - 噪声：pink（粉噪）、hospital（医院环境噪声）、white（白噪声）
  - SNR：0dB、-5dB、-10dB

每个样本 30 秒语音，frame 级别（10ms）二值标注（1=语音，0=非语音）。

标签格式：
```json
{
  "source": "synthetic",
  "sample_rate": 16000,
  "frame_ms": 10,
  "labels": [0, 1, 1, 0, ...],
  "segments": [{"start": 0.5, "end": 2.3}, ...],
  "noise_type": "pink",
  "snr_db": -5.0
}
```

## 目录结构

```
vad_benchmark/
├── generate_synthetic_data.py   # 生成带标注的测试数据
├── benchmark.py                  # 统一评测脚本
├── silero-vad/                   # Silero VAD
├── libfvad/                      # WebRTC VAD
├── test_labeled/                 # 生成的测试数据（gitignore）
└── test_audio/                   # 旧版测试音频
```

## 评测指标

- **Accuracy**：帧级准确率
- **Precision / Recall**：查准率 / 查全率
- **F1**：Precision 和 Recall 的调和平均
- **AUC**：ROC 曲线下面积（Silero 和 Energy VAD 提供概率分数）