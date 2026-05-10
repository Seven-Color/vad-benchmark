# VAD Benchmark

语音活动检测（Voice Activity Detection）模型对比基准测试。

## 包含模型

| 模型 | 来源 | 特点 |
|------|------|------|
| **Silero VAD** | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | PyTorch/ONNX，6000+语言，支持 8k/16kHz |
| **WebRTC VAD** | [dpirch/libfvad](https://github.com/dpirch/libfvad) | C 语言实现，WebRTC 原生引擎 |
| **nicklashansen VAD** | [nicklashansen/voice-activity-detection](https://github.com/nicklashansen/voice-activity-detection) | DenseNet/GRU 深度学习模型 |

## 目录结构

```
vad_benchmark/
├── silero-vad/              # Silero VAD（已克隆，含测试音频）
├── libfvad/                 # WebRTC VAD
├── nicklashansen_vad/       # nicklashansen VAD
├── test_audio/             # 测试音频文件
│   ├── clean_test.wav      # 原始干净音频
│   └── noisy_test.wav      # 加噪测试音频（SNR~-8dB）
└── benchmark.py            # 统一评测脚本
```

## 安装依赖

```bash
pip install torch torchaudio soundfile scipy numpy
```

## 运行测试

```bash
python benchmark.py
```

输出：`results.json`（各模型在 clean/noisy 音频上的检测结果）。

## 测试音频说明

- `clean_test.wav` — 来自 silero-vad 原始测试音频，60 秒 16kHz
- `noisy_test.wav` — 叠加高斯噪声（SNR ≈ -8dB），用于评估噪声鲁棒性

## 结果解读

`results.json` 字段：
- `method` — VAD 方法名称
- `model_type` — 模型类型（torch/onnx/c）
- `speech_segments` — 检测到的语音片段数
- `speech_duration_s` — 总语音时长（秒）
- `audio_file` — 测试音频路径
