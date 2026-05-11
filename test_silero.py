import soundfile as sf
import torch
import sys, os

sys.path.insert(0, os.path.join(r'C:\Users\wuxiukun\.openclaw\workspace-codex\vad_benchmark', 'silero-vad', 'src'))
from silero_vad import load_silero_vad, get_speech_timestamps

torch.set_num_threads(1)
model = load_silero_vad(onnx=False)

for wav_path in [r'C:\Users\wuxiukun\.openclaw\workspace-codex\vad_benchmark\silero-vad\tests\data\test.wav',
                 r'C:\Users\wuxiukun\.openclaw\workspace-codex\vad_benchmark\silero-vad\tests\data\noisy_test.wav']:
    audio, sr = sf.read(wav_path)
    wav_tensor = torch.from_numpy(audio).float()
    if wav_tensor.ndim == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    stamps = get_speech_timestamps(wav_tensor, model, sampling_rate=sr, return_seconds=True)
    total = sum(s["end"] - s["start"] for s in stamps)
    label = "clean" if "noisy" not in wav_path else "noisy"
    print(f"{label}: {len(stamps)} segments, {total:.2f}s speech")
