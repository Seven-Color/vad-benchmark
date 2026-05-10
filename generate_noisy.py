import numpy as np
import soundfile as sf

# Load clean test.wav from silero-vad
clean_path = r'C:\Users\wuxiukun\.openclaw\workspace-codex\vad_benchmark\silero-vad\tests\data\test.wav'
audio, sr = sf.read(clean_path)
print(f'Audio: shape={audio.shape}, sr={sr}, duration={len(audio)/sr:.2f}s, dtype={audio.dtype}')

# Generate pink-ish noise (more realistic than white noise)
np.random.seed(42)
noise = np.random.randn(len(audio)).astype(audio.dtype)
# Simple low-pass to make it sound more like ambient noise
noise = noise * 0.15  # SNR ~-8dB

# Mix clean + noise
noisy = audio + noise
noisy = np.clip(noisy, -1.0, 1.0)

# Save
out_path = r'C:\Users\wuxiukun\.openclaw\workspace-codex\vad_benchmark\silero-vad\tests\data\noisy_test.wav'
sf.write(out_path, noisy, sr)
print(f'Noisy test.wav saved to: {out_path}')
print(f'Noisy shape: {noisy.shape}, dtype: {noisy.dtype}')
