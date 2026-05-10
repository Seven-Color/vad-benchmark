"""
VAD Benchmark - Compare multiple VAD models on clean and noisy audio.
Usage: python benchmark.py
Output: results.json
"""

import json
import os
import sys

# ─────────────────────────────────────────────
# 0. Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_AUDIO_DIR = os.path.join(BASE_DIR, "silero-vad", "tests", "data")
RESULTS_FILE = os.path.join(BASE_DIR, "results.json")

CLEAN_WAV = os.path.join(TEST_AUDIO_DIR, "test.wav")
NOISY_WAV = os.path.join(TEST_AUDIO_DIR, "noisy_test.wav")

# ─────────────────────────────────────────────
# 1. Silero VAD
# ─────────────────────────────────────────────
def run_silero(wav_path):
    try:
        sys.path.insert(0, os.path.join(BASE_DIR, "silero-vad", "src"))
        from silero_vad import load_silero_vad, get_speech_timestamps
        import soundfile as sf
        import torch
        torch.set_num_threads(1)

        model = load_silero_vad(onnx=False)
        audio, sr = sf.read(wav_path)
        wav_tensor = torch.from_numpy(audio).float()
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        stamps = get_speech_timestamps(wav_tensor, model, sampling_rate=sr, return_seconds=True)
        total = sum(s["end"] - s["start"] for s in stamps)
        return {"method": "Silero VAD", "model_type": "torch", "speech_segments": len(stamps),
                "speech_duration_s": round(total, 2), "audio_file": wav_path}
    except Exception as e:
        return {"method": "Silero VAD", "model_type": "torch", "error": str(e),
                "speech_segments": 0, "speech_duration_s": 0.0, "audio_file": wav_path}

# ─────────────────────────────────────────────
# 2. WebRTC VAD
# ─────────────────────────────────────────────
def run_webrtc(wav_path):
    try:
        import numpy as np
        import soundfile as sf
        from webrtcvad import Vad

        audio, sr = sf.read(wav_path)
        if sr != 16000:
            from scipy.signal import resample
            target_len = int(len(audio) * 16000 / sr)
            audio = resample(audio, target_len)
            sr = 16000

        vad = Vad(2)  # aggressive mode 2
        frame_size = 480  # 30ms at 16kHz
        frames = []
        for i in range(0, len(audio), frame_size):
            frame = audio[i:i+frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            is_speech = vad.is_speech(pcm, sr)
            frames.append(is_speech)

        segments = []
        in_speech = False
        start = 0
        for i, f in enumerate(frames):
            if f and not in_speech:
                start = i
                in_speech = True
            elif not f and in_speech:
                segments.append({"start": round(start * 0.03, 3), "end": round(i * 0.03, 3)})
                in_speech = False
        if in_speech:
            segments.append({"start": round(start * 0.03, 3), "end": round(len(frames) * 0.03, 3)})

        total = sum(s["end"] - s["start"] for s in segments)
        return {"method": "WebRTC VAD", "model_type": "c-extension",
                "speech_segments": len(segments), "speech_duration_s": round(total, 2),
                "audio_file": wav_path}
    except Exception as e:
        return {"method": "WebRTC VAD", "model_type": "c-extension", "error": str(e),
                "speech_segments": 0, "speech_duration_s": 0.0, "audio_file": wav_path}

# ─────────────────────────────────────────────
# 3. nicklashansen VAD
# ─────────────────────────────────────────────
def run_nicklashansen(wav_path):
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import butter, lfilter

        audio, sr = sf.read(wav_path)
        if sr != 16000:
            from scipy.signal import resample
            target_len = int(len(audio) * 16000 / sr)
            audio = resample(audio, target_len)
            sr = 16000

        # Energy-based VAD on bandpass-filtered audio
        b, a = butter(5, [300/16000, 3000/16000], btype='band')
        filtered = lfilter(b, a, audio)
        win = 480
        hop = 160
        energies = []
        for i in range(0, len(filtered) - win, hop):
            e = np.sqrt(np.mean(filtered[i:i+win]**2))
            energies.append(e)
        energies = np.array(energies)
        thresh = np.mean(energies) + 0.8 * np.std(energies)
        is_speech = energies > thresh

        segments = []
        in_speech = False
        start = 0
        for i, f in enumerate(is_speech):
            t = i * hop / sr
            if f and not in_speech:
                start = t
                in_speech = True
            elif not f and in_speech:
                segments.append({"start": round(start, 3), "end": round(t, 3)})
                in_speech = False
        if in_speech:
            segments.append({"start": round(start, 3), "end": round(len(audio)/sr, 3)})

        total = sum(s["end"] - s["start"] for s in segments)
        return {"method": "nicklashansen VAD", "model_type": "energy-proxy",
                "speech_segments": len(segments), "speech_duration_s": round(total, 2),
                "audio_file": wav_path}
    except Exception as e:
        return {"method": "nicklashansen VAD", "model_type": "unknown", "error": str(e),
                "speech_segments": 0, "speech_duration_s": 0.0, "audio_file": wav_path}

# ─────────────────────────────────────────────
# 4. Run all
# ─────────────────────────────────────────────
def main():
    results = []
    for wav in [CLEAN_WAV, NOISY_WAV]:
        label = "clean" if "noisy" not in wav else "noisy"
        print(f"\n=== {label} audio: {os.path.basename(wav)} ===")
        for run_fn in [run_silero, run_webrtc, run_nicklashansen]:
            r = run_fn(wav)
            r["audio_label"] = label
            results.append(r)
            segs = r["speech_segments"]
            dur = r["speech_duration_s"]
            print(f"  {r['method']} ({r['model_type']}): {segs} segments, {dur}s speech")
            if "error" in r:
                print(f"  ERROR: {r['error']}")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
