"""
Generate synthetic labeled VAD test audio + frame-level ground truth.
No download needed — everything is programmatically generated.

Structure:
  test_labeled/
    sample_0000_clean.wav / .labels.json          # clean speech, no noise
    sample_0000_pink_SNR0.wav / .labels.json       # same speech + pink noise at 0dB
    sample_0000_pink_SNRm5.wav / .labels.json     # same speech + pink noise at -5dB
    sample_0000_pink_SNRm10.wav / .labels.json    # same speech + pink noise at -10dB
    sample_0000_hosp_SNR0.wav / .labels.json      # hospital noise at 0dB
    ... (6 base samples × 7 conditions each = 42 files)
"""

import os
import sys
import json
import math
import random
import argparse
import numpy as np
import soundfile as sf

np.random.seed(42)
random.seed(42)

SAMPLE_RATE = 16000
FRAME_MS = 10
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "test_labeled")


# ─────────────────────────────────────────────
# Audio synthesis helpers
# ─────────────────────────────────────────────
def make_speech_like(duration_s, amplitude=0.4):
    """Synthesize a speech-like signal: harmonics + formants + pitch variation."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, n, False)
    f0 = random.uniform(80, 200)
    pitch_var = 1 + 0.05 * np.sin(2 * np.pi * 3 * t)
    signal = np.zeros(n)
    for harmonic in range(1, 6):
        freq = f0 * harmonic * pitch_var
        amp = amplitude / (harmonic ** 1.2)
        phase = random.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    for ff in [500, 1500, 2500]:
        phase = random.uniform(0, 2 * np.pi)
        signal += (random.uniform(0.05, 0.15) * amplitude *
                   np.sin(2 * np.pi * ff * t + phase))
    return signal.astype(np.float32)

def make_pink_noise(duration_s, amplitude=0.02):
    n = int(SAMPLE_RATE * duration_s)
    white = np.random.randn(n)
    pink = np.zeros(n)
    b = 0.05
    prev = 0.0
    for i in range(n):
        pink[i] = b * white[i] + (1 - b) * prev
        prev = pink[i]
    return (pink / (np.max(np.abs(pink)) + 1e-10) * amplitude).astype(np.float32)

def make_white_noise(duration_s, amplitude=0.01):
    return (np.random.randn(int(SAMPLE_RATE * duration_s)) * amplitude).astype(np.float32)

def make_hospital_noise(duration_s, amplitude=0.015):
    """Electrical hum + broadband noise (like HVAC/equipment)."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, n, False)
    hum = (0.5 * np.sin(2 * np.pi * 50 * t) +
           0.3 * np.sin(2 * np.pi * 100 * t))
    white = np.random.randn(n) * 0.3
    pink = np.zeros(n)
    b = 0.02
    prev = 0.0
    for i in range(n):
        pink[i] = b * white[i] + (1 - b) * prev
        prev = pink[i]
    noise = hum * amplitude + pink * amplitude * 0.5
    return noise.astype(np.float32)

NOISE_BUILDERS = {
    'pink':    make_pink_noise,
    'white':   make_white_noise,
    'hospital': make_hospital_noise,
}

def generate_speech_segments(total_duration_s, speech_frac=0.45,
                             min_speech_s=0.3, max_speech_s=3.0,
                             min_gap_s=0.2, max_gap_s=1.5):
    """Return list of {start, end} segments in seconds."""
    segments = []
    t = random.uniform(0.05, 0.3)
    while t < total_duration_s - 0.5:
        speech_dur = random.uniform(min_speech_s, max_speech_s)
        gap_dur = random.uniform(min_gap_s, max_gap_s)
        start = t
        end = min(t + speech_dur, total_duration_s - 0.1)
        if end - start >= min_speech_s:
            segments.append({'start': round(start, 3), 'end': round(end, 3)})
        t += speech_dur + gap_dur
    return segments

def build_signal(duration_s, segments, noise_builder=None, snr_db=None):
    """Build audio signal from speech segments; optionally mix in noise at given SNR."""
    signal = make_pink_noise(duration_s, amplitude=0.005)  # quiet background
    for seg in segments:
        start_s, end_s = seg['start'], seg['end']
        speech = make_speech_like(end_s - start_s, amplitude=0.4)
        start_sample = int(start_s * SAMPLE_RATE)
        end_sample = start_sample + len(speech)
        if end_sample <= len(signal):
            signal[start_sample:end_sample] = speech[:end_sample - start_sample]

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0.95:
        signal = signal * (0.9 / peak)
    signal = np.clip(signal, -1.0, 1.0)

    # Mix noise if requested
    if noise_builder is not None and snr_db is not None:
        noise = noise_builder(duration_s)
        signal_rms = np.sqrt(np.mean(signal ** 2))
        noise_rms = np.sqrt(np.mean(noise ** 2))
        target_noise_rms = signal_rms / (10 ** (snr_db / 20))
        noise_scaled = noise * (target_noise_rms / (noise_rms + 1e-10))
        signal = np.clip(signal + noise_scaled, -1.0, 1.0)

    return signal

def segments_to_frame_labels(segments, n_frames):
    """Convert list of {start, end} segments → binary frame-level labels."""
    labels = np.zeros(n_frames, dtype=np.int8)
    for seg in segments:
        start_f = max(0, int(seg['start'] * 1000 / FRAME_MS))
        end_f = min(n_frames, int(seg['end'] * 1000 / FRAME_MS))
        labels[start_f:end_f] = 1
    return labels

def save_sample(fname, signal, labels, metadata):
    """Save wav + corresponding labels JSON."""
    sf.write(fname, signal, SAMPLE_RATE)
    label_fname = fname.replace('.wav', '.labels.json')
    with open(label_fname, 'w') as f:
        json.dump(metadata, f, indent=2)

def make_label_metadata(labels, segments, duration_s, speech_frac,
                        noise_type, snr_db, sample_id, condition):
    return {
        "source": "synthetic",
        "sample_rate": SAMPLE_RATE,
        "frame_ms": FRAME_MS,
        "labels": labels.tolist(),
        "segments": segments,
        "speech_frac": round(speech_frac, 3),
        "duration_s": round(duration_s, 2),
        "noise_type": noise_type,
        "snr_db": round(snr_db, 1) if snr_db is not None else None,
        "sample_id": sample_id,
        "condition": condition  # e.g. "clean", "pink_SNR0", "hospital_SNRm5"
    }


# ─────────────────────────────────────────────
# Main generation
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Generate synthetic labeled VAD test data')
    parser.add_argument('--num-samples', type=int, default=6,
                        help='Number of base speech samples (default: 6)')
    parser.add_argument('--duration', type=float, default=30,
                        help='Duration of each sample in seconds (default: 30)')
    parser.add_argument('--snr', type=float, nargs='+', default=[0, -5, -10],
                        help='SNR levels in dB (default: 0 -5 -10)')
    parser.add_argument('--noise-types', type=str, default='pink,hospital,white',
                        help='Comma-separated noise types (default: pink,hospital,white)')
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    noise_types = [n.strip() for n in args.noise_types.split(',')]
    # Conditions: clean + each noise_type at each SNR
    conditions = ['clean'] + [f'{n}_SNR{s}' for n in noise_types for s in args.snr]
    # Total files per sample: 1 clean + N_noise × N_snr noisy
    files_per_sample = 1 + len(noise_types) * len(args.snr)

    print("=" * 60)
    print("VAD Synthetic Labeled Data Generator")
    print(f"Output: {args.output_dir}")
    print(f"Base samples: {args.num_samples} × {args.duration}s")
    print(f"Noise types: {noise_types}")
    print(f"SNR levels: {args.snr} dB")
    print(f"Conditions per sample: {conditions}")
    print(f"Total files: {args.num_samples} × {files_per_sample} = {args.num_samples * files_per_sample}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    all_files = []

    for i in range(args.num_samples):
        # Generate speech segments once per base sample
        speech_frac = random.uniform(0.30, 0.60)
        segments = generate_speech_segments(args.duration, speech_frac)
        n_frames = int(args.duration * 1000 / FRAME_MS)
        labels = segments_to_frame_labels(segments, n_frames)

        # Condition 0: clean
        cond = 'clean'
        fname = os.path.join(args.output_dir, f"sample_{i:04d}_{cond}.wav")
        signal = build_signal(args.duration, segments, noise_builder=None, snr_db=None)
        meta = make_label_metadata(labels, segments, args.duration, speech_frac,
                                   noise_type='none', snr_db=None,
                                   sample_id=i, condition=cond)
        save_sample(fname, signal, labels, meta)
        all_files.append({'condition': cond, 'fname': os.path.basename(fname),
                          'sample_id': i, 'speech_frac': speech_frac,
                          'noise_type': 'none', 'snr_db': None})
        print(f"  [{i:02d}] {cond:20s} → {os.path.basename(fname)}")

        # Conditions 1..N: noisy variants
        for ni, nt in enumerate(noise_types):
            noise_builder = NOISE_BUILDERS[nt]
            for si, snr in enumerate(args.snr):
                cond = f'{nt}_SNR{snr}'
                fname = os.path.join(args.output_dir, f"sample_{i:04d}_{cond}.wav")
                # Use a different random state for each noise variant
                noise_signal = noise_builder(args.duration)
                signal = build_signal(args.duration, segments,
                                      noise_builder=noise_builder, snr_db=snr)
                meta = make_label_metadata(labels, segments, args.duration, speech_frac,
                                           noise_type=nt, snr_db=snr,
                                           sample_id=i, condition=cond)
                save_sample(fname, signal, labels, meta)
                all_files.append({'condition': cond, 'fname': os.path.basename(fname),
                                  'sample_id': i, 'speech_frac': speech_frac,
                                  'noise_type': nt, 'snr_db': snr})
                print(f"  [{i:02d}] {cond:20s} → {os.path.basename(fname)}")

    manifest = {
        'source': 'synthetic',
        'num_base_samples': args.num_samples,
        'duration_s': args.duration,
        'frame_ms': FRAME_MS,
        'sample_rate': SAMPLE_RATE,
        'noise_types': noise_types,
        'snr_levels': args.snr,
        'conditions': conditions,
        'files_per_sample': files_per_sample,
        'total_files': len(all_files),
        'files': all_files
    }
    manifest_path = os.path.join(args.output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! Generated {len(all_files)} files in {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"\nConditions breakdown per sample:")
    print(f"  1 clean + {len(noise_types)} noise types × {len(args.snr)} SNR levels = {files_per_sample} files/sample")
    print(f"\nTo evaluate with benchmark.py, it will auto-discover all .wav files")
    print(f"and use the corresponding .labels.json for ground truth.")

if __name__ == '__main__':
    main()