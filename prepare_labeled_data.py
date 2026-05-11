"""
Prepare labeled VAD test data from LibriSpeech.
Generates frame-level binary labels (10ms frames) from word-level .stm timestamps.

Usage:
    python prepare_labeled_data.py          # full download + generate
    python prepare_labeled_data.py --skip-download   # use existing LibriSpeech
    python prepare_labeled_data.py --max-speakers 3  # limit for quick test
"""

import os
import sys
import json
import tarfile
import struct
import argparse
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf

FRAME_MS = 10
SAMPLE_RATE = 16000
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "labeled_data")
LIBRI_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"


# ─────────────────────────────────────────────
# Parse LibriSpeech .stm files
# ─────────────────────────────────────────────
def parse_stm(stm_path):
    """
    Parse LibriSpeech .stm file.
    Format: speaker_id file_id channel start_time end_time label text
    Lines starting with 'INTERSPEAKER' or 'IGNORE' are silence/non-speech.
    """
    segments = []
    try:
        with open(stm_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';;') or line.startswith('INTERSPEAKER') or line.startswith('IGNORE'):
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    start = float(parts[3])
                    end = float(parts[4])
                    text = ' '.join(parts[6:])
                    # Skip silence markers
                    if text.upper() in ('SIL', 'SP', '', '<o,f0,fem>', '<o,f0,male>', '<o,f0>'):
                        continue
                    # Skip very short segments
                    if end - start < 0.05:
                        continue
                    segments.append({'start': round(start, 3), 'end': round(end, 3), 'text': text})
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"    Error reading {stm_path}: {e}")
    return segments


# ─────────────────────────────────────────────
# Convert segments → frame-level labels
# ─────────────────────────────────────────────
def segments_to_frame_labels(segments, audio_duration_s, frame_ms=10):
    """Convert word segments to binary frame labels (1=speech, 0=non-speech)."""
    n_frames = int(audio_duration_s * 1000 / frame_ms)
    labels = np.zeros(n_frames, dtype=np.int8)
    for seg in segments:
        start_f = max(0, int(seg['start'] * 1000 / frame_ms))
        end_f = min(n_frames, int(seg['end'] * 1000 / frame_ms))
        labels[start_f:end_f] = 1
    return labels


# ─────────────────────────────────────────────
# Load audio (supports FLAC via soundfile)
# ─────────────────────────────────────────────
def load_audio(path):
    """Load audio file (wav or flac) as float32 numpy array, resample to 16k if needed."""
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio[:, 0]  # stereo → mono
    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        target_len = int(len(audio) * SAMPLE_RATE / sr)
        audio = resample(audio, target_len)
    return audio.astype(np.float32)


# ─────────────────────────────────────────────
# Download LibriSpeech dev-clean
# ─────────────────────────────────────────────
def download_librispeech():
    """Download and extract LibriSpeech dev-clean.tar.gz."""
    tar_path = os.path.join(DATA_DIR, "dev-clean.tar.gz")
    extract_dir = os.path.join(DATA_DIR, "LibriSpeech")

    if os.path.exists(extract_dir) and any(os.scandir(extract_dir)):
        print(f"  [SKIP] LibriSpeech already extracted → {extract_dir}")
        return extract_dir

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(tar_path):
        print(f"  Downloading LibriSpeech dev-clean (~322MB)...")
        print(f"  Source: {LIBRI_URL}")
        print(f"  Dest:   {tar_path}")
        def progress(count, block_size, total_size):
            pct = min(100, int(count * block_size * 100 / total_size))
            if pct % 5 == 0:
                print(f"\r    {pct}%", end='', flush=True)
        try:
            urllib.request.urlretrieve(LIBRI_URL, tar_path, reporthook=progress)
            print("\n  Download complete!")
        except Exception as e:
            print(f"\n  Download failed: {e}")
            if os.path.exists(tar_path):
                os.remove(tar_path)
            return None

    print(f"  Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(DATA_DIR)
    print(f"  Extracted → {extract_dir}")

    # Remove tar to save space
    os.remove(tar_path)
    return extract_dir


# ─────────────────────────────────────────────
# Find audio + stm pairs in LibriSpeech
# ─────────────────────────────────────────────
def find_librispeech_pairs(librispeech_dir, max_speakers=None, max_files=None):
    """
    Walk LibriSpeech dev-clean, find all (audio, stm) pairs.
    Returns list of dicts with audio_path, stm_path, speaker, duration.
    """
    pairs = []

    # LibriSpeech structure: LibriSpeech/dev-clean/{speaker_id}/{file_id}/{file_id}.flac
    # .stm files are alongside the audio in the same folder

    for speaker_dir in os.listdir(librispeech_dir):
        sp_path = os.path.join(librispeech_dir, speaker_dir)
        if not os.path.isdir(sp_path):
            continue
        for chapter_dir in os.listdir(sp_path):
            ch_path = os.path.join(sp_path, chapter_dir)
            if not os.path.isdir(ch_path):
                continue

            # Find .stm files in this chapter directory
            for fname in os.listdir(ch_path):
                if not fname.endswith('.stm'):
                    continue
                stm_path = os.path.join(ch_path, fname)
                base = fname[:-4]  # remove .stm

                # Find corresponding flac
                flac_path = os.path.join(ch_path, base + '.flac')
                if not os.path.exists(flac_path):
                    # Try .wav
                    wav_path = os.path.join(ch_path, base + '.wav')
                    if os.path.exists(wav_path):
                        audio_path = wav_path
                    else:
                        continue
                else:
                    audio_path = flac_path

                try:
                    # Get audio duration without fully loading
                    info = sf.info(audio_path)
                    duration = info.duration
                except Exception:
                    continue

                pairs.append({
                    'audio_path': audio_path,
                    'stm_path': stm_path,
                    'speaker': speaker_dir,
                    'chapter': chapter_dir,
                    'file_id': base,
                    'duration': duration
                })

        if max_speakers and len(pairs) >= max_speakers * 5:
            break

    return pairs


# ─────────────────────────────────────────────
# Generate labeled test files
# ─────────────────────────────────────────────
def generate_labeled_files(pairs, output_dir, label_source="librispeech_stm"):
    """
    Process (audio, stm) pairs → save as 16kHz wav + .labels.json.
    Also generate a noisy version by adding white noise at SNR ~0dB.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "noisy"), exist_ok=True)

    # Load noise sample (white noise, 2 seconds)
    noise_sample = np.random.randn(2 * SAMPLE_RATE).astype(np.float32) * 0.02

    results = []
    for i, pair in enumerate(pairs):
        name = f"{pair['speaker']}_{pair['file_id']}"
        out_wav = os.path.join(output_dir, f"{name}.wav")
        out_labels = os.path.join(output_dir, f"{name}.labels.json")
        out_noisy_wav = os.path.join(output_dir, "noisy", f"{name}_noisy.wav")
        out_noisy_labels = os.path.join(output_dir, "noisy", f"{name}_noisy.labels.json")

        try:
            # Load audio
            audio = load_audio(pair['audio_path'])
            duration = len(audio) / SAMPLE_RATE

            # Parse STM
            segments = parse_stm(pair['stm_path'])
            if not segments:
                print(f"  [{i}] {name}: no speech segments found — skipping")
                continue

            # Convert to frame labels
            labels = segments_to_frame_labels(segments, duration, FRAME_MS)

            # Save clean wav (16kHz mono)
            sf.write(out_wav, audio, SAMPLE_RATE)
            with open(out_labels, 'w') as f:
                json.dump({
                    "source": label_source,
                    "sample_rate": SAMPLE_RATE,
                    "frame_ms": FRAME_MS,
                    "labels": labels.tolist(),
                    "segments": segments,
                    "original_file": pair['audio_path']
                }, f, indent=2)

            # Generate noisy version
            noise_len = len(audio)
            noise = np.tile(noise_sample, int(np.ceil(noise_len / len(noise_sample))))[:noise_len]
            # Random SNR between -5dB and +5dB
            snr_db = np.random.uniform(-5, 5)
            signal_rms = np.sqrt(np.mean(audio ** 2))
            noise_rms = np.sqrt(np.mean(noise ** 2))
            target_noise_rms = signal_rms / (10 ** (snr_db / 20))
            noise_scaled = noise * (target_noise_rms / (noise_rms + 1e-10))
            noisy_audio = audio + noise_scaled
            # Clip to [-1, 1]
            noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

            sf.write(out_noisy_wav, noisy_audio, SAMPLE_RATE)
            with open(out_noisy_labels, 'w') as f:
                json.dump({
                    "source": label_source,
                    "sample_rate": SAMPLE_RATE,
                    "frame_ms": FRAME_MS,
                    "labels": labels.tolist(),  # same labels as clean (noise doesn't change speech)
                    "segments": segments,
                    "snr_db": round(snr_db, 1),
                    "original_file": pair['audio_path']
                }, f, indent=2)

            print(f"  [{i}] {name}: {duration:.1f}s, {len(segments)} segments, SNR={snr_db:.1f}dB → saved")
            results.append({
                'name': name,
                'clean_wav': out_wav,
                'noisy_wav': out_noisy_wav,
                'duration': round(duration, 2),
                'segments': len(segments),
                'snr_db': round(snr_db, 1)
            })

        except Exception as e:
            print(f"  [{i}] {name}: ERROR {e}")
            continue

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Prepare labeled VAD test data from LibriSpeech')
    parser.add_argument('--max-speakers', type=int, default=5,
                        help='Max number of speakers to process (default: 5)')
    parser.add_argument('--max-per-speaker', type=int, default=2,
                        help='Max files per speaker (default: 2)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download, use existing LibriSpeech')
    parser.add_argument('--output-dir', default=os.path.join(BASE_DIR, 'test_labeled'),
                        help='Output directory for labeled test files')
    args = parser.parse_args()

    print("=" * 60)
    print("VAD Labeled Data Preparation — LibriSpeech dev-clean")
    print("=" * 60)

    # Step 1: Get LibriSpeech
    print("\n[1] Getting LibriSpeech dev-clean...")
    if args.skip_download:
        librispeech_dir = os.path.join(DATA_DIR, "LibriSpeech")
        if not os.path.exists(librispeech_dir):
            print("  ERROR: LibriSpeech not found at", librispeech_dir)
            print("  Run without --skip-download to download it.")
            return
    else:
        librispeech_dir = download_librispeech()
        if not librispeech_dir:
            print("Download failed.")
            return

    # Step 2: Find audio-STM pairs
    print(f"\n[2] Scanning for audio-STM pairs...")
    print(f"    (max {args.max_speakers} speakers × {args.max_per_speaker} files each)")
    pairs = find_librispeech_pairs(librispeech_dir, max_speakers=args.max_speakers)

    # Apply per-speaker limit
    speakers_seen = {}
    filtered = []
    for p in pairs:
        sp = p['speaker']
        count = speakers_seen.get(sp, 0)
        if count < args.max_per_speaker:
            filtered.append(p)
            speakers_seen[sp] = count + 1
    pairs = filtered

    print(f"    Found {len(pairs)} audio-STM pairs from {len(speakers_seen)} speakers")
    if not pairs:
        print("ERROR: No valid (audio, .stm) pairs found.")
        print("Check LibriSpeech dev-clean extraction.")
        return

    # Step 3: Generate labeled files
    print(f"\n[3] Generating labeled test files → {args.output_dir}")
    results = generate_labeled_files(pairs, args.output_dir)

    print(f"\n[4] Summary")
    print(f"    Generated {len(results)} labeled samples")
    print(f"    Clean wavs: {args.output_dir}/")
    print(f"    Noisy wavs: {args.output_dir}/noisy/")
    print(f"    Labels:     *.labels.json (10ms frame binary labels)")
    print(f"\n    To use with benchmark.py, set TEST_AUDIO_DIR = '{args.output_dir}'")
    print(f"    and ensure benchmark uses *.labels.json files.")

    # Save manifest
    manifest_path = os.path.join(args.output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump({
            'label_source': 'librispeech_dev_clean_stm',
            'frame_ms': FRAME_MS,
            'sample_rate': SAMPLE_RATE,
            'files': results
        }, f, indent=2)
    print(f"\n    Manifest: {manifest_path}")


if __name__ == '__main__':
    main()