"""
Download and prepare labeled VAD test data from LibriSpeech + QUT-NOISE.
This script:
1. Downloads LibriSpeech dev-clean (~322MB) - speech with word timestamps (.stm files)
2. Downloads QUT-NOISE (need to check URL)
3. Parses .stm files to extract word-level speech timestamps
4. Creates frame-level binary VAD labels (10ms frames)
5. Generates clean + noisy test wav files with labels

Usage: python prepare_dataset.py [--max-speakers N]
"""

import os
import sys
import json
import tarfile
import subprocess
import argparse
import urllib.request
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "labeled_data")
LIBRI_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
QUT_URL = "https://research.qut.edu.au/saivt/databases/qut-noise-databases-and-protocols/"

FRAME_MS = 10
SAMPLE_RATE = 16000

def parse_stm(stm_path):
    """Parse LibriSpeech .stm file to get word-level speech segments."""
    segments = []
    with open(stm_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            # Format: speaker_id file_id channel start_time end_time label text
            # Example: "1234-56789-AB 1234-56789-A A 0.000 1.230 <o,f0,fem> hello"
            try:
                start = float(parts[3])
                end = float(parts[4])
                text = ' '.join(parts[6:])
                if text and text != '<' and text.upper() != 'SIL' and text.upper() != 'SP':
                    segments.append({'start': start, 'end': end, 'word': text})
            except (ValueError, IndexError):
                continue
    return segments

def segments_to_frame_labels(segments, audio_duration_s, frame_ms=10):
    """Convert word segments to frame-level binary labels."""
    frame_samples = int(SAMPLE_RATE * frame_ms / 1000)
    n_frames = int(audio_duration_s * 1000 / frame_ms)
    labels = [0] * n_frames
    for seg in segments:
        start_f = max(0, int(seg['start'] * 1000 / frame_ms))
        end_f = min(n_frames, int(seg['end'] * 1000 / frame_ms))
        for i in range(start_f, end_f):
            labels[i] = 1
    return labels

def download_librispeech():
    """Download and extract LibriSpeech dev-clean."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tar_path = os.path.join(DATA_DIR, "dev-clean.tar.gz")
    extract_dir = os.path.join(DATA_DIR, "LibriSpeech")

    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"LibriSpeech already extracted at {extract_dir}")
        return extract_dir

    print(f"Downloading LibriSpeech dev-clean (~322MB)...")
    print(f"This may take several minutes...")

    # Use urllib with progress
    def report(count, block_size, total_size):
        pct = min(100, int(count * block_size * 100 / total_size))
        if pct % 10 == 0:
            print(f"\r  Downloaded: {pct}%", end='', flush=True)

    try:
        urllib.request.urlretrieve(LIBRI_URL, tar_path, reporthook=report)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print(f"Please download manually from: {LIBRI_URL}")
        print(f"Save to: {tar_path}")
        return None

    print(f"Extracting...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(DATA_DIR)
    print(f"Extracted to {extract_dir}")

    # Remove tar to save space
    os.remove(tar_path)
    return extract_dir

def create_labeled_samples(librispeech_dir, max_speakers=5, max_files_per_speaker=3):
    """Create labeled test samples from LibriSpeech dev-clean."""
    # Walk directory to find .stm files
    stm_files = []
    for root, dirs, files in os.walk(librispeech_dir):
        for f in files:
            if f.endswith('.stm'):
                stm_files.append(os.path.join(root, f))

    print(f"Found {len(stm_files)} .stm files")

    if not stm_files:
        print("No .stm files found! Check LibriSpeech extraction.")
        return []

    samples = []
    speakers_used = set()

    for stm_path in stm_files:
        speaker_id = os.path.basename(os.path.dirname(stm_path))
        if speaker_id in speakers_used and len(speakers_used) >= max_speakers:
            break
        if len(speakers_used) >= max_speakers:
            break

        # Get corresponding audio file
        stm_dir = os.path.dirname(stm_path)
        relative = os.path.relpath(stm_dir, librispeech_dir)
        audio_dir = os.path.join(librispeech_dir, relative)

        # Find flac/wav file
        audio_file = None
        for ext in ['.flac', '.wav']:
            potential = os.path.join(audio_dir, os.path.basename(stm_path).replace('.stm', ext))
            if os.path.exists(potential):
                audio_file = potential
                break

        if not audio_file:
            continue

        # Parse segments
        segments = parse_stm(stm_path)
        if not segments:
            continue

        samples.append({
            'audio_file': audio_file,
            'speaker_id': speaker_id,
            'segments': segments
        })
        speakers_used.add(speaker_id)
        print(f"  [{speaker_id}] {os.path.basename(audio_file)}: {len(segments)} word segments")

        if len(speakers_used) >= max_speakers:
            break

    return samples

def main():
    parser = argparse.ArgumentParser(description='Prepare labeled VAD test data')
    parser.add_argument('--max-speakers', type=int, default=5, help='Max speakers to process')
    parser.add_argument('--check-only', action='store_true', help='Just check what's available')
    args = parser.parse_args()

    print("=" * 60)
    print("VAD Dataset Preparation Tool")
    print("=" * 60)

    if args.check_only:
        # Just check what's available
        extract_dir = os.path.join(DATA_DIR, "LibriSpeech")
        if os.path.exists(extract_dir):
            stm_count = sum(1 for _ in Path(extract_dir).rglob('*.stm'))
            print(f"LibriSpeech found: {stm_count} .stm files")
        else:
            print("LibriSpeech not yet downloaded")
        return

    # Step 1: Download LibriSpeech
    librispeech_dir = download_librispeech()
    if not librispeech_dir:
        print("Failed to download LibriSpeech. Please download manually.")
        return

    # Step 2: Create labeled samples
    samples = create_labeled_samples(librispeech_dir, max_speakers=args.max_speakers)

    print(f"\nCreated {len(samples)} labeled samples")
    print("\nNext steps:")
    print("1. Review the samples")
    print("2. For noisy tests: mix with QUT-NOISE at various SNR levels")
    print("3. Generate .labels.json files for each test wav")
    print("4. Update benchmark.py to use the labeled test data")

if __name__ == '__main__':
    main()
