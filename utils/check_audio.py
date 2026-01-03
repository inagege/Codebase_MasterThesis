#!/usr/bin/env python3
import shutil
import subprocess
import json
from pathlib import Path
import sys
import csv

def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None

def get_audio_info(path: str | Path):
    """Return list of audio stream dicts or [] if none. Raises RuntimeError on ffprobe error."""
    if not ffprobe_available():
        raise RuntimeError("ffprobe not found on PATH")
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index,codec_name,channels,sample_rate",
        "-of", "json",
        str(path)
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
    info = json.loads(proc.stdout or "{}")
    return info.get("streams", [])


def _get_format_duration(path: str | Path) -> float | None:
    """Return duration in seconds using ffprobe, or None on error."""
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            return None
        info = json.loads(proc.stdout or "{}")
        return float(info.get('format', {}).get('duration')) if info.get('format') and info['format'].get('duration') else None
    except Exception:
        return None


def _get_volume_info(path: str | Path) -> dict:
    """Run ffmpeg volumedetect and return {'mean': float or None, 'max': float or None}.
    Returns None values on parse failure."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"]
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    stderr = proc.stderr or ''
    mean = None
    maxv = None
    import re
    m_mean = re.search(r'mean_volume:\s*([\-0-9.]+)\s*dB', stderr)
    m_max = re.search(r'max_volume:\s*([\-0-9.]+)\s*dB', stderr)
    if m_mean:
        try:
            mean = float(m_mean.group(1))
        except Exception:
            mean = None
    if m_max:
        try:
            maxv = float(m_max.group(1))
        except Exception:
            maxv = None
    return {'mean': mean, 'max': maxv}


def _get_silence_coverage(path: str | Path, silence_db: float = -50.0, min_silence_len: float = 0.5) -> float | None:
    """Run ffmpeg silencedetect and return fraction of duration that is silent (0..1), or None on error.
    Uses silence_db in dB and min_silence_len in seconds.
    """
    cmd = [
        'ffmpeg', '-hide_banner', '-nostats', '-i', str(path), '-vn', '-af', f"silencedetect=noise={silence_db}dB:d={min_silence_len}", '-f', 'null', '-'
    ]
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    stderr = proc.stderr or ''
    # sum silence_duration occurrences
    import re
    durations = re.findall(r'silence_duration:\s*([0-9]+\.?[0-9]*)', stderr)
    total_silence = 0.0
    try:
        total_silence = sum(float(x) for x in durations)
    except Exception:
        return None
    # get total duration
    duration = _get_format_duration(path)
    if duration is None or duration <= 0:
        return None
    return total_silence / duration


if __name__ == "__main__":
    # Usage: check_audio.py <directory> [out_csv]
    if len(sys.argv) < 2:
        print("Usage: check_audio.py <directory> [out_csv]", file=sys.stderr)
        sys.exit(2)

    dirpath = Path(sys.argv[1])
    if not dirpath.exists():
        print(f"Directory not found: {dirpath}", file=sys.stderr)
        sys.exit(2)

    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else dirpath / "missing_audio.csv"

    # File extensions to consider (same as your project VIDEO_EXTS)
    exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}

    # thresholds
    MAX_VOLUME_DB_THRESH = -50.0
    MEAN_VOLUME_DB_THRESH = -60.0
    SILENCE_COVERAGE_THRESH = 0.9  # 90%

    # Collect files deterministically
    files = sorted([p for p in dirpath.iterdir() if p.suffix.lower() in exts])

    missing = []

    for p in files:
        try:
            streams = get_audio_info(p)
        except Exception as e:
            missing.append({'file': str(p), 'reason': f'error: {e}', 'mean_db': None, 'max_db': None, 'silence_coverage': None})
            continue

        if not streams:
            missing.append({'file': str(p), 'reason': 'no_audio', 'mean_db': None, 'max_db': None, 'silence_coverage': None})
            continue

        # streams present -> check loudness and silence coverage
        vol = _get_volume_info(p)
        silence_cov = _get_silence_coverage(p)

        mean_db = vol.get('mean')
        max_db = vol.get('max')

        is_silent = False
        # criteria: max < MAX_VOLUME_DB_THRESH OR mean < MEAN_VOLUME_DB_THRESH OR silence_coverage > SILENCE_COVERAGE_THRESH
        if max_db is not None and max_db < MAX_VOLUME_DB_THRESH:
            is_silent = True
        if mean_db is not None and mean_db < MEAN_VOLUME_DB_THRESH:
            is_silent = True
        if silence_cov is not None and silence_cov > SILENCE_COVERAGE_THRESH:
            is_silent = True

        if is_silent:
            reason_details = 'silent'
            missing.append({'file': str(p), 'reason': reason_details, 'mean_db': mean_db, 'max_db': max_db, 'silence_coverage': silence_cov})

    # Write CSV with missing-audio entries (if any)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'reason', 'mean_db', 'max_db', 'silence_coverage'])
        writer.writeheader()
        writer.writerows(missing)

    print(f"Checked {len(files)} files. Found {len(missing)} with missing audio or silent/error. Wrote: {out_csv}")