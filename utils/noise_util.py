import os
import shutil
import subprocess
from pathlib import Path
import tempfile
import re

# Utility to add gaussian-like noise to audio or visual stream of videos in a directory
# Requires ffmpeg available on PATH.

VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


def _get_mean_volume_db(path: str) -> float:
    """Run ffmpeg volumedetect on the given audio/video file and return mean_volume in dBFS."""
    cmd = [
        'ffmpeg', '-hide_banner', '-nostats', '-i', str(path), '-vn', '-af', 'volumedetect', '-f', 'null', '-'
    ]
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    stderr = proc.stderr or ''
    m = re.search(r'mean_volume:\s*([\-0-9.]+)\s*dB', stderr)
    if not m:
        raise RuntimeError(f'Could not determine mean_volume for {path}. ffmpeg output:\n{stderr}')
    return float(m.group(1))


def add_noise_to_audio(in_path: str | Path, out_path: str | Path, amplitude: float = 0.02,
                       audio_snr_db: float | None = None, noise_color: str = 'white', overwrite: bool = False):
    """Add noise to the audio track.

    Two modes:
    - If audio_snr_db is None (default), behave like before and add lavfi noise with given linear amplitude.
    - If audio_snr_db is provided (float, in dB), adjust generated noise gain so that resulting noise has the
      requested Signal-to-Noise Ratio (SNR) relative to the input audio's mean level.

    Parameters:
    - in_path: input video path
    - out_path: output video path
    - amplitude: linear amplitude for anoisesrc when audio_snr_db is None
    - audio_snr_db: target SNR in dB (signal_level_db - noise_level_db). If provided, amplitude is ignored.
    - noise_color: color passed to anoisesrc (white, pink, ...)
    - overwrite: whether to overwrite existing output
    """
    in_path = str(in_path)
    out_path = str(out_path)

    if not _ffmpeg_available():
        raise RuntimeError('ffmpeg is not found on PATH')

    if os.path.exists(out_path) and not overwrite:
        return out_path

    if audio_snr_db is None:
        # Legacy/simple behavior: use anoisesrc amplitude directly
        cmd = [
            'ffmpeg', '-y' if overwrite else '-n', '-i', in_path,
            '-f', 'lavfi', '-i', f"anoisesrc=color={noise_color}:amplitude={amplitude}",
            '-filter_complex', '[0:a][1:a]amix=inputs=2:normalize=0',
            '-c:v', 'copy', '-c:a', 'aac', '-shortest', out_path,
        ]
        subprocess.run(cmd, check=True)
        return out_path

    # SNR mode: measure signal level, generate a short noise sample at amplitude=1 to measure its level,
    # compute required gain, then mix full-length noise scaled by that gain.
    # 1) measure signal mean volume (dBFS)
    signal_db = _get_mean_volume_db(in_path)

    # 2) create short temp noise sample at amplitude=1 to determine its mean level
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpf:
        tmp_noise = tmpf.name
    try:
        # generate 5s noise sample (duration doesn't affect mean level)
        gen_cmd = [
            'ffmpeg', '-hide_banner', '-y', '-f', 'lavfi', '-i', f"anoisesrc=color={noise_color}:amplitude=1:d=5:r=44100",
            '-c:a', 'pcm_s16le', tmp_noise,
        ]
        subprocess.run(gen_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        noise_db = _get_mean_volume_db(tmp_noise)

        # desired noise level (dB) to achieve target SNR
        desired_noise_db = signal_db - float(audio_snr_db)
        # gain (in dB) to apply to anoisesrc(amplitude=1) so its mean matches desired_noise_db
        noise_gain_db = desired_noise_db - noise_db

        # final mix: use lavfi anoisesrc with amplitude=1 and apply volume=<noise_gain_db>dB to it
        filter_complex = f"[1:a]volume={noise_gain_db}dB[n];[0:a][n]amix=inputs=2:normalize=0"
        cmd = [
            'ffmpeg', '-y' if overwrite else '-n', '-i', in_path,
            '-f', 'lavfi', '-i', f"anoisesrc=color={noise_color}:amplitude=1:r=44100",
            '-filter_complex', filter_complex,
            '-c:v', 'copy', '-c:a', 'aac', '-shortest', out_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.remove(tmp_noise)
        except Exception:
            pass

    return out_path


def add_noise_to_video(in_path: str | Path, out_path: str | Path, strength: float = 12.0,
                       distribution: str = 'u', overwrite: bool = False):
    """Add noise to video frames using ffmpeg's 'noise' filter.

    Parameters:
    - in_path: input video path
    - out_path: output video path
    - strength: strength parameter passed to ffmpeg's 'noise' filter (alls). Higher -> stronger noise.
    - distribution: passed through to the filter as allf (e.g. 'gauss' or 'uniform').
                    Note: supported values depend on your ffmpeg build; common values are 'u'/'g' or 'uniform'/'gauss'.
    - overwrite: whether to overwrite existing output
    """
    in_path = str(in_path)
    out_path = str(out_path)

    if not _ffmpeg_available():
        raise RuntimeError('ffmpeg is not found on PATH')

    if os.path.exists(out_path) and not overwrite:
        return out_path

    # Use libx264 for reasonable compatibility; copy audio stream unchanged.
    vf = f"noise=alls={strength}:allf={distribution}"
    cmd = [
        'ffmpeg', '-y' if overwrite else '-n', '-i', in_path,
        '-vf', vf,
        '-c:a', 'copy', '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast', out_path,
    ]

    subprocess.run(cmd, check=True)
    return out_path


def process_train_splits(base_dir: str | Path = os.path.join('data', 'MELD.Raw', 'output_repeated_splits_test'), audio_dir_name: str = 'audio', visual_dir_name: str = 'visual',
                         audio_amplitude: float = 0.02, video_strength: float = 12.0, audio_snr_db: float | None = 10.0, max_samples: int | None = None, overwrite: bool = False):
    """Iterate over video files in `base_dir`, create subdirectories for audio/visual noisy versions,
    and produce noisy copies.

    Returns a list of tuples (original, audio_noisy_path, visual_noisy_path).
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    audio_dir = base_dir / audio_dir_name
    visual_dir = base_dir / visual_dir_name
    audio_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    results = []
    processed = 0

    # Process only files directly in base_dir (no recursion into subdirectories)
    for p in sorted(base_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue

        path = p
        out_audio = audio_dir / path.name
        out_visual = visual_dir / path.name

        try:
            # audio_amplitude retained for legacy mode; use audio_snr_db to request a specific SNR (default 10 dB)
            add_noise_to_audio(path, out_audio, amplitude=audio_amplitude, audio_snr_db=audio_snr_db, overwrite=overwrite)
        except subprocess.CalledProcessError as e:
            print(f"Failed to add audio noise to {path}: {e}")
            out_audio = None

        try:
            # use gaussian distribution for visual noise by default
            add_noise_to_video(path, out_visual, strength=video_strength, distribution='u', overwrite=overwrite)
        except subprocess.CalledProcessError as e:
            print(f"Failed to add visual noise to {path}: {e}")
            out_visual = None

        results.append((str(path), str(out_audio) if out_audio else None, str(out_visual) if out_visual else None))

        processed += 1
        if max_samples is not None and processed >= max_samples:
            return results

    return results


if __name__ == '__main__':
    # Hardcoded settings (no argument parser)
    BASE_DIR = os.path.join('data', 'MELD.Raw', 'output_repeated_splits_test')
    AUDIO_AMPLITUDE = 0.02
    # Target audio SNR in dB when adding noise (signal_level - noise_level). Set to 10 dB by default.
    AUDIO_SNR_DB = 5.0
    VIDEO_STRENGTH = 60.0
    OVERWRITE = False
    MAX_SAMPLES = None

    #set current working directory
    os.chdir('/hkfs/work/workspace_haic/scratch/ulrat-masters/MasterThesis/Codebase_MasterThesis/')

    results = process_train_splits(BASE_DIR, audio_amplitude=AUDIO_AMPLITUDE, video_strength=VIDEO_STRENGTH, audio_snr_db=AUDIO_SNR_DB, overwrite=OVERWRITE, max_samples=MAX_SAMPLES)
    print(f"Processed {len(results)} files. Sample output:\n", results[:5])
