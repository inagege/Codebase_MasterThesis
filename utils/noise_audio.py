import argparse
import math
import shutil
import subprocess
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

def _ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")

def _run(cmd: list[str]):
    subprocess.run(cmd, check=True)

def _iter_videos(inp: Path, recursive: bool):
    if inp.is_file():
        if inp.suffix.lower() in VIDEO_EXTS:
            yield inp
        return
    if recursive:
        yield from (p for p in inp.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    else:
        yield from (p for p in inp.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)

def _snr_to_anoisesrc_amplitude(snr_db: float) -> float:
    """
    Pragmatic amplitude mapping:
      amplitude ≈ k * 10^(-SNR/20)
    (Exact SNR matching would require measuring RMS and scaling noise accordingly.)
    """
    k = 0.25
    return float(k * (10 ** (-snr_db / 20.0)))

def extract_audio_only(
    in_video_path: Path,
    out_audio_path: Path,
    overwrite: bool,
    sample_rate: int = 16000,
):
    """
    Extract audio track to WAV (PCM 16-bit, mono) for maximum compatibility.
    """
    out_audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-i", str(in_video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out_audio_path),
    ]
    _run(cmd)

def apply_audio_corruption(
    in_path: Path,
    out_video_path: Path,
    corruption: str,
    severity: int,
    overwrite: bool,
):
    """
    Apply corruption to the audio track and remux into a new video file.
    The video stream is copied unchanged.
    """
    severity = max(1, min(5, severity))
    out_video_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- corruption parameterizations (severity 1..5) ----
    if corruption == "snr_white":
        # lower SNR => stronger noise
        snr_levels = {1: 20, 2: 15, 3: 10, 4: 5, 5: 0}
        snr_db = snr_levels[severity]
        amp = _snr_to_anoisesrc_amplitude(snr_db)

        cmd = [
            "ffmpeg", "-y" if overwrite else "-n",
            "-i", str(in_path),
            "-f", "lavfi", "-i", f"anoisesrc=color=white:amplitude={amp}:r=44100",
            "-filter_complex", "[0:a][1:a]amix=inputs=2:normalize=0[aout]",
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-ac", "1",
            "-shortest", str(out_video_path),
        ]
        _run(cmd)
        return

    if corruption == "reverb":
        delays = {1: "40|70", 2: "60|90", 3: "80|120", 4: "100|160", 5: "120|200"}[severity]
        decays = {1: "0.25|0.20", 2: "0.35|0.25", 3: "0.45|0.30", 4: "0.55|0.35", 5: "0.65|0.40"}[severity]
        af = f"aecho=0.8:0.9:{delays}:{decays}"

    elif corruption == "clipping":
        limit = {1: 0.9, 2: 0.7, 3: 0.5, 4: 0.35, 5: 0.25}[severity]
        af = f"alimiter=limit={limit}"

    elif corruption == "bandlimit":
        low = {1: 120, 2: 200, 3: 300, 4: 400, 5: 500}[severity]
        high = {1: 6000, 2: 4500, 3: 3500, 4: 3000, 5: 2500}[severity]
        af = f"highpass=f={low},lowpass=f={high}"

    elif corruption == "mp3":
        # codec artifacts (audio re-encoded at low MP3 bitrate)
        br = {1: "96k", 2: "64k", 3: "48k", 4: "32k", 5: "24k"}[severity]
        cmd = [
            "ffmpeg", "-y" if overwrite else "-n",
            "-i", str(in_path),
            "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "copy",
            "-c:a", "libmp3lame", "-b:a", br, "-ac", "1",
            "-shortest", str(out_video_path),
        ]
        _run(cmd)
        return

    else:
        raise ValueError(f"Unknown corruption: {corruption}")

    # Single-stream audio filter path
    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-i", str(in_path),
        "-map", "0:v:0", "-map", "0:a:0",
        "-c:v", "copy",
        "-af", af,
        "-c:a", "aac", "-ac", "1",
        "-shortest", str(out_video_path),
    ]
    _run(cmd)

def main():
    ap = argparse.ArgumentParser("Apply audio corruptions to video files and also save audio-only outputs.")
    ap.add_argument("--input", required=True, help="Video file or directory.")
    ap.add_argument("--output_dir", required=True, help="Directory to save corrupted videos.")
    ap.add_argument(
        "--corruption", required=True,
        choices=["snr_white", "reverb", "clipping", "mp3", "bandlimit"],
        help="Audio corruption type."
    )
    ap.add_argument("--severity", type=int, default=3, help="Severity 1..5")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    # audio-only settings
    ap.add_argument("--audio_sr", type=int, default=16000, help="Sample rate for audio-only WAV. Default: 16000")
    args = ap.parse_args()

    _ffmpeg()
    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_only_dir = out_dir / "audio_only"
    audio_only_dir.mkdir(parents=True, exist_ok=True)

    for vid in _iter_videos(inp, args.recursive):
        # Preserve relative structure if input is a directory
        rel = vid.relative_to(inp) if inp.is_dir() else Path(vid.name)

        out_video_path = out_dir / rel
        out_audio_path = (audio_only_dir / rel).with_suffix(".wav")

        apply_audio_corruption(
            in_path=vid,
            out_video_path=out_video_path,
            corruption=args.corruption,
            severity=args.severity,
            overwrite=args.overwrite,
        )

        # Extract audio from the *corrupted* video, so audio-only matches exactly
        extract_audio_only(
            in_video_path=out_video_path,
            out_audio_path=out_audio_path,
            overwrite=args.overwrite,
            sample_rate=args.audio_sr,
        )

        print(f"[OK] video: {vid} -> {out_video_path}")
        print(f"[OK] audio: {out_video_path} -> {out_audio_path}")

if __name__ == "__main__":
    main()
