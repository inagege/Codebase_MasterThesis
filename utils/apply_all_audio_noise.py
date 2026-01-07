import argparse
import shutil
import subprocess
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

AUDIO_CORRUPTIONS = ["snr_white", "reverb", "clipping", "mp3", "bandlimit"]

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
        for p in inp.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                yield p
    else:
        for p in inp.iterdir():
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                yield p

def _snr_to_anoisesrc_amplitude(snr_db: float) -> float:
    # pragmatic mapping
    k = 0.25
    return float(k * (10 ** (-snr_db / 20.0)))

def extract_audio_only(in_video: Path, out_wav: Path, overwrite: bool, sr: int):
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-i", str(in_video),
        "-vn", "-ac", "1", "-ar", str(sr),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    _run(cmd)

def apply_audio_corruption(in_path: Path, out_video: Path, corruption: str, severity: int, overwrite: bool):
    severity = max(1, min(5, severity))
    out_video.parent.mkdir(parents=True, exist_ok=True)

    if corruption == "snr_white":
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
            "-shortest", str(out_video),
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
        br = {1: "96k", 2: "64k", 3: "48k", 4: "32k", 5: "24k"}[severity]
        cmd = [
            "ffmpeg", "-y" if overwrite else "-n",
            "-i", str(in_path),
            "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "copy",
            "-c:a", "libmp3lame", "-b:a", br, "-ac", "1",
            "-shortest", str(out_video),
        ]
        _run(cmd)
        return
    else:
        raise ValueError(f"Unknown corruption: {corruption}")

    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-i", str(in_path),
        "-map", "0:v:0", "-map", "0:a:0",
        "-c:v", "copy",
        "-af", af,
        "-c:a", "aac", "-ac", "1",
        "-shortest", str(out_video),
    ]
    _run(cmd)

def main():
    ap = argparse.ArgumentParser("Apply ALL audio perturbations to a video directory.")
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--out_dir", required=True, help="Base output directory.")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--audio_sr", type=int, default=16000)
    args = ap.parse_args()

    _ffmpeg()
    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = sorted(list(_iter_videos(videos_dir, args.recursive)))
    if not vids:
        raise RuntimeError(f"No videos found in {videos_dir}")

    for corr in AUDIO_CORRUPTIONS:
        combo_root = out_dir / f"A={corr}_S={args.severity}"
        videos_out = combo_root / "videos"
        audio_only_out = videos_out / "audio_only"
        videos_out.mkdir(parents=True, exist_ok=True)
        audio_only_out.mkdir(parents=True, exist_ok=True)

        for vid in vids:
            rel = vid.relative_to(videos_dir) if videos_dir.is_dir() else Path(vid.name)
            out_video = videos_out / rel
            out_video.parent.mkdir(parents=True, exist_ok=True)

            if out_video.exists() and not args.overwrite:
                continue

            apply_audio_corruption(vid, out_video, corr, args.severity, args.overwrite)

            out_wav = (audio_only_out / rel).with_suffix(".wav")
            extract_audio_only(out_video, out_wav, args.overwrite, args.audio_sr)

        print(f"[OK] Finished audio corruption: {combo_root}")

    print("[DONE] All audio perturbations applied.")

if __name__ == "__main__":
    main()
