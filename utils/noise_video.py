#!/usr/bin/env python3
import argparse
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

def _vf_zoom_blur(severity: int) -> str:
    """
    Practical video approximation of zoom blur:
      scale up -> center crop back -> gaussian blur
    The paper’s zoom blur severity corresponds to increasing zoom factor. :contentReference[oaicite:8]{index=8}
    """
    severity = max(1, min(5, severity))
    zoom = {1: 1.11, 2: 1.16, 3: 1.21, 4: 1.26, 5: 1.33}[severity]
    sigma = {1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0}[severity]
    return f"scale=iw*{zoom}:ih*{zoom},crop=iw:ih:(in_w-out_w)/2:(in_h-out_h)/2,gblur=sigma={sigma}"

def _vf_pixelate(severity: int) -> str:
    """
    Pixelate by downscaling then upscaling with nearest-neighbor.
    """
    severity = max(1, min(5, severity))
    # smaller factor => more pixelation
    f = {1: 0.65, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.22}[severity]
    return f"scale=iw*{f}:ih*{f}:flags=neighbor,scale=iw:ih:flags=neighbor"

def _vf_gaussian_noise(severity: int) -> str:
    """
    ffmpeg noise filter (synthetic Gaussian-like noise option).
    """
    severity = max(1, min(5, severity))
    strength = {1: 5, 2: 10, 3: 18, 4: 28, 5: 40}[severity]
    return f"noise=alls={strength}:allf=u"

def _vf_motion_blur(severity: int) -> str:
    """
    Approximate motion blur via temporal mixing of frames.
    """
    severity = max(1, min(5, severity))
    frames = {1: 3, 2: 5, 3: 7, 4: 9, 5: 11}[severity]
    # tmix averages a window of frames
    return f"tmix=frames={frames}:weights='1'"

def _build_vf(corruption: str, severity: int) -> str:
    if corruption == "zoom_blur":
        return _vf_zoom_blur(severity)
    if corruption == "pixelate":
        return _vf_pixelate(severity)
    if corruption == "gaussian_noise":
        return _vf_gaussian_noise(severity)
    if corruption == "motion_blur":
        return _vf_motion_blur(severity)
    raise ValueError(f"Unknown corruption: {corruption}")

def apply_visual_corruption(in_path: Path, out_path: Path, corruption: str, severity: int, overwrite: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vf = _build_vf(corruption, severity)

    # Ensure final dimensions are even (libx264 requires even width/height)
    vf = f"{vf},scale=trunc(iw/2)*2:trunc(ih/2)*2"

    # Re-encode audio to a safe mono AAC layout to avoid multi-channel encoder failures
    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-i", str(in_path),
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-ac", "1", "-c:a", "aac", "-b:a", "128k",
        "-shortest", str(out_path),
    ]
    _run(cmd)

def main():
    ap = argparse.ArgumentParser("Apply visual corruptions to video files.")
    ap.add_argument("--input", required=True, help="Video file or directory.")
    ap.add_argument("--output_dir", required=True, help="Directory to save corrupted videos.")
    ap.add_argument("--corruption", required=True,
                    choices=["zoom_blur", "pixelate", "motion_blur", "gaussian_noise"],
                    help="Visual corruption type.")
    ap.add_argument("--severity", type=int, default=3, help="Severity 1..5")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    _ffmpeg()
    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for vid in _iter_videos(inp, args.recursive):
        out_path = (out_dir / vid.relative_to(inp)) if inp.is_dir() else (out_dir / vid.name)
        apply_visual_corruption(vid, out_path, args.corruption, args.severity, args.overwrite)
        print(f"[OK] {vid} -> {out_path}")

if __name__ == "__main__":
    main()
