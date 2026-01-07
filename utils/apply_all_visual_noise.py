import argparse
import shutil
import subprocess
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

VISUAL_CORRUPTIONS = ["gaussian_noise"] #"zoom_blur", "pixelate", "motion_blur",

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

def _vf_zoom_blur(severity: int) -> str:
    severity = max(1, min(5, severity))
    zoom = {1: 1.11, 2: 1.16, 3: 1.21, 4: 1.26, 5: 1.33}[severity]
    sigma = {1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0}[severity]
    return f"scale=iw*{zoom}:ih*{zoom},crop=iw:ih:(in_w-out_w)/2:(in_h-out_h)/2,gblur=sigma={sigma}"

def _vf_pixelate(severity: int) -> str:
    severity = max(1, min(5, severity))
    f = {1: 0.65, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.22}[severity]
    return f"scale=iw*{f}:ih*{f}:flags=neighbor,scale=iw:ih:flags=neighbor"

def _vf_gaussian_noise(severity: int) -> str:
    severity = max(1, min(5, severity))
    strength = {1: 5, 2: 10, 3: 18, 4: 28, 5: 40}[severity]
    return f"noise=alls={strength}:allf=u"

def _vf_motion_blur(severity: int) -> str:
    severity = max(1, min(5, severity))
    frames = {1: 3, 2: 5, 3: 7, 4: 9, 5: 11}[severity]
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
    # Ensure final dimensions are even for libx264
    vf = f"{vf},scale=trunc(iw/2)*2:trunc(ih/2)*2"

    # Re-encode audio to mono AAC to avoid multi-channel encoding failures
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
    ap = argparse.ArgumentParser("Apply ALL visual perturbations to a video directory.")
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--out_dir", required=True, help="Base output directory.")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    _ffmpeg()
    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = sorted(list(_iter_videos(videos_dir, args.recursive)))
    if not vids:
        raise RuntimeError(f"No videos found in {videos_dir}")

    for corr in VISUAL_CORRUPTIONS:
        combo_root = out_dir / f"V={corr}_S={args.severity}"
        videos_out = combo_root / "videos"
        videos_out.mkdir(parents=True, exist_ok=True)

        for vid in vids:
            rel = vid.relative_to(videos_dir) if videos_dir.is_dir() else Path(vid.name)
            out_video = videos_out / rel
            out_video.parent.mkdir(parents=True, exist_ok=True)

            if out_video.exists() and not args.overwrite:
                continue

            apply_visual_corruption(vid, out_video, corr, args.severity, args.overwrite)

        print(f"[OK] Finished visual corruption: {combo_root}")

    print("[DONE] All visual perturbations applied.")

if __name__ == "__main__":
    main()
