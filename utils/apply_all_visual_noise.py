import argparse
import shutil
import subprocess
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

VISUAL_CORRUPTIONS = [
    "zoom_blur", "gaussian_noise", "pixelate", "motion_blur", "fps_drop", "scale_down", "occlusion"]


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


def _probe_wh(video: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    w, h = out.split("x")
    return int(w), int(h)


def _vf_zoom_blur(severity: int) -> str:
    severity = max(1, min(5, severity))
    zoom = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}[severity]
    sigma = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}[severity]
    alpha = {1: 0.75, 2: 0.5, 3: 0.25, 4: 0.1, 5: 0.01}[severity]
    # Important: crop back to original size using iw/zoom and ih/zoom
    return (
        "split=2[base][z];"
        f"[z]scale=iw*{zoom}:ih*{zoom},"
        f"crop=w=iw/{zoom}:h=ih/{zoom}:x=(in_w-out_w)/2:y=(in_h-out_h)/2,"
        f"gblur=sigma={sigma}[zb];"
        f"[base][zb]blend=all_mode=normal:all_opacity={alpha}")


def _vf_pixelate(severity: int) -> str:
    severity = max(1, min(5, severity))
    f = {1: 0.65, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.22}[severity]
    return (
        f"scale=trunc(iw*{f}/2)*2:trunc(ih*{f}/2)*2:flags=neighbor,"
        "scale=iw:ih:flags=neighbor"
    )


def _vf_occlusion(severity: int) -> str:
    severity = max(1, min(5, severity))
    frac = {1: 0.10, 2: 0.18, 3: 0.26, 4: 0.34, 5: 0.45}[severity]
    # center box occlusion
    return f"drawbox=x=iw*0.5-iw*{frac}/2:y=ih*0.5-ih*{frac}/2:w=iw*{frac}:h=ih*{frac}:color=black@1:t=fill"


def _vf_scale_down(severity: int) -> str:
    severity = max(1, min(5, severity))
    r = {1: 0.85, 2: 0.70, 3: 0.55, 4: 0.42, 5: 0.30}[severity]
    return f"scale=trunc(iw*{r}/2)*2:trunc(ih*{r}/2)*2,scale=iw:ih"


def _vf_fps_drop(severity: int) -> str:
    severity = max(1, min(5, severity))
    fps = {1: 24, 2: 20, 3: 15, 4: 12, 5: 8}[severity]
    return f"fps={fps}"


def _vf_gaussian_noise(severity: int) -> str:
    severity = max(1, min(5, severity))
    strength = {1: 8, 2: 16, 3: 28, 4: 42, 5: 60}[severity]
    return f"noise=alls={strength}:c0f=t+u"


def _vf_motion_blur(severity: int) -> str:
    severity = max(1, min(5, severity))
    frames = {1: 4, 2: 6, 3: 8, 4: 10, 5: 14}[severity]
    weight_bank = {
        4: "1 0.7 0.45 0.3",
        6: "1 0.8 0.6 0.45 0.3 0.2",
        8: "1 0.85 0.7 0.55 0.42 0.32 0.24 0.18",
        10: "1 0.88 0.75 0.62 0.50 0.40 0.32 0.25 0.19 0.14",
        14: "1 0.9 0.82 0.74 0.66 0.58 0.50 0.43 0.36 0.30 0.24 0.19 0.15 0.12",
    }
    weights = weight_bank[frames]
    return f"tmix=frames={frames}:weights='{weights}'"


def _build_vf(corruption: str, severity: int) -> str:
    if corruption == "pixelate":
        return _vf_pixelate(severity)
    if corruption == "gaussian_noise":
        return _vf_gaussian_noise(severity)
    if corruption == "motion_blur":
        return _vf_motion_blur(severity)
    if corruption == "fps_drop":
        return _vf_fps_drop(severity)
    if corruption == "scale_down":
        return _vf_scale_down(severity)
    if corruption == "occlusion":
        return _vf_occlusion(severity)
    if corruption == "zoom_blur":
        return _vf_zoom_blur(severity)
    raise ValueError(f"Unknown corruption: {corruption}")


def apply_visual_corruption(in_path: Path, out_path: Path, corruption: str, severity: int, overwrite: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vf = _build_vf(corruption, severity)
    vf = f"{vf},scale=trunc(iw/2)*2:trunc(ih/2)*2"
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
        videos_out = combo_root
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
