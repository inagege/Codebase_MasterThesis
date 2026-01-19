import argparse
import shutil
import subprocess
import tempfile
import math
from pathlib import Path
from typing import Tuple

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

VISUAL_CORRUPTIONS = ["moving_occlusion"]
    #"motion_blur", "zoom_blur", "gaussian_noise", "pixelate", "motion_blur", "fps_drop", "scale_down", "occlusion"]


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


def _probe_fps_duration(video: Path) -> Tuple[float, float]:
    """Return (fps, duration_seconds) for the given video."""
    # get avg_frame_rate
    cmd_fps = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "csv=p=0",
        str(video),
    ]
    fps_raw = subprocess.check_output(cmd_fps, text=True).strip()
    if "/" in fps_raw:
        num, den = fps_raw.split("/")
        try:
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        except Exception:
            fps = 0.0
    else:
        try:
            fps = float(fps_raw)
        except Exception:
            fps = 0.0

    cmd_dur = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(video),
    ]
    dur_raw = subprocess.check_output(cmd_dur, text=True).strip()
    try:
        duration = float(dur_raw)
    except Exception:
        duration = 0.0

    # Guard fps
    if fps <= 0:
        fps = 25.0
    if duration <= 0:
        duration = 1.0
    return fps, duration


def _generate_moving_overlay(in_video: Path, overlay_path: Path, severity: int) -> bool:
    """Generate an overlay MOV with alpha channel that contains a moving black rectangle.

    The occluder follows a seeded random-walk trajectory (bounces at edges).
    Seeding with the video path + severity makes the movement deterministic per-video
    while still appearing random. Returns True on success, False on failure.
    """
    try:
        from PIL import Image, ImageDraw
        import random
    except Exception:
        return False

    w, h = _probe_wh(in_video)
    fps, duration = _probe_fps_duration(in_video)
    n_frames = max(1, int(math.ceil(duration * fps)))

    # size of occluder
    frac = {1: 0.1, 2: 0.2, 3: 0.35, 4: 0.50, 5: 0.65}[severity]
    box_w = int(round(w * frac))
    box_h = int(round(h * frac))

    # Seed RNG deterministically per-video so results are repeatable
    seed_val = (hash(str(in_video)) ^ (severity << 16)) & 0xFFFFFFFF
    rnd = random.Random(seed_val)

    # speed fraction (fraction of full travel per frame)
    speed_frac = {1: 0.02, 2: 0.035, 3: 0.06, 4: 0.10, 5: 0.16}[severity]

    max_x = max(0, w - box_w)
    max_y = max(0, h - box_h)

    # initial position (randomized near center)
    x = int(round(max_x * 0.5 + (rnd.random() - 0.5) * max_x * 0.2))
    y = int(round(max_y * 0.5 + (rnd.random() - 0.5) * max_y * 0.2))

    # initial velocity (pixels per frame)
    vx = (rnd.uniform(-1, 1) * speed_frac) * max_x
    vy = (rnd.uniform(-1, 1) * speed_frac) * max_y

    # acceleration perturbation magnitude
    accel_x = speed_frac * max_x * 0.25
    accel_y = speed_frac * max_y * 0.25

    # occasionally change direction more strongly every few frames
    change_interval = max(1, n_frames // max(3, int(2 + rnd.random() * 4)))

    # create temporary dir for PNG frames
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for i in range(n_frames):
            # small random perturbation to velocity
            vx += rnd.uniform(-accel_x, accel_x) * 0.02
            vy += rnd.uniform(-accel_y, accel_y) * 0.02

            # stronger random nudge at intervals
            if (i % change_interval) == 0:
                vx += rnd.uniform(-accel_x, accel_x) * 0.2
                vy += rnd.uniform(-accel_y, accel_y) * 0.2

            # clamp velocity to reasonable bounds
            max_vx = max(1.0, speed_frac * max_x * 2.0)
            max_vy = max(1.0, speed_frac * max_y * 2.0)
            vx = max(-max_vx, min(max_vx, vx))
            vy = max(-max_vy, min(max_vy, vy))

            # update position
            x = int(round(x + vx))
            y = int(round(y + vy))

            # bounce at edges
            if x < 0:
                x = -x
                vx = -vx
            if x > max_x:
                x = max_x - (x - max_x)
                vx = -vx
            if y < 0:
                y = -y
                vy = -vy
            if y > max_y:
                y = max_y - (y - max_y)
                vy = -vy

            # ensure integers within bounds
            x = max(0, min(max_x, x))
            y = max(0, min(max_y, y))

            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            # draw opaque black rectangle
            draw.rectangle([x, y, x + box_w - 1, y + box_h - 1], fill=(0, 0, 0, 255))

            fname = td_path / f"frame_{i:06d}.png"
            img.save(fname, "PNG")

        # assemble PNG sequence into a MOV with png codec (preserves alpha)
        # Use framerate equal to source fps
        seq_pattern = str(td_path / "frame_%06d.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", seq_pattern,
            "-c:v", "png",
            "-pix_fmt", "rgba",
            str(overlay_path),
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            return False

    return True


def _vf_zoom_blur(severity: int) -> str:
    zoom = {1: 1, 2: 3, 3: 5, 4: 7, 5: 10}[severity]
    sigma = {1: 2, 2: 6, 3: 10, 4: 14, 5: 18}[severity]
    alpha = {1: 0.75, 2: 0.25, 3: 0.01, 4: 0.005, 5: 0.001}[severity]
    # Important: crop back to original size using iw/zoom and ih/zoom
    return (
        "split=2[base][z];"
        f"[z]scale=iw*{zoom}:ih*{zoom},"
        f"crop=w=iw/{zoom}:h=ih/{zoom}:x=(in_w-out_w)/2:y=(in_h-out_h)/2,"
        f"gblur=sigma={sigma}[zb];"
        f"[base][zb]blend=all_mode=normal:all_opacity={alpha}")


def _vf_pixelate(severity: int) -> str:
    f = {1: 0.65, 2: 0.4, 3: 0.2, 4: 0.1, 5: 0.05}[severity]
    return (
        f"scale=trunc(iw*{f}/2)*2:trunc(ih*{f}/2)*2:flags=neighbor,"
        "scale=iw:ih:flags=neighbor"
    )


def _vf_occlusion(severity: int) -> str:
    frac = {1: 0.10, 2: 0.26, 3: 0.45, 4: 0.66, 5: 0.85}[severity]
    # center box occlusion
    return f"drawbox=x=iw*0.5-iw*{frac}/2:y=ih*0.5-ih*{frac}/2:w=iw*{frac}:h=ih*{frac}:color=black@1:t=fill"


def _vf_scale_down(severity: int) -> str:
    r = {1: 0.85, 2: 0.55, 3: 0.30, 4: 0.20, 5: 0.10}[severity]
    return f"scale=trunc(iw*{r}/2)*2:trunc(ih*{r}/2)*2,scale=iw:ih"


def _vf_fps_drop(severity: int) -> str:
    fps = {1: 24, 2: 15, 3: 8, 4: 4, 5: 2}[severity]
    return f"fps={fps}"


def _vf_gaussian_noise(severity: int) -> str:
    strength = {1: 8, 2: 28, 3: 60, 4: 80, 5: 100}[severity]
    return f"noise=alls={strength}:c0f=t+u"


def _vf_motion_blur(severity: int) -> str:
    frames = {1: 4, 2: 8, 3: 14, 4: 18, 5: 22}[severity]
    weight_bank = {
        4: "1 0.7 0.45 0.3",
        8: "1 0.85 0.7 0.55 0.42 0.32 0.24 0.18",
        14: "1 0.9 0.82 0.74 0.66 0.58 0.50 0.43 0.36 0.30 0.24 0.19 0.15 0.12",
        18: "1 0.96 0.92 0.88 0.84 0.80 0.76 0.72 0.68 0.64 0.60 0.56 0.52 0.48 0.44 0.40 0.36 0.32",
        22: "1 0.97 0.94 0.91 0.88 0.85 0.82 0.79 0.76 0.73 0.70 0.67 0.64 0.61 0.58 0.55 0.52 0.49 0.46 0.43 0.40 0.37",
    }
    weights = weight_bank[frames]
    return f"tmix=frames={frames}:weights='{weights}'"


def _vf_moving_occlusion(severity: int) -> str:
    """Create a moving occlusion if supported, otherwise fallback to a static center occlusion.

    Many ffmpeg builds don't support per-frame evaluation for drawbox (eval option missing),
    which prevents animating the box. To keep the tool robust across environments we
    fallback to a center occlusion sized by severity. If you need animated occlusion,
    use an ffmpeg build that supports drawbox's eval option, or provide an alternative
    filtergraph that overlays a moving colored box.
    """
    # Use same sizing as _vf_occlusion for compatibility
    frac = {1: 0.05, 2: 0.12, 3: 0.25, 4: 0.40, 5: 0.65}[severity]
    return f"drawbox=x=iw*0.5-iw*{frac}/2:y=ih*0.5-ih*{frac}/2:w=iw*{frac}:h=ih*{frac}:color=black@1:t=fill"


def _build_vf(corruption: str, severity: int) -> str:
    severity = max(1, min(5, severity))
    if corruption == "pixelate":
        return _vf_pixelate(severity)
    if corruption == "gaussian_noise":
        return _vf_gaussian_noise(severity)
    if corruption == "motion_blur":
        return _vf_motion_blur(severity)
    if corruption == "moving_occlusion":
        return _vf_moving_occlusion(severity)
    if corruption == "fps_drop":
        return _vf_fps_drop(severity)
    if corruption == "scale_down":
        return _vf_scale_down(severity)
    if corruption == "occlusion":
        return _vf_occlusion(severity)
    if corruption == "zoom_blur":
        return _vf_zoom_blur(severity)
    raise ValueError(f"Unknown corruption: {corruption}")


def _has_audio(video: Path) -> bool:
    """Return True if the input file contains at least one audio stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return bool(out)


def apply_visual_corruption(in_path: Path, out_path: Path, corruption: str, severity: int, overwrite: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Special-case moving occlusion: try to generate an overlay video with alpha and composite it.
    if corruption == "moving_occlusion":
        # temporary files
        tmp = out_path.with_suffix(out_path.suffix + ".tmp.mp4")
        overlay_tmp = out_path.with_suffix(out_path.suffix + ".overlay.mov")

        # generate overlay; if successful, composite overlay over input during encode
        ok = _generate_moving_overlay(in_path, overlay_tmp, severity)
        has_audio = _has_audio(in_path)
        if ok:
            # Use filter_complex to overlay the alpha-enabled overlay
            # label the output video as [v]
            filter_complex = "[0:v][1:v]overlay=0:0:format=auto[v]"
            enc_cmd = [
                "ffmpeg", "-y" if overwrite else "-n",
                "-i", str(in_path),
                "-i", str(overlay_tmp),
                "-filter_complex", filter_complex,
                "-map", "[v]",
            ]
            if has_audio:
                enc_cmd += ["-map", "0:a:0", "-ac", "1", "-c:a", "aac", "-b:a", "128k"]
            else:
                enc_cmd += ["-an"]
            enc_cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", str(tmp)]

            _run(enc_cmd)

            # remux to final output (drop metadata/subtitle/data)
            remux_cmd = ["ffmpeg", "-y" if overwrite else "-n", "-i", str(tmp), "-map", "0:v:0"]
            if has_audio:
                remux_cmd += ["-map", "0:a:0"]
            remux_cmd += ["-c", "copy", "-dn", "-sn", "-map_metadata", "-1", "-movflags", "+faststart", str(out_path)]

            _run(remux_cmd)

            # cleanup
            try:
                overlay_tmp.unlink()
            except Exception:
                pass
            try:
                tmp.unlink()
            except Exception:
                pass

            return
        # if overlay generation failed, fall through to static/encode flow below

    # Default flow: use vf filter (static occlusion or other filters)
    vf = _build_vf(corruption, severity)
    vf = f"{vf},scale=trunc(iw/2)*2:trunc(ih/2)*2"

    # Encode to a temporary file first. Some input files contain binary "data" tracks
    # that can survive complex filtergraphs; remuxing the encoded file while explicitly
    # mapping only the desired streams guarantees the final file won't include those.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp.mp4")

    # First pass: encode video (and audio if present) to tmp
    has_audio = _has_audio(in_path)
    enc_cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-i", str(in_path),
        "-vf", vf,
    ]
    if has_audio:
        enc_cmd += ["-map", "0:v:0", "-map", "0:a:0", "-ac", "1", "-c:a", "aac", "-b:a", "128k"]
    else:
        enc_cmd += ["-map", "0:v:0", "-an"]
    enc_cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", str(tmp)]

    _run(enc_cmd)

    # Second pass: remux (copy) from tmp to final output while dropping data/subtitle streams
    # and metadata. Use -movflags +faststart to improve QuickTime compatibility.
    remux_cmd = ["ffmpeg", "-y" if overwrite else "-n", "-i", str(tmp), "-map", "0:v:0"]
    if has_audio:
        remux_cmd += ["-map", "0:a:0"]
    remux_cmd += ["-c", "copy", "-dn", "-sn", "-map_metadata", "-1", "-movflags", "+faststart", str(out_path)]

    _run(remux_cmd)

    # Remove temporary file
    try:
        tmp.unlink()
    except Exception:
        pass


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
