#!/usr/bin/env python3
import argparse
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_CORRUPTIONS = [
    "gaussian_noise",
    "motion_blur",
    "zoom_blur",
    "pixelate",
    "jpeg",
    "scale_down",
    "occlusion",
]


def _iter_images(inp: Path, recursive: bool):
    if inp.is_file():
        if inp.suffix.lower() in IMAGE_EXTS:
            yield inp
        return

    if recursive:
        for p in inp.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p
    else:
        for p in inp.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p


def _apply_gaussian_noise(img: Image.Image, severity: int) -> Image.Image:
    # Tuned to match the visual-noise severity progression used for video.
    sigma_map = {1: 8.0, 2: 28.0, 3: 60.0, 4: 80.0, 5: 100.0}
    # Mid severities are intentionally softer for better perceptual spacing.
    alpha_map = {1: 0.18, 2: 0.25, 3: 0.40, 4: 0.70, 5: 0.85}
    sigma = sigma_map[severity]
    alpha = alpha_map[severity]
    noise = Image.effect_noise(img.size, sigma).convert("L").convert("RGB")
    return Image.blend(img, noise, alpha=alpha)


def _apply_motion_blur(img: Image.Image, severity: int) -> Image.Image:
    radius = {1: 1.0, 2: 2.0, 3: 3.5, 4: 5.0, 5: 7.0}[severity]
    return img.filter(ImageFilter.BoxBlur(radius=radius))


def _apply_zoom_blur(img: Image.Image, severity: int) -> Image.Image:
    steps = {1: 6, 2: 10, 3: 14, 4: 20, 5: 28}[severity]
    max_zoom = {1: 1.04, 2: 1.08, 3: 1.14, 4: 1.22, 5: 1.32}[severity]
    w, h = img.size

    acc = img.copy().convert("RGB")
    for i in range(1, steps + 1):
        z = 1.0 + (max_zoom - 1.0) * (i / steps)
        zw = max(1, int(round(w * z)))
        zh = max(1, int(round(h * z)))
        scaled = img.resize((zw, zh), Image.Resampling.BILINEAR)
        x0 = (zw - w) // 2
        y0 = (zh - h) // 2
        cropped = scaled.crop((x0, y0, x0 + w, y0 + h))
        acc = Image.blend(acc, cropped, alpha=1.0 / (i + 1))
    return acc


def _apply_pixelate(img: Image.Image, severity: int) -> Image.Image:
    f = {1: 0.65, 2: 0.40, 3: 0.20, 4: 0.10, 5: 0.05}[severity]
    w, h = img.size
    down = img.resize((max(1, int(w * f)), max(1, int(h * f))), Image.Resampling.NEAREST)
    return down.resize((w, h), Image.Resampling.NEAREST)


def _apply_jpeg(img: Image.Image, severity: int) -> Image.Image:
    q = {1: 50, 2: 25, 3: 10, 4: 5, 5: 2}[severity]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q, optimize=False)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    buf.close()
    return out


def _apply_scale_down(img: Image.Image, severity: int) -> Image.Image:
    r = {1: 0.85, 2: 0.55, 3: 0.30, 4: 0.20, 5: 0.10}[severity]
    w, h = img.size
    down = img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.Resampling.BILINEAR)
    return down.resize((w, h), Image.Resampling.BILINEAR)


def _apply_occlusion(img: Image.Image, severity: int) -> Image.Image:
    frac = {1: 0.10, 2: 0.26, 3: 0.45, 4: 0.66, 5: 0.85}[severity]
    w, h = img.size
    box_w = int(round(w * frac))
    box_h = int(round(h * frac))
    x0 = (w - box_w) // 2
    y0 = (h - box_h) // 2

    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(0, 0, 0))
    return out


def apply_image_corruption(in_path: Path, out_path: Path, corruption: str, severity: int):
    severity = max(1, min(5, severity))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(in_path) as pil_img:
        src_mode = pil_img.mode
        img = pil_img.convert("RGB")

    if corruption == "gaussian_noise":
        out = _apply_gaussian_noise(img, severity)
    elif corruption == "motion_blur":
        out = _apply_motion_blur(img, severity)
    elif corruption == "zoom_blur":
        out = _apply_zoom_blur(img, severity)
    elif corruption == "pixelate":
        out = _apply_pixelate(img, severity)
    elif corruption == "jpeg":
        out = _apply_jpeg(img, severity)
    elif corruption == "scale_down":
        out = _apply_scale_down(img, severity)
    elif corruption == "occlusion":
        out = _apply_occlusion(img, severity)
    else:
        raise ValueError(f"Unknown corruption: {corruption}")

    if src_mode in {"L", "RGB", "RGBA"}:
        out = out.convert(src_mode)

    out.save(out_path)


def main():
    ap = argparse.ArgumentParser("Apply ALL image perturbations to an image directory.")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True, help="Base output directory.")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(_iter_images(images_dir, args.recursive)))
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")

    for corr in IMAGE_CORRUPTIONS:
        combo_root = out_dir / f"I={corr}_S={args.severity}"
        combo_root.mkdir(parents=True, exist_ok=True)

        for img in images:
            rel = img.relative_to(images_dir) if images_dir.is_dir() else Path(img.name)
            out_img = combo_root / rel
            if out_img.exists() and not args.overwrite:
                continue
            apply_image_corruption(img, out_img, corr, args.severity)

        print(f"[OK] Finished image corruption: {combo_root}")

    print("[DONE] All image perturbations applied.")


if __name__ == "__main__":
    main()
