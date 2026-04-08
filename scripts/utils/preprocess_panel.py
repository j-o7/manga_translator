#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

import cv2
import numpy as np

def resize_keep_aspect(img_bgr, max_side=2400):
    h, w = img_bgr.shape[:2]
    scale = min(max_side / max(h, w), 1.0)  # only shrink if needed
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized, scale

def preprocess_for_manga(img_bgr, scale=1.0, max_side=2400, invert=False):
    if scale != 1.0:
        img_bgr = cv2.resize(
            img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    img_bgr, resize_scale = resize_keep_aspect(img_bgr, max_side=max_side)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=8)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if invert:
        th = 255 - th

    th_bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return th_bgr, resize_scale


def convert_one(in_path: Path, out_path: Path) -> None:
    # Handle EXIF rotation + ensure RGB (manga scans often include EXIF orientation)
    with Image.open(in_path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA"):
            # Flatten alpha onto white background
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        preprocess_for_manga(np.array(im))  # just to test the function; not modifying im here
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, format="PNG", optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="data/raw")
    ap.add_argument("--out_dir", type=str, default="data/pages_png")
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if args.recursive else "*"
    files = [p for p in in_dir.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]

    if not files:
        print(f"No supported images found in {in_dir}")
        return

    for p in sorted(files):
        out_path = out_dir / (p.stem + ".png")
        if out_path.exists():
            print(f"Skipping {p} (already exists: {out_path})")
            continue
        convert_one(p, out_path)
        print(f"Wrote: {out_path}")

    print(f"Done. Converted {len(files)} file(s).")


if __name__ == "__main__":
    main()