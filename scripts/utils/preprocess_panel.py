#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

import cv2
import numpy as np

def preprocess_for_manga(img_bgr, scale=2.0, invert=False):
    # upscale
    if scale != 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # mild denoise
    gray = cv2.fastNlMeansDenoising(gray, h=8)

    # binarize (try Otsu; adaptive also works)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if invert:
        th = 255 - th

    # return 3-channel because PaddleOCR often expects 3-ch
    th_bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return th_bgr


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