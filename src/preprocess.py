#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps
import cv2
import numpy as np


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

TECHNIQUES = {
    "all",
    "gray",
    "clahe",
    "otsu",
    "otsu_inv",
    "adaptive",
    "adaptive_inv",
    "bg_norm",
    "mild_binary",
}


def resize_keep_aspect(img_bgr, max_side=2400):
    h, w = img_bgr.shape[:2]
    scale = min(max_side / max(h, w), 1.0)  # only shrink if needed
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized, scale


def resize_once(img_bgr, scale=1.0, max_side=2400):
    if scale != 1.0:
        img_bgr = cv2.resize(
            img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    img_bgr, _ = resize_keep_aspect(img_bgr, max_side=max_side)
    return img_bgr


def _to_bgr(img_gray_or_bgr):
    if len(img_gray_or_bgr.shape) == 2:
        return cv2.cvtColor(img_gray_or_bgr, cv2.COLOR_GRAY2BGR)
    return img_gray_or_bgr


def preprocess_for_manga(
    img_bgr,
    technique="otsu",
    denoise_h=8,
):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=denoise_h)

    if technique == "gray":
        out = gray

    elif technique == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(gray)

    elif technique == "otsu":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif technique == "otsu_inv":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out = 255 - out

    elif technique == "adaptive":
        out = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

    elif technique == "adaptive_inv":
        out = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        out = 255 - out

    elif technique == "bg_norm":
        bg = cv2.GaussianBlur(gray, (0, 0), 15)
        norm = cv2.divide(gray, bg, scale=255)
        _, out = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif technique == "mild_binary":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, out = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    else:
        raise ValueError(
            f"Unknown technique '{technique}'. Choose from: {sorted(TECHNIQUES)}"
        )

    return _to_bgr(out)


def _load_image_rgb(in_path: Path) -> np.ndarray:
    with Image.open(in_path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")

        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="data/raw")
    ap.add_argument("--out_dir", type=str, default="data/pages_png")
    ap.add_argument("--recursive", action="store_true")

    ap.add_argument(
        "--technique",
        type=str,
        default="all",
        choices=sorted(TECHNIQUES),
        help="Preprocessing technique to apply.",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Upscale factor before resize_keep_aspect. Example: 2.0",
    )
    ap.add_argument(
        "--max_side",
        type=int,
        default=2400,
        help="Max side length after preprocessing resize.",
    )
    ap.add_argument(
        "--denoise_h",
        type=int,
        default=8,
        help="Strength for fastNlMeansDenoising.",
    )

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    pattern = "**/*" if args.recursive else "*"
    files = [p for p in in_dir.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]

    if not files:
        print(f"No supported images found in {in_dir}")
        return

    techniques_to_run = (
        [t for t in sorted(TECHNIQUES) if t != "all"]
        if args.technique == "all"
        else [args.technique]
    )

    out_root = Path(args.out_dir)
    resized_dir = out_root / "resized_original"
    resized_dir.mkdir(parents=True, exist_ok=True)

    for tech in techniques_to_run:
        (out_root / tech).mkdir(parents=True, exist_ok=True)

    written = 0

    for p in sorted(files):
        img_bgr = _load_image_rgb(p)

        resized_bgr = resize_once(
            img_bgr,
            scale=args.scale,
            max_side=args.max_side,
        )

        resized_out = resized_dir / f"{p.stem}.png"
        if not resized_out.exists():
            ok = cv2.imwrite(str(resized_out), resized_bgr)
            if not ok:
                raise RuntimeError(f"Failed to write resized original: {resized_out}")
            print(f"Wrote resized original: {resized_out}")

        for tech in techniques_to_run:
            out_path = out_root / tech / f"{p.stem}.png"
            if out_path.exists():
                print(f"Skipping {p} for '{tech}' (already exists: {out_path})")
                continue

            processed_bgr = preprocess_for_manga(
                resized_bgr,
                technique=tech,
                denoise_h=args.denoise_h,
            )

            ok = cv2.imwrite(str(out_path), processed_bgr)
            if not ok:
                raise RuntimeError(f"Failed to write output image: {out_path}")

            written += 1
            print(f"Wrote: {out_path}")

    if args.technique == "all":
        print(f"Done. Converted {len(files)} file(s) across {len(techniques_to_run)} techniques.")
    else:
        print(f"Done. Converted {len(files)} file(s) with technique='{args.technique}'.")

    print(f"Resized originals saved in: {resized_dir}")


if __name__ == "__main__":
    main()
