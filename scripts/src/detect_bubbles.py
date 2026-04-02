#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Bubble:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    score: float                     # heuristic score


def clamp_bbox(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, x); y = max(0, y)
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def detect_bubbles(
    img_bgr: np.ndarray,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.35,
    min_wh: int = 40,
) -> List[Bubble]:
    """
    Heuristic bubble detection:
    - convert to gray
    - denoise + edge detect
    - close gaps
    - find external contours
    - filter by area, aspect ratio, and 'whiteness' inside bbox
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Light denoise to stabilize edges
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(gray_blur, 50, 150)

    # Close gaps between edge fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Optional: fill interiors by dilating a bit
    dil = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = min_area_ratio * (W * H)
    max_area = max_area_ratio * (W * H)

    bubbles: List[Bubble] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_wh or h < min_wh:
            continue

        area = w * h
        if area < min_area or area > max_area:
            continue

        # Filter extreme aspect ratios (speech bubbles usually not super skinny)
        ar = w / float(h)
        if ar < 0.2 or ar > 5.0:
            continue

        # Whiteness check: bubbles often have lighter interior than surrounding art
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        # Percent of pixels above a brightness threshold
        bright = np.mean(roi > 200)
        # Also allow darker bubbles sometimes; use mean as weak cue
        mean_intensity = float(np.mean(roi))

        # Heuristic scoring: prefer bright interiors and medium-sized regions
        size_score = 1.0 - abs((area / (W * H)) - 0.05)  # peak around 5% of page
        score = 0.55 * bright + 0.25 * (mean_intensity / 255.0) + 0.20 * np.clip(size_score, 0, 1)

        # Basic cutoff; tune if needed
        if bright < 0.08 and mean_intensity < 150:
            continue

        bubbles.append(Bubble(bbox=(x, y, w, h), score=score))

    # Sort by score descending
    bubbles.sort(key=lambda b: b.score, reverse=True)

    # Light dedup: remove boxes that are almost fully contained in a higher-scoring box
    final: List[Bubble] = []
    for b in bubbles:
        x, y, w, h = b.bbox
        bx1, by1, bx2, by2 = x, y, x + w, y + h
        contained = False
        for k in final:
            x2, y2, w2, h2 = k.bbox
            kx1, ky1, kx2, ky2 = x2, y2, x2 + w2, y2 + h2
            inter_x1, inter_y1 = max(bx1, kx1), max(by1, ky1)
            inter_x2, inter_y2 = min(bx2, kx2), min(by2, ky2)
            inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            b_area = w * h
            if b_area > 0 and inter_area / b_area > 0.90:
                contained = True
                break
        if not contained:
            final.append(b)

    return final


def draw_boxes(img_bgr: np.ndarray, bubbles: List[Bubble]) -> np.ndarray:
    vis = img_bgr.copy()
    for i, b in enumerate(bubbles):
        x, y, w, h = b.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            vis, f"{i}:{b.score:.2f}", (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
        )
    return vis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="data/pages_png")
    ap.add_argument("--overlay_dir", type=str, default="outputs/bubbles_overlay")
    ap.add_argument("--crops_dir", type=str, default="outputs/bubble_crops")
    ap.add_argument("--min_area_ratio", type=float, default=0.002)
    ap.add_argument("--max_area_ratio", type=float, default=0.35)
    ap.add_argument("--min_wh", type=int, default=40)
    ap.add_argument("--topk", type=int, default=0, help="Keep only top-K bubbles per page (0 = keep all)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    overlay_dir = Path(args.overlay_dir)
    crops_dir = Path(args.crops_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    pages = sorted([p for p in in_dir.glob("*.png") if p.is_file()])
    if not pages:
        raise SystemExit(f"No .png files found in {in_dir}. Run conversion first.")

    for page_path in pages:
        img = cv2.imread(str(page_path))
        if img is None:
            print(f"Skipping unreadable: {page_path}")
            continue

        bubbles = detect_bubbles(
            img,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            min_wh=args.min_wh,
        )

        if args.topk > 0:
            bubbles = bubbles[: args.topk]

        # Save overlay
        overlay = draw_boxes(img, bubbles)
        out_overlay = overlay_dir / page_path.name
        cv2.imwrite(str(out_overlay), overlay)

        # Save crops
        page_crop_dir = crops_dir / page_path.stem
        page_crop_dir.mkdir(parents=True, exist_ok=True)

        H, W = img.shape[:2]
        for i, b in enumerate(bubbles):
            x, y, w, h = b.bbox
            x, y, w, h = clamp_bbox(x, y, w, h, W, H)
            crop = img[y:y+h, x:x+w]
            cv2.imwrite(str(page_crop_dir / f"bubble_{i:03d}.png"), crop)

        print(f"{page_path.name}: {len(bubbles)} bubbles | overlay={out_overlay} | crops={page_crop_dir}")

    print("Done.")


if __name__ == "__main__":
    main()