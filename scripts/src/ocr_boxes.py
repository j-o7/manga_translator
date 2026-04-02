#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
import cv2
import numpy as np
from paddleocr import PaddleOCR

def quad_to_xyxy(quad):
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

def draw_quad(img, quad, label):
    quad = np.array(quad, dtype=np.int32)
    cv2.polylines(img, [quad], True, (0, 255, 0), 2)
    x, y = quad[0]
    cv2.putText(img, label, (int(x), int(max(0, y-5))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/pages_png")
    ap.add_argument("--out_json_dir", default="outputs/ocr_json")
    ap.add_argument("--out_vis_dir", default="outputs/ocr_vis")
    ap.add_argument("--lang", default="japan", help="PaddleOCR language model (japan works well)")
    ap.add_argument("--use_angle_cls", action="store_true", help="enable angle classifier (slower, sometimes helps)")
    ap.add_argument("--min_conf", type=float, default=0.50)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_json_dir = Path(args.out_json_dir); out_json_dir.mkdir(parents=True, exist_ok=True)
    out_vis_dir = Path(args.out_vis_dir); out_vis_dir.mkdir(parents=True, exist_ok=True)

    # Note: "japan" includes Japanese text OCR model in PaddleOCR
    ocr = PaddleOCR(use_angle_cls=args.use_angle_cls, lang=args.lang)

    pages = sorted(in_dir.glob("*.png"))
    if not pages:
        raise SystemExit(f"No .png pages found in {in_dir}")

    for page_path in pages:
        img = cv2.imread(str(page_path))
        if img is None:
            print(f"Skip unreadable: {page_path}")
            continue

        result = ocr.ocr(img, cls=args.use_angle_cls)
        items = []
        vis = img.copy()

        idx = 0
        # result is typically: [ [ [quad], (text, conf) ], ... ] per page
        for line in (result[0] if isinstance(result, list) and len(result) > 0 else []):
            quad, (text, conf) = line[0], line[1]
            if conf < args.min_conf:
                continue
            xyxy = quad_to_xyxy(quad)

            items.append({
                "id": idx,
                "bbox": [[float(p[0]), float(p[1])] for p in quad],
                "bbox_xyxy": xyxy,
                "text_ja": text,
                "conf": float(conf)
            })
            draw_quad(vis, quad, f"{idx}:{conf:.2f}")
            idx += 1

        out = {"page_id": page_path.stem, "items": items}
        (out_json_dir / f"{page_path.stem}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        cv2.imwrite(str(out_vis_dir / f"{page_path.stem}.png"), vis)

        print(f"{page_path.name}: {len(items)} text boxes -> {out_json_dir}/{page_path.stem}.json")

if __name__ == "__main__":
    main()