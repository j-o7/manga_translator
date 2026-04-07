#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
from PIL import Image, ImageDraw


DEFAULT_CONFIG: Dict[str, Any] = {
    "env": {
        "FLAGS_use_mkldnn": "0",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "FLAGS_use_new_executor": "0",
    },
    "paths": {
        "input_png_dir": "data/jap_imgs",
        "out_json_dir": "outputs/ocr_json_jap",
        "out_vis_dir": "outputs/ocr_vis_jap",
        "stats_json": "outputs/debug/ocr_stats.json",
    },
    "ocr": {
        "lang": "japan",
        "use_angle_cls": True,
        "det_db_thresh": 0.1,
        "det_db_box_thresh": 0.4,
        "det_db_unclip_ratio": 1.8,
        "det_limit_side_len": 4096,
    },
    "postprocess": {
        "merge": {
            "enabled": True,
            "expand_px": 10,
            "iou_thresh": 0.01,
            "max_center_dist": 80.0,
            "max_gap_x": 40.0,
            "max_gap_y": 30.0,
            "require_overlap_after_expand": True,
        },
        "filter": {
            "enabled": True,
            "min_conf": 0.4,
            "min_chars": 2,
            "drop_patterns": ["raw", "\\.net", "http", "www"],
        },
        "contained": {
            "enabled": True,
            "containment_thresh": 0.90,
            "prefer_keep": "bigger",
        },
    },
    "output": {
        "draw_text": False,
        "overlay_suffix": "_overlay.png",
        "save_stats": True,
    },
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    return deep_update(cfg, user_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR manga panels with PaddleOCR using a JSON config.")
    parser.add_argument("--config", required=True, help="Path to JSON config file")

    # Optional overrides
    parser.add_argument("--in_dir", help="Override input PNG directory")
    parser.add_argument("--out_json_dir", help="Override output OCR JSON directory")
    parser.add_argument("--out_vis_dir", help="Override output OCR overlay directory")
    parser.add_argument("--lang", help="Override PaddleOCR language")
    parser.add_argument("--use_angle_cls", type=str, choices=["true", "false"], help="Override angle classifier")
    parser.add_argument("--min_conf", type=float, help="Override filter.min_conf")
    parser.add_argument("--draw_text", type=str, choices=["true", "false"], help="Override output.draw_text")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N images (0 means all)")
    return parser.parse_args()


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.in_dir:
        cfg["paths"]["input_png_dir"] = args.in_dir
    if args.out_json_dir:
        cfg["paths"]["out_json_dir"] = args.out_json_dir
    if args.out_vis_dir:
        cfg["paths"]["out_vis_dir"] = args.out_vis_dir
    if args.lang:
        cfg["ocr"]["lang"] = args.lang
    if args.use_angle_cls:
        cfg["ocr"]["use_angle_cls"] = args.use_angle_cls.lower() == "true"
    if args.min_conf is not None:
        cfg["postprocess"]["filter"]["min_conf"] = args.min_conf
    if args.draw_text:
        cfg["output"]["draw_text"] = args.draw_text.lower() == "true"


def quad_to_xyxy(quad: List[List[float]]) -> List[int]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))]


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_paddle_result(raw_ocr_output: Any, image_path: Path) -> Dict[str, Any]:
    with Image.open(image_path) as im:
        width, height = im.size

    lines = raw_ocr_output
    if isinstance(raw_ocr_output, list) and len(raw_ocr_output) == 1 and isinstance(raw_ocr_output[0], list):
        if len(raw_ocr_output[0]) > 0 and isinstance(raw_ocr_output[0][0], (list, tuple)):
            lines = raw_ocr_output[0]

    items = []
    idx = 0
    for entry in lines:
        if not entry:
            continue

        quad = entry[0]
        txt_conf = entry[1] if len(entry) > 1 else ("", None)

        text = safe_str(txt_conf[0]) if isinstance(txt_conf, (list, tuple)) and len(txt_conf) > 0 else ""
        conf = float(txt_conf[1]) if isinstance(txt_conf, (list, tuple)) and len(txt_conf) > 1 and txt_conf[1] is not None else None

        x1, y1, x2, y2 = quad_to_xyxy(quad)
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

        items.append(
            {
                "id": idx,
                "bbox_xyxy": [x1, y1, x2, y2],
                "poly": [[float(p[0]), float(p[1])] for p in quad],
                "text": text,
                "conf": conf,
            }
        )
        idx += 1

    return {
        "image": str(image_path),
        "width": width,
        "height": height,
        "items": items,
        "engine": "paddleocr",
    }


def _area(box: List[int]) -> float:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _expand(box: List[int], px: int) -> List[int]:
    x1, y1, x2, y2 = box
    return [x1 - px, y1 - px, x2 + px, y2 + px]


def _center(box: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _dims(box: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return max(1, x2 - x1), max(1, y2 - y1)


def _reading_key(item: Dict[str, Any]) -> Tuple[int, int]:
    x1, y1, x2, y2 = item["bbox_xyxy"]
    return y1, x1


def merge_close_boxes(
    items: List[Dict[str, Any]],
    expand_px: int,
    iou_thresh: float,
    max_center_dist: float,
    max_gap_x: float,
    max_gap_y: float,
    require_overlap_after_expand: bool,
) -> List[Dict[str, Any]]:
    if not items:
        return []

    normalized = []
    for item in items:
        if "bbox_xyxy" not in item:
            continue
        normalized.append(
            {
                "bbox_xyxy": [int(v) for v in item["bbox_xyxy"]],
                "text": str(item.get("text", "")),
                "conf": item.get("conf", None),
                **{k: v for k, v in item.items() if k not in ["bbox_xyxy", "text", "conf"]},
            }
        )

    n = len(normalized)
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for i in range(n):
        bi = normalized[i]["bbox_xyxy"]
        ei = _expand(bi, expand_px)
        ci = _center(bi)
        _ = _dims(bi)

        for j in range(i + 1, n):
            bj = normalized[j]["bbox_xyxy"]
            ej = _expand(bj, expand_px)
            cj = _center(bj)
            _ = _dims(bj)

            overlap = _iou(ei, ej)
            if overlap >= iou_thresh:
                union(i, j)
                continue

            if require_overlap_after_expand:
                continue

            dx = abs(ci[0] - cj[0])
            dy = abs(ci[1] - cj[1])
            cdist = math.hypot(dx, dy)
            if cdist > max_center_dist:
                continue

            ix_gap = max(0, max(bi[0], bj[0]) - min(bi[2], bj[2]))
            iy_gap = max(0, max(bi[1], bj[1]) - min(bi[3], bj[3]))
            if ix_gap <= max_gap_x and iy_gap <= max_gap_y:
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    merged = []
    for idxs in clusters.values():
        xs1 = [normalized[k]["bbox_xyxy"][0] for k in idxs]
        ys1 = [normalized[k]["bbox_xyxy"][1] for k in idxs]
        xs2 = [normalized[k]["bbox_xyxy"][2] for k in idxs]
        ys2 = [normalized[k]["bbox_xyxy"][3] for k in idxs]
        merged_box = [min(xs1), min(ys1), max(xs2), max(ys2)]

        members = [normalized[k] for k in idxs]
        members.sort(key=_reading_key)
        text = " ".join([m["text"].strip() for m in members if m["text"].strip()])

        confs = []
        weights = []
        for m in members:
            c = m.get("conf", None)
            if c is None:
                continue
            confs.append(float(c))
            weights.append(_area(m["bbox_xyxy"]))

        if confs:
            weight_sum = sum(weights) if sum(weights) > 0 else len(confs)
            avg_conf = sum(c * w for c, w in zip(confs, weights)) / weight_sum
        else:
            avg_conf = None

        merged.append(
            {
                "bbox_xyxy": merged_box,
                "text": text,
                "conf": avg_conf,
                "merged_from": len(idxs),
            }
        )

    merged.sort(key=_reading_key)
    return merged


def filter_items(items: List[Dict[str, Any]], min_conf: float, min_chars: int, drop_patterns: List[str]) -> List[Dict[str, Any]]:
    import re

    result = []
    for item in items:
        text = (item.get("text") or "").strip()
        conf = item.get("conf")

        if conf is not None and conf < min_conf:
            continue
        if len(text) < min_chars:
            continue
        if drop_patterns and re.search("|".join(drop_patterns), text.lower()):
            continue

        result.append(item)
    return result


def intersection_area(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def remove_mostly_contained_boxes(
    items: List[Dict[str, Any]], containment_thresh: float, prefer_keep: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not items:
        return [], []

    normalized = []
    for item in items:
        box = item.get("bbox_xyxy")
        if box is None:
            continue
        normalized.append({"box": [int(round(v)) for v in box], "item": item})

    n = len(normalized)
    remove_flags = [False] * n
    areas = [_area(normalized[i]["box"]) for i in range(n)]

    for i in range(n):
        if remove_flags[i]:
            continue
        ai = areas[i]
        bi = normalized[i]["box"]

        if ai <= 0:
            remove_flags[i] = True
            continue

        for j in range(n):
            if i == j or remove_flags[i]:
                continue
            aj = areas[j]
            bj = normalized[j]["box"]
            if aj <= 0:
                continue

            inter = intersection_area(bi, bj)
            contained_ratio = inter / ai
            if contained_ratio < containment_thresh:
                continue

            if prefer_keep == "higher_conf":
                ci = normalized[i]["item"].get("conf", None)
                cj = normalized[j]["item"].get("conf", None)
                if ci is not None and cj is not None:
                    if float(ci) < float(cj):
                        remove_flags[i] = True
                    else:
                        remove_flags[j] = True
                else:
                    if ai < aj:
                        remove_flags[i] = True
                    else:
                        remove_flags[j] = True
            else:
                if ai <= aj:
                    remove_flags[i] = True
                else:
                    remove_flags[j] = True

    kept, removed = [], []
    for i in range(n):
        if remove_flags[i]:
            removed.append(normalized[i]["item"])
        else:
            kept.append(normalized[i]["item"])

    return kept, removed


def draw_overlay(image_path: Path, items: List[Dict[str, Any]], out_path: Path, draw_text: bool = False) -> None:
    image = Image.open(image_path).convert("RGB")
    drawer = ImageDraw.Draw(image)

    for item in items:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        drawer.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        if draw_text:
            text = item.get("text", "")
            if text:
                drawer.text((x1, max(0, y1 - 12)), text[:12], fill=(255, 0, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def configure_environment(cfg: Dict[str, Any]) -> None:
    for key, value in cfg.get("env", {}).items():
        os.environ[str(key)] = str(value)


def create_ocr_engine(cfg: Dict[str, Any]):
    from paddleocr import PaddleOCR

    ocr_cfg = cfg["ocr"]
    return PaddleOCR(
        lang=ocr_cfg["lang"],
        use_angle_cls=ocr_cfg["use_angle_cls"],
        det_db_thresh=ocr_cfg["det_db_thresh"],
        det_db_box_thresh=ocr_cfg["det_db_box_thresh"],
        det_db_unclip_ratio=ocr_cfg["det_db_unclip_ratio"],
        det_limit_side_len=ocr_cfg["det_limit_side_len"],
    )


def run_pipeline(cfg: Dict[str, Any], limit: int = 0) -> None:
    paths = cfg["paths"]
    post = cfg["postprocess"]
    output_cfg = cfg["output"]

    input_dir = Path(paths["input_png_dir"])
    out_json_dir = Path(paths["out_json_dir"])
    out_vis_dir = Path(paths["out_vis_dir"])
    stats_json = Path(paths["stats_json"])

    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_vis_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        raise SystemExit(f"No PNG files found in {input_dir}")

    if limit > 0:
        png_files = png_files[:limit]

    print(f"Found PNG files: {len(png_files)}")

    ocr = create_ocr_engine(cfg)
    all_stats = []

    for img_path in png_files:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        raw = ocr.ocr(img_bgr, cls=cfg["ocr"]["use_angle_cls"])
        norm = normalize_paddle_result(raw, img_path)

        items = norm["items"]

        if post["merge"].get("enabled", True):
            m = post["merge"]
            items = merge_close_boxes(
                items=items,
                expand_px=int(m["expand_px"]),
                iou_thresh=float(m["iou_thresh"]),
                max_center_dist=float(m["max_center_dist"]),
                max_gap_x=float(m["max_gap_x"]),
                max_gap_y=float(m["max_gap_y"]),
                require_overlap_after_expand=bool(m["require_overlap_after_expand"]),
            )

        if post["filter"].get("enabled", True):
            f = post["filter"]
            items = filter_items(
                items=items,
                min_conf=float(f["min_conf"]),
                min_chars=int(f["min_chars"]),
                drop_patterns=list(f.get("drop_patterns", [])),
            )

        removed_count = 0
        if post["contained"].get("enabled", True):
            c = post["contained"]
            items, removed = remove_mostly_contained_boxes(
                items=items,
                containment_thresh=float(c["containment_thresh"]),
                prefer_keep=str(c["prefer_keep"]),
            )
            removed_count = len(removed)

        norm["items"] = items
        norm["lang"] = cfg["ocr"]["lang"]

        json_path = out_json_dir / f"{img_path.stem}.json"
        vis_path = out_vis_dir / f"{img_path.stem}{output_cfg['overlay_suffix']}"

        save_json(norm, json_path)
        draw_overlay(img_path, items, vis_path, draw_text=bool(output_cfg["draw_text"]))

        confs = [it["conf"] for it in items if it.get("conf") is not None]
        all_stats.append(
            {
                "page": img_path.stem,
                "num_boxes": len(items),
                "avg_conf": (sum(confs) / len(confs)) if confs else None,
                "removed_contained": removed_count,
                "json": str(json_path),
                "overlay": str(vis_path),
            }
        )

        print(f"{img_path.name}: {len(items)} boxes")

    if output_cfg.get("save_stats", True):
        save_json(all_stats, stats_json)
        print(f"Saved stats: {stats_json}")

    print(
        "Done. JSON files:",
        len(list(out_json_dir.glob("*.json"))),
        "Overlays:",
        len(list(out_vis_dir.glob("*.png"))),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    apply_cli_overrides(cfg, args)

    configure_environment(cfg)
    run_pipeline(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
