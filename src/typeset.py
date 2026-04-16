import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

DEFAULT_ROOT = Path("outputs/jap_ocr_otsu")
DEFAULT_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\times.ttf",
]

EN_CHAR_RE = re.compile(r"[A-Za-z]")
JP_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
FONT_PATH: Optional[str] = None


def resolve_font_path(explicit_font_path: Optional[str]) -> str:
    if explicit_font_path:
        if not os.path.exists(explicit_font_path):
            raise FileNotFoundError(f"Font file not found: {explicit_font_path}")
        return explicit_font_path

    found = next((p for p in DEFAULT_FONT_CANDIDATES if os.path.exists(p)), None)
    if found is None:
        raise FileNotFoundError(
            "No valid font found. Pass --font-path or install one of: "
            + ", ".join(DEFAULT_FONT_CANDIDATES)
        )
    return found


def load_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u3000", " ").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def has_japanese(s: str) -> bool:
    return bool(JP_CHAR_RE.search(s))


def english_ratio(s: str) -> float:
    s = normalize_text(s)
    if not s:
        return 0.0
    alpha_chars = sum(ch.isalpha() for ch in s)
    en_chars = len(EN_CHAR_RE.findall(s))
    return en_chars / max(1, alpha_chars)


def is_english_only_candidate(
    s: str, min_en_ratio: float = 0.5, min_en_letters: int = 2
) -> bool:
    s = normalize_text(s)
    if not s:
        return False
    if has_japanese(s):
        return False
    en_chars = len(EN_CHAR_RE.findall(s))
    return en_chars >= min_en_letters and english_ratio(s) >= min_en_ratio


def validate_schema_lengths(obj: Dict[str, Any], name: str) -> Tuple[int, int, int]:
    polys = obj.get("polys", [])
    texts = obj.get("texts", [])
    scores = obj.get("scores", [])
    if not isinstance(polys, list) or not isinstance(texts, list) or not isinstance(scores, list):
        raise ValueError(f"{name}: polys/texts/scores must be lists")
    return len(polys), len(texts), len(scores)


def poly_to_xyxy(poly: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1 = int(round(min(xs)))
    y1 = int(round(min(ys)))
    x2 = int(round(max(xs)))
    y2 = int(round(max(ys)))
    return x1, y1, x2, y2


def pick_best_text_for_index(
    i: int,
    fb_obj: Dict[str, Any],
    hl_obj: Dict[str, Any],
    nllb_obj: Dict[str, Any],
) -> Dict[str, Any]:
    candidates = []
    for model_name, obj in [("fb", fb_obj), ("hl", hl_obj), ("nllb", nllb_obj)]:
        txt = normalize_text(obj["texts"][i] if i < len(obj["texts"]) else "")
        sc = float(obj["scores"][i]) if i < len(obj["scores"]) else -1.0
        en_ok = is_english_only_candidate(txt)
        candidates.append(
            {
                "model": model_name,
                "text": txt,
                "score": sc,
                "is_english": en_ok,
                "en_ratio": english_ratio(txt),
            }
        )

    english_cands = [c for c in candidates if c["is_english"]]

    if english_cands:
        best = sorted(
            english_cands,
            key=lambda c: (c["score"], c["en_ratio"], len(c["text"])),
            reverse=True,
        )[0]
        best["reason"] = "best_english_highest_score"
        return best

    return {
        "model": None,
        "text": "",
        "score": -1.0,
        "is_english": False,
        "en_ratio": 0.0,
        "reason": "no_english_candidate",
    }


def build_best_page_json(
    base_path: Path, fb_path: Path, hl_path: Path, nllb_path: Path
) -> Dict[str, Any]:
    base_obj = load_json(base_path)
    fb_obj = load_json(fb_path)
    hl_obj = load_json(hl_path)
    nllb_obj = load_json(nllb_path)

    b_lp, b_lt, b_ls = validate_schema_lengths(base_obj, "base")
    f_lp, f_lt, f_ls = validate_schema_lengths(fb_obj, "fb")
    h_lp, h_lt, h_ls = validate_schema_lengths(hl_obj, "hl")
    n_lp, n_lt, n_ls = validate_schema_lengths(nllb_obj, "nllb")

    n = b_lp
    checks = {
        "base_texts": b_lt == n,
        "base_scores": b_ls == n,
        "fb_polys": f_lp == n,
        "fb_texts": f_lt == n,
        "fb_scores": f_ls == n,
        "hl_polys": h_lp == n,
        "hl_texts": h_lt == n,
        "hl_scores": h_ls == n,
        "nllb_polys": n_lp == n,
        "nllb_texts": n_lt == n,
        "nllb_scores": n_ls == n,
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ValueError(f"Length mismatch in {base_path.name}: {bad}")

    best_texts = []
    best_models = []
    best_scores = []
    selection_debug = []

    for i in range(n):
        best = pick_best_text_for_index(i, fb_obj, hl_obj, nllb_obj)
        best_texts.append(best["text"])
        best_models.append(best["model"])
        best_scores.append(best["score"])
        selection_debug.append(best)

    return {
        "input": base_obj.get("input"),
        "lang": "en",
        "polys": base_obj["polys"],
        "texts": best_texts,
        "scores": best_scores,
        "best_model": best_models,
        "best_model_score": best_scores,
        "source_texts": base_obj.get("texts", []),
        "base_ocr_scores": base_obj.get("scores", []),
        "selection_debug": selection_debug,
        "sources": {
            "base_json": str(base_path),
            "fb_json": str(fb_path),
            "hl_json": str(hl_path),
            "nllb_json": str(nllb_path),
        },
    }


def text_bbox(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont
) -> Tuple[int, int]:
    bb = draw.multiline_textbbox((0, 0), text, font=font, spacing=2, align="center")
    return bb[2] - bb[0], bb[3] - bb[1]


def wrap_to_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int
) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    words = text.split(" ")
    if len(words) == 1:
        return text

    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        tw, _ = text_bbox(draw, t, font)
        if tw <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_w: int,
    box_h: int,
    max_font: int = 42,
    min_font: int = 10,
) -> Dict[str, Any]:
    text = normalize_text(text)
    if not text:
        return {"fits": True, "font_size": None, "wrapped": ""}

    for fs in range(max_font, min_font - 1, -1):
        font = ImageFont.truetype(FONT_PATH, fs)
        wrapped = wrap_to_width(draw, text, font, max_w=box_w)
        tw, th = text_bbox(draw, wrapped, font)
        if tw <= box_w and th <= box_h:
            return {
                "fits": True,
                "font_size": fs,
                "wrapped": wrapped,
                "tw": tw,
                "th": th,
            }

    fs = min_font
    font = ImageFont.truetype(FONT_PATH, fs)
    wrapped = wrap_to_width(draw, text, font, max_w=box_w)
    tw, th = text_bbox(draw, wrapped, font)
    return {
        "fits": (tw <= box_w and th <= box_h),
        "font_size": fs,
        "wrapped": wrapped,
        "tw": tw,
        "th": th,
    }

# Previous helper function
def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    wrapped: str,
    font_size: int,
) -> None:
    x1, y1, x2, y2 = box
    font = ImageFont.truetype(FONT_PATH, font_size)
    tw, th = text_bbox(draw, wrapped, font)
    bx, by = x2 - x1, y2 - y1
    x = x1 + (bx - tw) // 2
    y = y1 + (by - th) // 2
    draw.multiline_text((x, y), wrapped, font=font, fill=(0, 0, 0), spacing=2, align="center")


def boxes_overlap(
    box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int], pad: int = 5
) -> bool:
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2
    return not (x2a + pad < x1b or x2b + pad < x1a or y2a + pad < y1b or y2b + pad < y1a)

def draw_centered_with_bg(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    wrapped: str,
    font_size: int,
    bg_pad: int = 1,
) -> None:
    x1, y1, x2, y2 = box
    font = ImageFont.truetype(FONT_PATH, font_size)
    tw, th = text_bbox(draw, wrapped, font)

    bx, by = x2 - x1, y2 - y1
    x = x1 + (bx - tw) // 2
    y = y1 + (by - th) // 2

    # Only whiten the text footprint (+small padding), clipped to box.
    rx1 = max(x1, x - bg_pad)
    ry1 = max(y1, y - bg_pad)
    rx2 = min(x2, x + tw + bg_pad)
    ry2 = min(y2, y + th + bg_pad)

    draw.rectangle([rx1, ry1, rx2, ry2], fill=(229, 229, 229))
    draw.multiline_text((x, y), wrapped, font=font, fill=(0, 0, 0), spacing=2, align="center")


def render_best_page(
    page_png: Path,
    best_json_path: Path,
    pad: int = 2,
    max_font: int = 42,
    min_font: int = 10,
) -> Dict[str, Any]:
    page = Image.open(page_png).convert("RGB")
    draw = ImageDraw.Draw(page)
    obj = load_json(best_json_path)

    polys = obj["polys"]
    texts = obj["texts"]
    scores = obj.get("scores", [-1.0] * len(polys))
    best_models = obj.get("best_model", [None] * len(polys))

    render_list = []
    for i, poly in enumerate(polys):
        en = normalize_text(texts[i] if i < len(texts) else "")
        if not en:
            continue

        x1, y1, x2, y2 = poly_to_xyxy(poly)
        box_area = (x2 - x1) * (y2 - y1)
        score = float(scores[i]) if i < len(scores) else -1.0
        model = best_models[i] if i < len(best_models) else None

        render_list.append(
            {
                "idx": i,
                "text": en,
                "poly": poly,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "area": box_area,
                "score": score,
                "best_model": model,
                "priority": (score, box_area),
            }
        )

    render_list.sort(key=lambda r: r["priority"], reverse=True)

    rendered_items = []
    debug = []
    merged_count = 0

    for item in render_list:
        i = item["idx"]
        en = item["text"]
        x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]

        x1p, y1p, x2p, y2p = x1 + pad, y1 + pad, x2 - pad, y2 - pad
        x1p, y1p = max(0, x1p), max(0, y1p)
        x2p, y2p = min(page.width, x2p), min(page.height, y2p)

        bw, bh = x2p - x1p, y2p - y1p
        if bw < 6 or bh < 6:
            continue

        current_box = (x1p, y1p, x2p, y2p)
        current_text = en
        current_indices = [i]

        overlapping_idx = None
        for idx, rendered in enumerate(rendered_items):
            if boxes_overlap(current_box, rendered["box"], pad=3):
                overlapping_idx = idx
                break

        if overlapping_idx is not None:
            rendered_item = rendered_items[overlapping_idx]

            merge_x1 = min(rendered_item["box"][0], current_box[0])
            merge_y1 = min(rendered_item["box"][1], current_box[1])
            merge_x2 = max(rendered_item["box"][2], current_box[2])
            merge_y2 = max(rendered_item["box"][3], current_box[3])
            merged_box = (merge_x1, merge_y1, merge_x2, merge_y2)

            merged_text = rendered_item["text"] + "\n" + current_text

            rendered_items[overlapping_idx] = {
                "box": merged_box,
                "text": merged_text,
                "indices": rendered_item["indices"] + current_indices,
                "merged": True,
            }
            merged_count += 1

            debug.append(
                {
                    "idx": i,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "text": en,
                    "status": "merged_with",
                    "merged_with_idx": rendered_item["indices"][0],
                    "best_model": item["best_model"],
                    "best_model_score": item["score"],
                }
            )
        else:
            rendered_items.append(
                {
                    "box": current_box,
                    "text": current_text,
                    "indices": current_indices,
                    "merged": False,
                }
            )

            debug.append(
                {
                    "idx": i,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "text": en,
                    "status": "added",
                    "best_model": item["best_model"],
                    "best_model_score": item["score"],
                }
            )

    for rendered in rendered_items:
        x1p, y1p, x2p, y2p = rendered["box"]
        text = rendered["text"]

        bw, bh = x2p - x1p, y2p - y1p
        if bw < 6 or bh < 6:
            continue

        # draw.rectangle([x1p, y1p, x2p, y2p], fill=(255, 255, 255))

        fit = fit_text(draw, text, bw, bh, max_font=max_font, min_font=min_font)
        if fit["wrapped"] and fit["font_size"] is not None:
            draw_centered_with_bg(
                draw,
                (x1p, y1p, x2p, y2p),
                fit["wrapped"],
                fit["font_size"],
                bg_pad=2,
            )

    print(f"  Merged {merged_count} overlapping boxes")
    return {"image": page, "debug": debug}


def compute_model_counts(all_debug: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    model_counts = {"fb": 0, "hl": 0, "nllb": 0, "none": 0}
    for items in all_debug.values():
        for it in items:
            m = it.get("best_model")
            if m in model_counts:
                model_counts[m] += 1
            else:
                model_counts["none"] += 1
    return model_counts


def run_pipeline(args: argparse.Namespace) -> int:
    global FONT_PATH
    FONT_PATH = resolve_font_path(args.font_path)
    print("Using font:", FONT_PATH)

    root = Path(args.root)
    base_json_dir = root / args.base_json_dir
    fb_dir = root / args.fb_dir
    hl_dir = root / args.hl_dir
    nllb_dir = root / args.nllb_dir
    og_dir = root / args.og_dir

    out_best_json_dir = root / args.out_best_json_dir
    out_render_dir = root / args.out_render_dir
    out_debug_dir = root / args.out_debug_dir

    out_best_json_dir.mkdir(parents=True, exist_ok=True)
    out_render_dir.mkdir(parents=True, exist_ok=True)
    out_debug_dir.mkdir(parents=True, exist_ok=True)

    base_files = sorted(base_json_dir.glob("*.json"))
    print("Base pages:", len(base_files))

    run_rows = []
    all_debug: Dict[str, List[Dict[str, Any]]] = {}

    for base_path in base_files:
        name = base_path.name
        stem = base_path.stem

        fb_path = fb_dir / name
        hl_path = hl_dir / name
        nllb_path = nllb_dir / name
        page_png = og_dir / f"{stem}.png"

        if not (fb_path.exists() and hl_path.exists() and nllb_path.exists()):
            run_rows.append({"page": stem, "status": "skip_missing_translation_file"})
            continue
        if not page_png.exists():
            run_rows.append({"page": stem, "status": "skip_missing_png"})
            continue

        try:
            best_obj = build_best_page_json(base_path, fb_path, hl_path, nllb_path)
            out_best_json = out_best_json_dir / name
            save_json(best_obj, out_best_json)

            rendered = render_best_page(
                page_png,
                out_best_json,
                pad=args.pad,
                max_font=args.max_font,
                min_font=args.min_font,
            )
            out_img = out_render_dir / f"{stem}_best.png"
            rendered["image"].save(out_img)

            all_debug[stem] = rendered["debug"]

            n_boxes = len(best_obj["polys"])
            n_filled = sum(1 for t in best_obj["texts"] if normalize_text(t))
            run_rows.append(
                {
                    "page": stem,
                    "status": "ok",
                    "num_boxes": n_boxes,
                    "num_selected_english": n_filled,
                    "selected_ratio": n_filled / max(1, n_boxes),
                }
            )

        except Exception as e:
            run_rows.append({"page": stem, "status": f"error: {e}"})

    save_json(all_debug, out_debug_dir / args.debug_json_name)

    df_run = pd.DataFrame(run_rows)
    if not df_run.empty and "page" in df_run.columns:
        df_run = df_run.sort_values("page")
    df_run.to_csv(out_debug_dir / args.summary_csv_name, index=False)
    print(df_run)

    model_counts = compute_model_counts(all_debug)
    print("Model counts:", model_counts)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select best JP->EN translation per OCR box and render typeset pages."
    )
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help="Root output directory")
    parser.add_argument("--base-json-dir", type=str, default="json", help="Base OCR JSON subdir under root")
    parser.add_argument("--fb-dir", type=str, default="translation_fb", help="Facebook translation subdir under root")
    parser.add_argument("--hl-dir", type=str, default="translation_hl", help="Helsinki translation subdir under root")
    parser.add_argument("--nllb-dir", type=str, default="translation_nllb", help="NLLB translation subdir under root")
    parser.add_argument("--og-dir", type=str, default="resized_original", help="Original page image subdir under root")
    parser.add_argument("--out-best-json-dir", type=str, default="translation_best", help="Output selected translation JSON subdir under root")
    parser.add_argument("--out-render-dir", type=str, default="translation_best_render", help="Output rendered image subdir under root")
    parser.add_argument("--out-debug-dir", type=str, default="translation_best_debug", help="Output debug artifacts subdir under root")
    parser.add_argument("--debug-json-name", type=str, default="render_debug_best.json", help="Debug JSON filename")
    parser.add_argument("--summary-csv-name", type=str, default="run_summary_best.csv", help="Run summary CSV filename")
    parser.add_argument("--font-path", type=str, default=None, help="Optional explicit .ttf path")
    parser.add_argument("--pad", type=int, default=1, help="Padding applied to each box before rendering")
    parser.add_argument("--max-font", type=int, default=42, help="Maximum font size for fitting")
    parser.add_argument("--min-font", type=int, default=10, help="Minimum font size for fitting")

    args = parser.parse_args()
    if args.min_font < 1:
        parser.error("--min-font must be >= 1")
    if args.max_font < args.min_font:
        parser.error("--max-font must be >= --min-font")
    if args.pad < 0:
        parser.error("--pad must be >= 0")
    return args


def main() -> int:
    args = parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
