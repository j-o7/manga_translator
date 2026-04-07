#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate OCR JSON (JA->EN) and render translated text back onto page images."
    )

    parser.add_argument("--ocr-json-dir", default="outputs/ocr_json_jap", help="Input OCR JSON directory")
    parser.add_argument("--pages-dir", default="data/jap_imgs", help="Input page PNG directory")
    parser.add_argument("--out-tjson-dir", default="outputs/translated_json_jap", help="Output translated JSON directory")
    parser.add_argument("--out-render-dir", default="outputs/final_pages_bbox_jap", help="Output rendered image directory")
    parser.add_argument("--debug-json", default="outputs/debug/render_debug_bbox_jap.json", help="Render debug JSON path")
    parser.add_argument("--metrics-csv", default="outputs/metrics_bbox_jap.csv", help="Metrics CSV path")

    parser.add_argument("--font-path", default=None, help="Path to a TTF font")
    parser.add_argument(
        "--font-candidates",
        nargs="+",
        default=[
            r"C:\\Windows\\Fonts\\arial.ttf",
            r"C:\\Windows\\Fonts\\calibri.ttf",
            r"C:\\Windows\\Fonts\\times.ttf",
        ],
        help="Candidate TTF paths used when --font-path is not set",
    )

    parser.add_argument("--pad", type=int, default=2, help="Inward padding for text boxes")
    parser.add_argument("--max-font", type=int, default=42, help="Maximum font size")
    parser.add_argument("--min-font", type=int, default=10, help="Minimum font size")

    parser.add_argument("--use-marian", action="store_true", help="Use MarianMT model for translation")
    parser.add_argument("--model-name", default="susiexyf/marian-finetuned-ja-en", help="Hugging Face model name")
    parser.add_argument("--max-in-len", type=int, default=256, help="Maximum tokenizer input length")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens")

    parser.add_argument("--limit", type=int, default=0, help="Process at most N pages (0 means all)")
    parser.add_argument("--skip-render", action="store_true", help="Only write translated JSON, skip rendering")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip metrics CSV generation")

    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    out = str(text)
    out = out.replace("\u3000", " ")
    out = out.replace("\n", " ").replace("\r", " ")
    out = re.sub(r"\s+", " ", out).strip()
    return out


def get_items(page_obj: Any) -> List[Dict[str, Any]]:
    if isinstance(page_obj, list):
        return page_obj
    if isinstance(page_obj, dict):
        for key in ["items", "boxes", "bubbles", "ocr"]:
            if key in page_obj and isinstance(page_obj[key], list):
                return page_obj[key]
    raise ValueError("Unrecognized OCR JSON schema")


def get_bbox_xyxy(item: Dict[str, Any]) -> Tuple[int, int, int, int]:
    box = item.get("bbox_xyxy", item.get("bbox", None))
    if box is None:
        raise KeyError("Missing bbox_xyxy/bbox")
    x1, y1, x2, y2 = box
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def get_ja_text(item: Dict[str, Any]) -> str:
    for key in ["ja_text", "text", "ocr_text", "jp_text", "text_ja"]:
        if key in item:
            return normalize_text(item[key])
    return ""


def get_conf(item: Dict[str, Any]) -> Optional[float]:
    for key in ["conf", "avg_conf", "confidence", "ocr_conf"]:
        if key in item:
            try:
                return float(item[key])
            except Exception:
                return None
    return None


def resolve_font_path(font_path: Optional[str], font_candidates: List[str]) -> str:
    if font_path:
        if os.path.exists(font_path):
            return font_path
        raise FileNotFoundError(f"Provided --font-path does not exist: {font_path}")

    for candidate in font_candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("No valid font found. Pass --font-path to a valid .ttf file.")


def init_translator(use_marian: bool, model_name: str):
    mode = "placeholder"
    tokenizer = None
    model = None
    device = None

    if not use_marian:
        print("Using placeholder translator (prefixes with [EN]).")
        return mode, tokenizer, model, device

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for translation: {device}")
        model = model.to(device).eval()
        mode = "marian"
        print(f"MarianMT loaded: {model_name} | device: {device}")

    except Exception as exc:
        mode = "placeholder"
        tokenizer, model, device = None, None, None
        print(f"MarianMT unavailable, falling back to placeholder: {exc!r}")

    return mode, tokenizer, model, device


def translate_ja_to_en(
    text: Optional[str],
    translator_mode: str,
    tokenizer,
    model,
    device,
    max_in_len: int,
    max_new_tokens: int,
) -> str:
    text = normalize_text(text)
    if not text:
        return ""

    if translator_mode == "placeholder" or tokenizer is None or model is None:
        return f"[EN] {text}"

    import torch

    with torch.no_grad():
        batch = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_in_len,
        ).to(device)
        generated = model.generate(**batch, max_new_tokens=max_new_tokens)
        out = tokenizer.decode(generated[0], skip_special_tokens=True)

    return normalize_text(out)


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    bb = draw.multiline_textbbox((0, 0), text, font=font, spacing=2, align="center")
    return bb[2] - bb[0], bb[3] - bb[1]


def wrap_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> str:
    text = normalize_text(text)
    if not text:
        return ""

    words = text.split(" ")
    if len(words) == 1:
        return text

    lines: List[str] = []
    cur = ""
    for word in words:
        test = (cur + " " + word).strip()
        tw, _ = text_bbox(draw, test, font)
        if tw <= max_w or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)

    return "\n".join(lines)


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_w: int,
    box_h: int,
    font_path: str,
    max_font: int,
    min_font: int,
) -> Dict[str, Any]:
    text = normalize_text(text)
    if not text:
        return {"fits": True, "font_size": None, "wrapped": ""}

    for fs in range(max_font, min_font - 1, -1):
        font = ImageFont.truetype(font_path, fs)
        wrapped = wrap_to_width(draw, text, font, max_w=box_w)
        tw, th = text_bbox(draw, wrapped, font)
        if tw <= box_w and th <= box_h:
            return {"fits": True, "font_size": fs, "wrapped": wrapped, "tw": tw, "th": th}

    fs = min_font
    font = ImageFont.truetype(font_path, fs)
    wrapped = wrap_to_width(draw, text, font, max_w=box_w)
    tw, th = text_bbox(draw, wrapped, font)
    return {"fits": (tw <= box_w and th <= box_h), "font_size": fs, "wrapped": wrapped, "tw": tw, "th": th}


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    wrapped: str,
    font_size: int,
    font_path: str,
) -> None:
    x1, y1, x2, y2 = box
    font = ImageFont.truetype(font_path, font_size)
    tw, th = text_bbox(draw, wrapped, font)
    bw, bh = x2 - x1, y2 - y1
    x = x1 + (bw - tw) // 2
    y = y1 + (bh - th) // 2
    draw.multiline_text((x, y), wrapped, font=font, fill=(0, 0, 0), spacing=2, align="center")


def translate_page_json(
    ocr_json_path: Path,
    translator_mode: str,
    tokenizer,
    model,
    device,
    max_in_len: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    obj = load_json(ocr_json_path)
    items = get_items(obj)

    out_items = []
    for item in items:
        ja = get_ja_text(item)
        en = translate_ja_to_en(
            ja,
            translator_mode=translator_mode,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_in_len=max_in_len,
            max_new_tokens=max_new_tokens,
        ) if ja else ""

        new_item = dict(item)
        new_item["ja_text"] = ja
        new_item["en_text"] = en

        conf = get_conf(item)
        if conf is not None:
            new_item["conf"] = conf

        out_items.append(new_item)

    return {
        "source_ocr_json": str(ocr_json_path),
        "translator": translator_mode,
        "items": out_items,
    }


def render_page_from_translated_json(
    page_png: Path,
    translated_json: Path,
    font_path: str,
    pad: int,
    max_font: int,
    min_font: int,
) -> Dict[str, Any]:
    page = Image.open(page_png).convert("RGB")
    draw = ImageDraw.Draw(page)

    obj = load_json(translated_json)
    items = obj["items"] if isinstance(obj, dict) and "items" in obj else get_items(obj)

    debug = []
    for item in items:
        x1, y1, x2, y2 = get_bbox_xyxy(item)

        x1p, y1p, x2p, y2p = x1 + pad, y1 + pad, x2 - pad, y2 - pad
        x1p, y1p = max(0, x1p), max(0, y1p)
        x2p, y2p = min(page.width, x2p), min(page.height, y2p)

        bw, bh = x2p - x1p, y2p - y1p
        if bw < 6 or bh < 6:
            continue

        en = normalize_text(item.get("en_text", ""))
        if not en:
            continue

        draw.rectangle([x1p, y1p, x2p, y2p], fill=(255, 255, 255))

        fit = fit_text(
            draw=draw,
            text=en,
            box_w=bw,
            box_h=bh,
            font_path=font_path,
            max_font=max_font,
            min_font=min_font,
        )
        if fit["wrapped"] and fit["font_size"] is not None:
            draw_centered(draw, (x1p, y1p, x2p, y2p), fit["wrapped"], fit["font_size"], font_path=font_path)

        debug.append(
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "pad_bbox_xyxy": [x1p, y1p, x2p, y2p],
                "fits": fit.get("fits"),
                "font_size": fit.get("font_size"),
                "en_len": len(en),
                "ja_len": len(item.get("ja_text", "")),
                "conf": item.get("conf", None),
            }
        )

    return {"image": page, "debug": debug}


def save_metrics_csv(debug_json_path: Path, metrics_csv_path: Path) -> None:
    import pandas as pd

    dbg = load_json(debug_json_path)
    rows = []
    for page, items in dbg.items():
        n = len(items)
        if n == 0:
            continue

        overflowed = sum(1 for it in items if it.get("fits") is False)
        small_font = sum(1 for it in items if (it.get("font_size") is not None and it["font_size"] < 12))
        confs = [it["conf"] for it in items if it.get("conf") is not None]
        font_sizes = [it["font_size"] for it in items if it.get("font_size") is not None]

        rows.append(
            {
                "page": page,
                "num_boxes": n,
                "avg_conf": (sum(confs) / len(confs)) if confs else None,
                "overflowed": overflowed,
                "small_font_lt12": small_font,
                "avg_font_size": (sum(font_sizes) / max(1, len(font_sizes))) if font_sizes else None,
            }
        )

    if not rows:
        print("No metrics rows to write.")
        return

    df = pd.DataFrame(rows).sort_values("page")
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics CSV: {metrics_csv_path}")


def main() -> None:
    args = parse_args()

    ocr_json_dir = Path(args.ocr_json_dir)
    pages_dir = Path(args.pages_dir)
    out_tjson_dir = Path(args.out_tjson_dir)
    out_render_dir = Path(args.out_render_dir)
    debug_json_path = Path(args.debug_json)
    metrics_csv_path = Path(args.metrics_csv)

    out_tjson_dir.mkdir(parents=True, exist_ok=True)
    out_render_dir.mkdir(parents=True, exist_ok=True)

    font_path = resolve_font_path(args.font_path, args.font_candidates)
    print(f"Using font: {font_path}")

    translator_mode, tokenizer, model, device = init_translator(args.use_marian, args.model_name)

    ocr_files = sorted(ocr_json_dir.glob("*.json"))
    if not ocr_files:
        raise SystemExit(f"No OCR JSON files found in: {ocr_json_dir}")

    if args.limit > 0:
        ocr_files = ocr_files[: args.limit]

    print(f"OCR pages: {len(ocr_files)}")

    all_debug: Dict[str, List[Dict[str, Any]]] = {}
    translated_count = 0
    rendered_count = 0

    for ocr_path in ocr_files:
        stem = ocr_path.stem
        page_png = pages_dir / f"{stem}.png"

        if not page_png.exists():
            print(f"Missing page image for: {stem}")
            continue

        translated_obj = translate_page_json(
            ocr_json_path=ocr_path,
            translator_mode=translator_mode,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_in_len=args.max_in_len,
            max_new_tokens=args.max_new_tokens,
        )

        tjson_path = out_tjson_dir / ocr_path.name
        save_json(translated_obj, tjson_path)
        translated_count += 1

        if args.skip_render:
            continue

        out = render_page_from_translated_json(
            page_png=page_png,
            translated_json=tjson_path,
            font_path=font_path,
            pad=args.pad,
            max_font=args.max_font,
            min_font=args.min_font,
        )

        out_img_path = out_render_dir / f"{stem}_translated.png"
        out["image"].save(out_img_path)
        all_debug[stem] = out["debug"]
        rendered_count += 1

    if not args.skip_render:
        save_json(all_debug, debug_json_path)
        print(f"Saved debug JSON: {debug_json_path}")

    if not args.skip_render and not args.skip_metrics:
        save_metrics_csv(debug_json_path, metrics_csv_path)

    print(f"Done. Translated JSON pages: {translated_count}")
    if not args.skip_render:
        print(f"Rendered pages: {rendered_count}")


if __name__ == "__main__":
    main()
