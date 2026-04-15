from argparse import ArgumentParser
import glob
import os, json, math, re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from PIL import Image, ImageDraw, ImageFont

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _poly_to_xyxy(poly: Any) -> List[int]:
    # poly can be [[x,y], ...] (quad or polygon)
    if not isinstance(poly, list) or len(poly) == 0:
        return [0, 0, 0, 0]
    xs, ys = [], []
    for pt in poly:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            xs.append(float(pt[0]))
            ys.append(float(pt[1]))
    if not xs or not ys:
        return [0, 0, 0, 0]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

def _from_parallel_arrays(polys: List[Any], texts: List[Any], scores: List[Any]) -> List[Dict[str, Any]]:
    n = min(len(polys), len(texts), len(scores) if scores is not None else len(texts))
    items = []
    for i in range(n):
        poly = polys[i]
        txt = "" if texts[i] is None else str(texts[i])
        conf = None
        if scores is not None and i < len(scores):
            try:
                conf = float(scores[i]) if scores[i] is not None else None
            except Exception:
                conf = None

        items.append({
            "id": i,
            "poly": poly,
            "bbox_xyxy": _poly_to_xyxy(poly),
            "text": txt,
            "ja_text": txt,   # keeps your get_ja_text logic working
            "conf": conf
        })
    return items

def get_items(page_obj: Any) -> List[Dict[str, Any]]:
    # 1) already a list of item dicts
    if isinstance(page_obj, list):
        return page_obj

    # 2) dict-based schemas
    if isinstance(page_obj, dict):
        # old schema: {"items":[...]} or similar
        for k in ["items", "boxes", "bubbles", "ocr"]:
            if k in page_obj and isinstance(page_obj[k], list):
                return page_obj[k]

        # new compact schema: {"polys": [...], "texts": [...], "scores": [...]}
        if "polys" in page_obj and "texts" in page_obj:
            polys = list(page_obj.get("polys", []))
            texts = list(page_obj.get("texts", []))
            scores = list(page_obj.get("scores", [])) if "scores" in page_obj else None
            return _from_parallel_arrays(polys, texts, scores)

        # raw Paddle-style schema if present: {"dt_polys","rec_texts","rec_scores"}
        if "dt_polys" in page_obj and "rec_texts" in page_obj:
            polys = list(page_obj.get("dt_polys", []))
            texts = list(page_obj.get("rec_texts", []))
            scores = list(page_obj.get("rec_scores", [])) if "rec_scores" in page_obj else None
            return _from_parallel_arrays(polys, texts, scores)

    raise ValueError("Unrecognized OCR JSON schema")

def normalize_text(text: Optional[str]) -> str:
    # ✅ None-safe
    if text is None:
        return ""
    text = str(text)

    # basic cleanup (tweak as you like)
    text = text.replace("\u3000", " ")          # full-width space -> space
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_ja_text(it: Dict[str, Any]) -> str:
    for k in ["ja_text", "text", "ocr_text", "jp_text"]:
        if k in it:
            return normalize_text(it[k])
    return ""

def get_conf(it: Dict[str, Any]) -> Optional[float]:
    for k in ["conf", "avg_conf", "confidence", "ocr_conf"]:
        if k in it:
            try:
                return float(it[k])
            except:
                return None
    return None

def items_to_ocr_json(src_obj: Any, items: List[Dict[str, Any]], text_key: str = "en_text") -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # Preserve useful metadata from original OCR JSON when available
    if isinstance(src_obj, dict):
        for k in ["input", "lang", "image", "width", "height", "engine", "page_index"]:
            if k in src_obj:
                out[k] = src_obj[k]

    out["polys"] = []
    out["texts"] = []
    out["scores"] = []

    for it in items:
        poly = it.get("poly")
        if not poly:
            # fallback from bbox if poly missing
            x1, y1, x2, y2 = it.get("bbox_xyxy", [0, 0, 0, 0])
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        out["polys"].append(poly)

        txt = it.get(text_key)
        if txt is None:
            txt = it.get("text", "")
        out["texts"].append(normalize_text(txt))

        conf = it.get("conf")
        try:
            out["scores"].append(float(conf) if conf is not None else None)
        except Exception:
            out["scores"].append(None)

    # Optional: keep source JP text for debugging/eval
    out["source_texts"] = [normalize_text(it.get("ja_text", it.get("text", ""))) for it in items]

    return out

def translate_fb(model, tokenizer, items):
    def translate_ja_to_en_fb(text: Optional[str], max_in_len: int = 256) -> str:
        text = normalize_text(text)
        if not text:
            return ""
        if tokenizer is None or model is None: # translator_mode == "placeholder" or 
            return f"[EN] {text}"

        with torch.no_grad():
            batch = tokenizer(
                text,
                return_tensors="pt"
            ).to(device)

            gen = model.generate(**batch, forced_bos_token_id=tokenizer.get_lang_id("en"))
            out = tokenizer.decode(gen, skip_special_tokens=True)[0]

        return normalize_text(out)    

    new_items = []
    for it in items:
        ja = get_ja_text(it)
        en = translate_ja_to_en_fb(ja) if ja else ""
        new_it = dict(it)
        new_it["ja_text"] = ja
        new_it["en_text"] = en
        c = get_conf(it)
        if c is not None:
            new_it["conf"] = c
        new_items.append(new_it)
    return new_items

def translate_hl(model, tokenizer,items):
    def translate_ja_to_en_hl(text: Optional[str], max_in_len: int = 256) -> str:
        text = normalize_text(text)
        if not text:
            return ""
        if tokenizer is None or model is None: # translator_mode == "placeholder" or 
            return f"[EN] {text}"

        with torch.no_grad():
            batch = tokenizer(
                [text],
                return_tensors="pt"
            ).to(device)

            gen = model.generate(**batch)
            out = tokenizer.decode(gen, skip_special_tokens=True)[0]

        return normalize_text(out)
    

    # placeholder: just copy ja_text to en_text
    new_items = []
    for it in items:
        ja = get_ja_text(it)
        en = translate_ja_to_en_hl(ja) if ja else ""
        new_it = dict(it)
        new_it["ja_text"] = ja
        new_it["en_text"] = en
        c = get_conf(it)
        if c is not None:
            new_it["conf"] = c
        new_items.append(new_it)
    return new_items

def translate_nllb(model, tokenizer, items):
    def translate_ja_to_en_nllb(text: Optional[str], max_in_len: int = 256) -> str:
        text = normalize_text(text)
        if not text:
            return ""
        if tokenizer is None or model is None: # translator_mode == "placeholder" or 
            return f"[EN] {text}"

        with torch.no_grad():
            batch = tokenizer(
                text,
                return_tensors="pt"
            ).to(device)

            gen = model.generate(
                **batch,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_length=256
            )
            out = tokenizer.decode(gen, skip_special_tokens=True)[0]

        return normalize_text(out)

    # placeholder: just copy ja_text to en_text
    new_items = []
    for it in items:
        ja = get_ja_text(it)
        en = translate_ja_to_en_nllb(ja) if ja else ""
        new_it = dict(it)
        new_it["ja_text"] = ja
        new_it["en_text"] = en
        c = get_conf(it)
        if c is not None:
            new_it["conf"] = c
        new_items.append(new_it)
    return new_items

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--in_dir", type=str, default="outputs/ocr_json")
    args = args.parse_args()

    parent_dir = Path(args.in_dir).parent
    fb_out_dir = parent_dir / "translation_fb"
    hl_out_dir = parent_dir / "translation_hl"
    nllb_out_dir = parent_dir / "translation_nllb"
    os.makedirs(fb_out_dir, exist_ok=True)
    os.makedirs(hl_out_dir, exist_ok=True)
    os.makedirs(nllb_out_dir, exist_ok=True)
    
    json_files = glob.glob(str(Path(args.in_dir) / "*.json"))

    fb_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    fb_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    fb_tokenizer.src_lang = "ja"
    fb_model = fb_model.to(device)

    hl_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
    hl_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
    hl_model = hl_model.to(device)

    nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    nllb_model = nllb_model.to(device)
    
    for jf in json_files:
        base_name = os.path.basename(jf).replace(".json", "")
        print(f"Translating: {base_name}")
        data = load_json(Path(jf))
        items = get_items(data)
        print(f"Translating using facebook/m2m100_418M: {base_name}")
        fb_it = translate_fb(fb_model, fb_tokenizer, items)
        print(f"Translating using Helsinki-NLP/opus-mt-ja-en: {base_name}")
        hl_it = translate_hl(hl_model, hl_tokenizer, items)
        print(f"Translating using facebook/nllb-200-distilled-600M: {base_name}")
        nllb_it = translate_nllb(nllb_model, nllb_tokenizer, items)

        fb_json = items_to_ocr_json(data, fb_it, text_key="en_text")
        hl_json = items_to_ocr_json(data, hl_it, text_key="en_text")
        nllb_json = items_to_ocr_json(data, nllb_it, text_key="en_text")
        with open(fb_out_dir / f"{base_name}.json", "w", encoding="utf-8") as f:
            json.dump(fb_json, f, ensure_ascii=False, indent=2)
        with open(hl_out_dir / f"{base_name}.json", "w", encoding="utf-8") as f:
            json.dump(hl_json, f, ensure_ascii=False, indent=2)
        with open(nllb_out_dir / f"{base_name}.json", "w", encoding="utf-8") as f:
            json.dump(nllb_json, f, ensure_ascii=False, indent=2)
        print(f"Translated and saved: {base_name}")