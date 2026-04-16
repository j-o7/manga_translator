#!/usr/bin/env python3
"""
Evaluate translated OCR JSONs against ground-truth JSONs for a whole directory.

What this script does
---------------------
For each matching JSON filename in:
    gt_dir/
    pred_dir/

it:
1. Loads polygons + texts
2. Matches each GT box to the best predicted box
   - first by highest IoU
   - if all IoUs are 0, falls back to nearest center distance
3. Computes per-box metrics
4. Writes:
   - per_box_metrics.csv
   - per_page_metrics.csv

Expected JSON format
--------------------
{
  "polys": [
    [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
    ...
  ],
  "texts": ["text1", "text2", ...]
}

Optional installs
-----------------
pip install pandas numpy shapely rapidfuzz sacrebleu
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from rapidfuzz.distance import Levenshtein
from sacrebleu.metrics import CHRF


# ----------------------------
# text helpers
# ----------------------------

def normalize_text(s: str) -> str:
    """Light normalization for fair comparison."""
    if s is None:
        return ""
    s = str(s).strip().lower()

    # normalize quotes/dashes a bit
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-")

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalized_levenshtein_similarity(a: str, b: str) -> float:
    """1 - normalized edit distance."""
    a = normalize_text(a)
    b = normalize_text(b)
    if not a and not b:
        return 1.0
    dist = Levenshtein.distance(a, b)
    return 1.0 - dist / max(len(a), len(b), 1)


def cer_like_similarity(a: str, b: str) -> float:
    """
    Character-level similarity.
    Same formula as normalized Levenshtein, but kept separately
    in case you later want different preprocessing.
    """
    return normalized_levenshtein_similarity(a, b)


def word_level_similarity(a: str, b: str) -> float:
    """
    Simple word-level similarity using tokenized Levenshtein-like idea.
    Not exact WER, but useful as a compact comparable score.
    """
    a_tokens = normalize_text(a).split()
    b_tokens = normalize_text(b).split()

    if not a_tokens and not b_tokens:
        return 1.0

    # join with separator so token identities stay distinct
    a_str = "\u241f".join(a_tokens)
    b_str = "\u241f".join(b_tokens)
    dist = Levenshtein.distance(a_str, b_str)
    return 1.0 - dist / max(len(a_str), len(b_str), 1)


# ----------------------------
# geometry helpers
# ----------------------------

def safe_polygon(points: List[List[float]]) -> Polygon:
    """
    Build a valid polygon from point list.
    buffer(0) often fixes minor self-intersection issues.
    """
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def polygon_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    """Intersection over Union."""
    if poly_a.is_empty or poly_b.is_empty:
        return 0.0
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    return inter / union if union > 0 else 0.0


def intersection_over_gt(poly_gt: Polygon, poly_pred: Polygon) -> float:
    """How much of GT is covered by pred."""
    if poly_gt.is_empty or poly_pred.is_empty:
        return 0.0
    inter = poly_gt.intersection(poly_pred).area
    gt_area = poly_gt.area
    return inter / gt_area if gt_area > 0 else 0.0


def polygon_center(points: List[List[float]]) -> np.ndarray:
    arr = np.array(points, dtype=float)
    return arr.mean(axis=0)


def center_distance(points_a: List[List[float]], points_b: List[List[float]]) -> float:
    ca = polygon_center(points_a)
    cb = polygon_center(points_b)
    return float(np.linalg.norm(ca - cb))


def bbox_from_points(points: List[List[float]]) -> Tuple[float, float, float, float]:
    arr = np.array(points, dtype=float)
    x1 = float(arr[:, 0].min())
    y1 = float(arr[:, 1].min())
    x2 = float(arr[:, 0].max())
    y2 = float(arr[:, 1].max())
    return x1, y1, x2, y2


# ----------------------------
# json loading
# ----------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "polys" not in data or "texts" not in data:
        raise ValueError(f"{path} missing 'polys' or 'texts' keys")

    polys = data["polys"]
    texts = data["texts"]

    if len(polys) != len(texts):
        raise ValueError(f"{path} has len(polys) != len(texts)")

    return data


# ----------------------------
# matching
# ----------------------------

def match_gt_to_pred(
    gt_polys_raw: List[List[List[float]]],
    gt_texts: List[str],
    pred_polys_raw: List[List[List[float]]],
    pred_texts: List[str],
    iou_threshold: float = 0.0,
    enforce_one_to_one: bool = False,
) -> List[Dict[str, Any]]:
    """
    Match each GT box to one predicted box.

    Strategy:
    - Compute IoU against all predicted boxes
    - Pick highest IoU
    - If all IoUs are 0, fall back to nearest center distance
    - Optionally enforce one-to-one usage of predicted boxes
    """
    gt_polys = [safe_polygon(p) for p in gt_polys_raw]
    pred_polys = [safe_polygon(p) for p in pred_polys_raw]

    used_pred = set()
    matches = []

    for gt_idx, (gt_poly, gt_pts, gt_text) in enumerate(zip(gt_polys, gt_polys_raw, gt_texts)):
        best_pred_idx = None
        best_iou = -1.0

        # first try best IoU
        for pred_idx, pred_poly in enumerate(pred_polys):
            if enforce_one_to_one and pred_idx in used_pred:
                continue
            iou = polygon_iou(gt_poly, pred_poly)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx

        match_reason = "iou"

        # fallback to nearest center if no overlap
        if best_pred_idx is None or best_iou <= iou_threshold:
            best_pred_idx = None
            best_dist = math.inf

            for pred_idx, pred_pts in enumerate(pred_polys_raw):
                if enforce_one_to_one and pred_idx in used_pred:
                    continue
                dist = center_distance(gt_pts, pred_pts)
                if dist < best_dist:
                    best_dist = dist
                    best_pred_idx = pred_idx

            match_reason = "center_fallback"

        if best_pred_idx is not None and enforce_one_to_one:
            used_pred.add(best_pred_idx)

        pred_pts = pred_polys_raw[best_pred_idx] if best_pred_idx is not None else None
        pred_text = pred_texts[best_pred_idx] if best_pred_idx is not None else ""

        if best_pred_idx is not None:
            pred_poly = pred_polys[best_pred_idx]
            final_iou = polygon_iou(gt_poly, pred_poly)
            final_iogt = intersection_over_gt(gt_poly, pred_poly)
            final_center_dist = center_distance(gt_pts, pred_pts)
        else:
            final_iou = 0.0
            final_iogt = 0.0
            final_center_dist = None

        matches.append({
            "gt_idx": gt_idx,
            "pred_idx": best_pred_idx,
            "match_reason": match_reason,
            "iou": final_iou,
            "intersection_over_gt": final_iogt,
            "center_distance": final_center_dist,
            "gt_text": gt_text,
            "pred_text": pred_text,
            "gt_points": gt_pts,
            "pred_points": pred_pts,
        })

    return matches


# ----------------------------
# scoring
# ----------------------------

def score_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chrf_metric = CHRF()
    scored = []

    for m in matches:
        gt_text = m["gt_text"]
        pred_text = m["pred_text"]

        lev_sim = normalized_levenshtein_similarity(gt_text, pred_text)
        cer_sim = cer_like_similarity(gt_text, pred_text)
        word_sim = word_level_similarity(gt_text, pred_text)

        if pred_text.strip():
            chrf_score = chrf_metric.sentence_score(
                normalize_text(pred_text),
                [normalize_text(gt_text)]
            ).score / 100.0
        else:
            chrf_score = 0.0

        end2end_lev = m["iou"] * lev_sim
        end2end_chrf = m["iou"] * chrf_score

        gt_x1, gt_y1, gt_x2, gt_y2 = bbox_from_points(m["gt_points"])
        if m["pred_points"] is not None:
            pr_x1, pr_y1, pr_x2, pr_y2 = bbox_from_points(m["pred_points"])
        else:
            pr_x1 = pr_y1 = pr_x2 = pr_y2 = None

        row = {
            **m,
            "gt_text_norm": normalize_text(gt_text),
            "pred_text_norm": normalize_text(pred_text),
            "gt_len_chars": len(normalize_text(gt_text)),
            "pred_len_chars": len(normalize_text(pred_text)),
            "lev_sim": lev_sim,
            "cer_sim": cer_sim,
            "word_sim": word_sim,
            "chrf": chrf_score,
            "end2end_lev": end2end_lev,
            "end2end_chrf": end2end_chrf,
            "gt_x1": gt_x1,
            "gt_y1": gt_y1,
            "gt_x2": gt_x2,
            "gt_y2": gt_y2,
            "pred_x1": pr_x1,
            "pred_y1": pr_y1,
            "pred_x2": pr_x2,
            "pred_y2": pr_y2,
        }
        scored.append(row)

    return scored


def summarize_page(rows: List[Dict[str, Any]], filename: str) -> Dict[str, Any]:
    if not rows:
        return {
            "file": filename,
            "num_gt_boxes": 0,
            "num_matched_boxes": 0,
            "mean_iou": 0.0,
            "mean_iogt": 0.0,
            "mean_center_distance": None,
            "mean_lev_sim": 0.0,
            "mean_cer_sim": 0.0,
            "mean_word_sim": 0.0,
            "mean_chrf": 0.0,
            "mean_end2end_lev": 0.0,
            "mean_end2end_chrf": 0.0,
            "recall_iou_03": 0.0,
            "recall_iou_05": 0.0,
        }

    df = pd.DataFrame(rows)

    num_gt = len(df)
    num_matched = df["pred_idx"].notna().sum()

    summary = {
        "file": filename,
        "num_gt_boxes": int(num_gt),
        "num_matched_boxes": int(num_matched),
        "mean_iou": float(df["iou"].mean()),
        "mean_iogt": float(df["intersection_over_gt"].mean()),
        "mean_center_distance": float(df["center_distance"].dropna().mean()) if df["center_distance"].notna().any() else None,
        "mean_lev_sim": float(df["lev_sim"].mean()),
        "mean_cer_sim": float(df["cer_sim"].mean()),
        "mean_word_sim": float(df["word_sim"].mean()),
        "mean_chrf": float(df["chrf"].mean()),
        "mean_end2end_lev": float(df["end2end_lev"].mean()),
        "mean_end2end_chrf": float(df["end2end_chrf"].mean()),
        "recall_iou_03": float((df["iou"] >= 0.3).mean()),
        "recall_iou_05": float((df["iou"] >= 0.5).mean()),
    }

    # length-weighted text metrics
    weights = df["gt_len_chars"].clip(lower=1).to_numpy(dtype=float)
    weights = weights / weights.sum()

    summary["weighted_lev_sim"] = float(np.sum(df["lev_sim"].to_numpy() * weights))
    summary["weighted_chrf"] = float(np.sum(df["chrf"].to_numpy() * weights))
    summary["weighted_end2end_lev"] = float(np.sum(df["end2end_lev"].to_numpy() * weights))
    summary["weighted_end2end_chrf"] = float(np.sum(df["end2end_chrf"].to_numpy() * weights))

    # concatenated page text score in GT order
    gt_concat = " ".join(df.sort_values("gt_idx")["gt_text_norm"].tolist())
    pred_concat = " ".join(df.sort_values("gt_idx")["pred_text_norm"].tolist())
    summary["page_concat_lev_sim"] = normalized_levenshtein_similarity(gt_concat, pred_concat)

    chrf_metric = CHRF()
    summary["page_concat_chrf"] = (
        chrf_metric.sentence_score(pred_concat, [gt_concat]).score / 100.0
        if pred_concat.strip() else 0.0
    )

    return summary


# ----------------------------
# directory evaluation
# ----------------------------

def evaluate_directories(
    gt_dir: Path,
    pred_dir: Path,
    out_dir: Path,
    iou_threshold: float = 0.0,
    enforce_one_to_one: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_files = {p.name: p for p in gt_dir.glob("*.json")}
    pred_files = {p.name: p for p in pred_dir.glob("*.json")}

    common = sorted(set(gt_files.keys()) & set(pred_files.keys()))
    only_gt = sorted(set(gt_files.keys()) - set(pred_files.keys()))
    only_pred = sorted(set(pred_files.keys()) - set(gt_files.keys()))

    if not common:
        raise ValueError("No matching JSON filenames found between gt_dir and pred_dir")

    per_box_rows = []
    per_page_rows = []

    for filename in common:
        gt_path = gt_files[filename]
        pred_path = pred_files[filename]

        gt_data = load_json(gt_path)
        pred_data = load_json(pred_path)

        matches = match_gt_to_pred(
            gt_polys_raw=gt_data["polys"],
            gt_texts=gt_data["texts"],
            pred_polys_raw=pred_data["polys"],
            pred_texts=pred_data["texts"],
            iou_threshold=iou_threshold,
            enforce_one_to_one=enforce_one_to_one,
        )

        scored_rows = score_matches(matches)

        for row in scored_rows:
            row["file"] = filename
            row["gt_json_path"] = str(gt_path)
            row["pred_json_path"] = str(pred_path)
            per_box_rows.append(row)

        page_summary = summarize_page(scored_rows, filename)
        page_summary["gt_json_path"] = str(gt_path)
        page_summary["pred_json_path"] = str(pred_path)
        per_page_rows.append(page_summary)

        print(f"[done] {filename} | GT={len(gt_data['texts'])} Pred={len(pred_data['texts'])}")

    per_box_df = pd.DataFrame(per_box_rows)
    per_page_df = pd.DataFrame(per_page_rows)

    per_box_csv = out_dir / "per_box_metrics.csv"
    per_page_csv = out_dir / "per_page_metrics.csv"
    missing_report = out_dir / "missing_files_report.json"

    per_box_df.to_csv(per_box_csv, index=False, encoding="utf-8")
    per_page_df.to_csv(per_page_csv, index=False, encoding="utf-8")

    with open(missing_report, "w", encoding="utf-8") as f:
        json.dump(
            {
                "matched_files": common,
                "only_in_gt": only_gt,
                "only_in_pred": only_pred,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSaved:")
    print(f"  {per_box_csv}")
    print(f"  {per_page_csv}")
    print(f"  {missing_report}")

    # dataset-level summary
    print("\nDataset summary:")
    print(per_page_df[[
        "mean_iou",
        "mean_lev_sim",
        "mean_chrf",
        "mean_end2end_lev",
        "mean_end2end_chrf",
        "recall_iou_03",
        "recall_iou_05",
        "page_concat_lev_sim",
        "page_concat_chrf",
    ]].mean(numeric_only=True).to_string())


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory with ground-truth JSONs")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory with translated/predicted JSONs")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save CSV results")
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.0,
        help="If best IoU <= this threshold, fallback to nearest center distance"
    )
    parser.add_argument(
        "--one_to_one",
        action="store_true",
        help="Enforce one predicted box to be used at most once"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_directories(
        gt_dir=Path(args.gt_dir),
        pred_dir=Path(args.pred_dir),
        out_dir=Path(args.out_dir),
        iou_threshold=args.iou_threshold,
        enforce_one_to_one=args.one_to_one,
    )


if __name__ == "__main__":
    main()