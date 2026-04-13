import re
import json
import numpy as np


def poly_to_xyxy(poly):
    pts = np.array(poly, dtype=np.float32)
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    return [x1, y1, x2, y2]


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_height(box):
    return box[3] - box[1]


def box_width(box):
    return box[2] - box[0]


def horizontal_gap(box1, box2):
    # box2 assumed to be on the right
    return box2[0] - box1[2]


def merge_boxes(box1, box2):
    return [
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3]),
    ]


def make_poly_from_box(box):
    x1, y1, x2, y2 = box
    return [
        [int(round(x1)), int(round(y1))],
        [int(round(x2)), int(round(y1))],
        [int(round(x2)), int(round(y2))],
        [int(round(x1)), int(round(y2))],
    ]


def should_merge_by_gap_and_center(box1, box2,
                                   max_xcenter_diff_ratio=0.45,
                                   max_ygap_ratio=1.0,
                                   max_abs_gap=45):
    """
    Swapped x/y version of the previous function.

    Merge if:
    1. x centers are close enough
    2. vertical gap is small enough
    """

    c1x, c1y = box_center(box1)
    c2x, c2y = box_center(box2)

    w1 = box_width(box1)
    w2 = box_width(box2)
    avg_w = (w1 + w2) / 2.0

    # alignment using x centers
    xcenter_diff = abs(c1x - c2x)
    if xcenter_diff > avg_w * max_xcenter_diff_ratio:
        return False

    # vertical gap
    gap = box2[1] - box1[3]

    if gap <= 0:
        return True

    allowed_gap = min(max_abs_gap, avg_w * max_ygap_ratio)
    return gap <= allowed_gap


def poly_to_xyxy(poly):
    pts = np.array(poly, dtype=np.float32)
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    return [x1, y1, x2, y2]


def is_english_text(text):
    """
    Keep only strings made of English letters, digits, spaces,
    and common punctuation.
    """
    t = text.strip()
    if not t:
        return False

    return re.fullmatch(r"[A-Za-z0-9\s.,!?;:'\"()\-\[\]/&]+", t) is not None


def filter_bad_english_boxes(
    polys,
    texts,
    scores=None,
    min_w=12,
    min_h=12,
    min_area=180,
):
    """
    Remove a box if:
    1. it is too small
    2. text length is 1
    3. text is not English-compatible
    """

    kept_polys = []
    kept_texts = []
    kept_scores = [] if scores is not None else None

    removed = []

    for i, (poly, text) in enumerate(zip(polys, texts)):
        t = text.strip()
        x1, y1, x2, y2 = poly_to_xyxy(poly)
        w = x2 - x1
        h = y2 - y1
        area = w * h

        remove_reason = None

        if w < min_w or h < min_h or area < min_area:
            remove_reason = "too_small"
        elif len(t) == 1:
            remove_reason = "single_char"
        elif not is_english_text(t):
            remove_reason = "non_english"

        if remove_reason is not None:
            removed.append({
                "index": i,
                "text": text,
                "box": [x1, y1, x2, y2],
                "reason": remove_reason
            })
            continue

        kept_polys.append(poly)
        kept_texts.append(text)
        if scores is not None:
            kept_scores.append(scores[i])

    if scores is not None:
        return kept_polys, kept_texts, kept_scores
    return kept_polys, kept_texts


def merge_english_boxes_center_gap(polys, texts, scores=None,
                                   max_xcenter_diff_ratio=0.45,
                                   max_ygap_ratio=1.0,
                                   max_abs_gap=45):
    items = []
    for i, (poly, text) in enumerate(zip(polys, texts)):
        box = poly_to_xyxy(poly)
        items.append({
            "poly": poly,
            "box": box,
            "text": text,
            "score": scores[i] if scores is not None else None
        })

    # top-to-bottom, then left-to-right
    items.sort(key=lambda d: (d["box"][1], box_center(d["box"])[0]))

    merged = []
    used = [False] * len(items)

    for i in range(len(items)):
        if used[i]:
            continue

        curr_box = items[i]["box"][:]
        curr_texts = [items[i]["text"]]
        curr_scores = []
        if items[i]["score"] is not None:
            curr_scores.append(items[i]["score"])

        used[i] = True

        while True:
            best_j = None
            best_gap = None

            for j in range(len(items)):
                if used[j]:
                    continue

                next_box = items[j]["box"]

                # only consider boxes to the right, or slightly overlapping
                gap = next_box[1] - curr_box[3]
                if gap < -20:
                    continue

                if should_merge_by_gap_and_center(
                    curr_box,
                    next_box,
                    max_xcenter_diff_ratio=max_xcenter_diff_ratio,
                    max_ygap_ratio=max_ygap_ratio,
                    max_abs_gap=max_abs_gap
                ):
                    if best_j is None or gap < best_gap:
                        best_j = j
                        best_gap = gap

            if best_j is None:
                break

            curr_box = merge_boxes(curr_box, items[best_j]["box"])
            curr_texts.append(items[best_j]["text"])

            if items[best_j]["score"] is not None:
                curr_scores.append(items[best_j]["score"])

            used[best_j] = True

        merged.append({
            "poly": make_poly_from_box(curr_box),
            "text": " ".join(curr_texts),
            "score": float(np.mean(curr_scores)) if curr_scores else None
        })

    filtered_polys, filtered_texts, filtered_scores = filter_bad_english_boxes(
        [m["poly"] for m in merged],
        [m["text"] for m in merged],
        [m["score"] for m in merged] if merged and merged[0]["score"] is not None else None
    )

    filtered_merged = []
    for i, (poly, text, score) in enumerate(zip(filtered_polys, filtered_texts, filtered_scores if filtered_scores is not None else [None] * len(filtered_polys))):
        filtered_merged.append({
            "poly": poly,
            "text": text,
            "score": score
        })


    return filtered_merged