from argparse import ArgumentParser

import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import os
from paddleocr import PaddleOCR

os.environ["FLAGS_use_mkldnn"] = "0"     # disable oneDNN (MKLDNN)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # sometimes helps on Windows envs
os.environ["FLAGS_use_new_executor"] = "0"

from utils.merging_ocr_en import merge_english_boxes_center_gap

# helper function to display the OCR results on the image
def visualize_ocr_results(res_dict: dict) -> Image.Image:
    vis = np.array(res_dict['output_img'], dtype=np.uint8)
    vis = np.ascontiguousarray(vis)

    polys = res_dict['polys']
    for it in polys:
        pts = np.array(it, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

    img_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("NotoSansCJKjp-VF.ttf", 20)

    for i, poly in enumerate(res_dict['polys']):
        pts = np.array(poly, dtype=np.int32)
        text = res_dict['texts'][i]
        x = int(np.min(pts[:, 0]))
        y = max(int(np.min(pts[:, 1])) - 24, 0)

        draw.rectangle([x, y, x + 8 * max(len(text), 1), y + 24], fill=(0, 255, 0))
        draw.text((x + 2, y), f"{i}: {text}", font=font, fill=(0, 0, 0))

    # img_rgb = cv2.cvtColor(img_pil, cv2.COLOR_BGR2RGB)
    return img_pil

def save_results(res_dicts, out_path):
    def numpy_converter(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    json_dir = f"{out_path}/json"
    vis_dir = f"{out_path}/vis"
    base_dir = f"{out_path}/og"
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)

    for res_dict in res_dicts:
        base_name = os.path.basename(res_dict["input"]).replace(".png", "")
        print("Saving results for:", base_name)
        json_path = os.path.join(json_dir, f"{base_name}.json")
        vis_path = os.path.join(vis_dir, f"{base_name}.png")
        base_path = os.path.join(base_dir, f"{base_name}.png")

        vis_img = visualize_ocr_results(res_dict)
        vis_img.save(vis_path)

        base_img = Image.fromarray(cv2.cvtColor(res_dict['output_img'], cv2.COLOR_BGR2RGB))
        base_img.save(base_path)

        del res_dict['output_img']

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(res_dict, f, default=numpy_converter, ensure_ascii=False, indent=2)
        
        print(f"Saved info for: {base_name}")

def save_base(res_dicts, out_path):
    base_dir = f"{out_path}/base"
    os.makedirs(base_dir, exist_ok=True)

    for res_dict in res_dicts:
        base_name = os.path.basename(res_dict["input"]).replace(".png", "")
        print("Saving base image for:", base_name)
        base_path = os.path.join(base_dir, f"{base_name}.png")

        base_img = visualize_ocr_results(res_dict)
        base_img.save(base_path)

def undo_right_angle_rotation(img, polys, angle_deg):
    """
    angle_deg = angle used during preprocessing
    This undoes it for image + OCR polys
    Supports: 90, -90, 180, 270, -270
    """
    h, w = img.shape[:2]
    a = angle_deg % 360

    if a == 270:
        # preprocessing rotated +90, so undo with -90
        out_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        new_polys = []
        for poly in polys:
            new_poly = []
            for x, y in poly:
                nx = y
                ny = w - 1 - x
                new_poly.append([int(nx), int(ny)])
            new_polys.append(new_poly)

    elif a == 90:
        # preprocessing rotated -90, so undo with +90
        out_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        new_polys = []
        for poly in polys:
            new_poly = []
            for x, y in poly:
                nx = h - 1 - y
                ny = x
                new_poly.append([int(nx), int(ny)])
            new_polys.append(new_poly)

    elif a == 180:
        out_img = cv2.rotate(img, cv2.ROTATE_180)

        new_polys = []
        for poly in polys:
            new_poly = []
            for x, y in poly:
                nx = w - 1 - x
                ny = h - 1 - y
                new_poly.append([int(nx), int(ny)])
            new_polys.append(new_poly)

    elif a == 0:
        out_img = img.copy()
        new_polys = polys

    else:
        raise ValueError("Use this only for 90/180/270 rotations.")

    return out_img, new_polys

def reprocessed_main_dict(res_out):
    assert len(res_out['dt_polys']) == len(res_out['rec_texts']) == len(res_out['rec_scores']), "Length mismatch in polys, texts, scores"

    angle= res_out['doc_preprocessor_res']['angle']
    img = res_out['doc_preprocessor_res']['output_img']
    old_polys = res_out['dt_polys']

    fixed_img, fixed_polys = undo_right_angle_rotation(img, old_polys, angle)

    return_dict = {
        'input': res_out['input_path'],
        'output_img': fixed_img,
        'lang': 'en',
        'polys': fixed_polys,
        'texts': res_out['rec_texts'],
        'scores': res_out['rec_scores']
    }
    
    return return_dict

def get_merged_dict(res_dict):
    processed_dict = reprocessed_main_dict(res_dict)

    fixed_dict = processed_dict.copy()
    # display(visualize_ocr_results(fixed_dict))
    merged = merge_english_boxes_center_gap(
        fixed_dict['polys'], 
        fixed_dict['texts'], 
        fixed_dict['scores'],
        max_xcenter_diff_ratio=0.1,
        max_ygap_ratio=0.3,
        max_abs_gap=50
        )
    fixed_dict['polys'] = [m['poly'] for m in merged]
    fixed_dict['texts'] = [m['text'] for m in merged]
    fixed_dict['scores'] = [m['score'] for m in merged]
    # display(visualize_ocr_results(fixed_dict))
    return fixed_dict, processed_dict


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--in_dir", type=str, default="data/jap_imgs")
    args.add_argument("--out_dir", type=str, default="outputs/")
    args = args.parse_args()

    INPUT_PNG_DIR = Path(args.in_dir)

    if not os.path.exists(INPUT_PNG_DIR):
        raise ValueError("Input directory does not exist.")

    OUT_DIR = Path(args.out_dir)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)

    png_files = sorted(INPUT_PNG_DIR.glob("*.png"))
    print("Found PNGs:", len(png_files))
    
    ocr = PaddleOCR(
        lang='en',
        ocr_version="PP-OCRv5",
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        device = "gpu")

    ocr_dicts = []
    og_dicts = []
    for png_path in png_files:
        print("Processing:", png_path.name)
        res_ocr = ocr.predict(str(png_path))[0]
        out_dict, og_dict = get_merged_dict(res_ocr)
        ocr_dicts.append(out_dict)
        og_dicts.append(og_dict)

    save_base(og_dicts, OUT_DIR)
    save_results(ocr_dicts, OUT_DIR)