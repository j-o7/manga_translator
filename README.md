# COMET Manga Translation Pipeline

End-to-end OCR + translation + re-typesetting pipeline for manga/comic pages.

This repository implements a full workflow:
1. Preprocess Japanese and English pages
2. Run OCR (PaddleOCR PP-OCRv5)
3. Translate Japanese OCR text to English with multiple MT models
4. Select best translation per text box and render back into pages
5. Evaluate predictions against English ground truth JSON

The project also includes many exploratory notebooks under base_jupyter for OCR, translation, and evaluation experiments.

## Table of Contents
- Project Overview
- Repository Layout
- Pipeline Stages
- Data and File Formats
- Environment Setup
- Quick Start
- Script-by-Script Usage
- Evaluation Outputs
- Notebooks
- Troubleshooting
- Known Limitations

## Project Overview

### Main ideas
- OCR is run separately for Japanese source pages and English ground-truth pages.
- Japanese OCR boxes are merged/reordered using custom logic in src/utils/merging_ocr_jap.py.
- English OCR boxes are merged using src/utils/merging_ocr_en.py.
- Three translation models are run on the same OCR JSON:
  - facebook/m2m100_418M
  - Helsinki-NLP/opus-mt-ja-en
  - facebook/nllb-200-distilled-600M
- For each box, typesetting chooses the best candidate using an English-text heuristic and score.
- Rendered pages are evaluated with geometric + text metrics.

### Primary entry point
- run_all.bat executes the full 5-stage pipeline with configurable paths.

## Repository Layout

- run_all.bat: Batch orchestrator for preprocess -> OCR -> translate -> typeset -> eval
- requirements.txt: Python environment snapshot
- NotoSansCJKjp-VF.ttf: Japanese-capable font used in OCR visualizations
- src/
  - preprocess.py: Image preprocessing techniques and resized original export
  - pp_ocrv5_jap.py: Japanese OCR + translation-ready Japanese merge pipeline
  - pp_ocrv5_en.py: English OCR + vertical merge and filtering
  - translate_jp_to_en.py: Multi-model JA->EN translation
  - typeset.py: Best-model selection and image rendering into bubbles
  - eval.py: Directory-level GT vs prediction evaluation to CSV
  - utils/
    - merging_ocr_jap.py: Region grouping, iterative merging, reading order, furigana-aware handling
    - merging_ocr_en.py: Center/gap merge + English filtering
- base_jupyter/: Experimental notebooks for OCR/translation/evaluation
- data/: Input datasets and preprocessed variants
- outputs/: Generated images, JSON, rendered pages, debug, and CSV metrics
- docs/: Working notes and planning docs

## Pipeline Stages

### Stage 1: Preprocess
Script: src/preprocess.py

- Reads images from input directory
- Applies one or all techniques:
  - gray
  - clahe
  - otsu
  - otsu_inv
  - adaptive
  - adaptive_inv
  - bg_norm
  - mild_binary
- Saves resized originals to out_dir/resized_original
- Saves technique outputs to out_dir/<technique>

### Stage 2: OCR
Scripts:
- src/pp_ocrv5_jap.py
- src/pp_ocrv5_en.py

Japanese OCR flow:
- PP-OCRv5 with lang=japan (GPU by default)
- Builds translation-ready blocks via build_translation_ready_japanese_ocr_v4
- Saves:
  - out_dir/base (base OCR visualization)
  - out_dir/vis (merged OCR visualization)
  - out_dir/og (OCR preprocessor output image)
  - out_dir/json (merged OCR JSON)

English OCR flow:
- PP-OCRv5 with lang=en (GPU by default)
- Rotational correction from OCR preprocessor angle
- Box merging and filtering for English text
- Saves same directory structure as Japanese OCR

### Stage 3: Translation
Script: src/translate_jp_to_en.py

- Input: directory of Japanese OCR JSON files (typically outputs/jap/json)
- Runs three JA->EN models and writes parallel outputs:
  - translation_fb
  - translation_hl
  - translation_nllb
- Normalizes text and removes repetitive model artifacts
- Output schema includes polys/texts/scores plus source_texts

### Stage 4: Typeset
Script: src/typeset.py

- Inputs:
  - base OCR JSON
  - three translation JSON directories
  - original resized page images
- For each box:
  - picks best candidate using English-only heuristic + score
  - fits wrapped text by shrinking font as needed
  - merges overlapping render boxes
  - paints compact background patch + centered text
- Outputs:
  - selected best JSON per page (translation_best by default)
  - rendered page images (translation_best_render)
  - debug JSON + run summary CSV (translation_best_debug)

### Stage 5: Evaluate
Script: src/eval.py

- Matches GT and prediction JSON files by filename
- Per-GT-box matching logic:
  - best IoU first
  - center-distance fallback if IoU below threshold
- Computes text and geometry metrics:
  - IoU, intersection_over_gt
  - normalized Levenshtein similarity
  - CER-like similarity
  - word-level similarity
  - chrF
  - end-to-end weighted metrics (IoU * text scores)
- Writes:
  - per_box_metrics.csv
  - per_page_metrics.csv
  - missing_files_report.json

## Data and File Formats

### Expected OCR/translation JSON shape
Core scripts expect a page-level JSON with parallel arrays:

{
  "polys": [
    [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    ...
  ],
  "texts": ["text1", "text2", ...],
  "scores": [0.98, 0.87, ...]
}

Additional keys are allowed (for example input/lang/source_texts).

### Ground truth for evaluation
- GT and prediction directories must contain JSON files with matching names.
- Each matched pair should contain polys/texts arrays with equal lengths inside each file.

## Environment Setup

## 1) Python
Recommended: Python 3.10+ with CUDA-compatible setup if using GPU OCR/translation.

### Windows (venv)
1. Create environment
   - python -m venv .venv
2. Activate
   - .venv\Scripts\activate
3. Install dependencies
   - pip install -r requirements.txt

If requirements.txt is too heavy for your machine, install a minimal set first:
- numpy
- pandas
- pillow
- opencv-python
- torch
- transformers
- paddleocr
- paddlepaddle-gpu (or paddlepaddle CPU)
- shapely
- rapidfuzz
- sacrebleu

## 2) Fonts
- Root file NotoSansCJKjp-VF.ttf is used by OCR visualization scripts.
- typeset.py defaults to Windows fonts:
  - C:\Windows\Fonts\arial.ttf
  - C:\Windows\Fonts\calibri.ttf
  - C:\Windows\Fonts\times.ttf
- You can override with --font-path in typeset.py.

## 3) Model downloads
First translation run will download Hugging Face models:
- facebook/m2m100_418M
- Helsinki-NLP/opus-mt-ja-en
- facebook/nllb-200-distilled-600M

First OCR run will download PaddleOCR model assets if not cached.

## Quick Start

### Option A: Full pipeline (recommended)
Edit variables at the top of run_all.bat, then run:

run_all.bat

Default flow in run_all.bat:
- preprocess Japanese and English with adaptive thresholding
- OCR both sets
- translate Japanese OCR JSON
- select best translation + render pages
- evaluate against English OCR JSON

### Option B: Run each stage manually

1) Preprocess
- python src/preprocess.py --in_dir data/onepiece/jap --out_dir outputs/jap --technique adaptive
- python src/preprocess.py --in_dir data/onepiece/en --out_dir outputs/en --technique adaptive

2) OCR
- python src/pp_ocrv5_jap.py --in_dir outputs/jap/adaptive --out_dir outputs/jap
- python src/pp_ocrv5_en.py --in_dir outputs/en/adaptive --out_dir outputs/en

3) Translate
- python src/translate_jp_to_en.py --in_dir outputs/jap/json

4) Typeset
- python src/typeset.py --root outputs/jap --base-json-dir json --fb-dir translation_fb --hl-dir translation_hl --nllb-dir translation_nllb --og-dir resized_original --out-best-json-dir translation_best --out-render-dir translation_best_render --out-debug-dir translation_best_debug --pad 1 --max-font 42 --min-font 10

5) Evaluate
- python src/eval.py --gt_dir outputs/en/json --pred_dir outputs/jap/translation_best --out_dir outputs/csv_metrics --iou_threshold 0.1

## Script-by-Script Usage

### src/preprocess.py
Main flags:
- --in_dir
- --out_dir
- --recursive
- --technique {all, gray, clahe, otsu, otsu_inv, adaptive, adaptive_inv, bg_norm, mild_binary}
- --scale
- --max_side
- --denoise_h

### src/pp_ocrv5_jap.py
Main flags:
- --in_dir (PNG input directory)
- --out_dir (root output directory)

Notes:
- Uses PaddleOCR device="gpu" in code.
- If no GPU is available, change device in script.

### src/pp_ocrv5_en.py
Main flags:
- --in_dir (PNG input directory)
- --out_dir (root output directory)

Notes:
- Uses PaddleOCR device="gpu" in code.
- Merges text boxes with merge_english_boxes_center_gap.

### src/translate_jp_to_en.py
Main flags:
- --in_dir (directory containing OCR JSON files)

Behavior:
- Creates sibling directories relative to input parent:
  - translation_fb
  - translation_hl
  - translation_nllb

### src/typeset.py
Main flags:
- --root
- --base-json-dir
- --fb-dir
- --hl-dir
- --nllb-dir
- --og-dir
- --out-best-json-dir
- --out-render-dir
- --out-debug-dir
- --debug-json-name
- --summary-csv-name
- --font-path
- --pad
- --max-font
- --min-font

### src/eval.py
Main flags:
- --gt_dir
- --pred_dir
- --out_dir
- --iou_threshold
- --one_to_one

## Evaluation Outputs

Output directory (for example outputs/csv_metrics) includes:
- per_box_metrics.csv
  - one row per GT box with matched prediction details
- per_page_metrics.csv
  - page-level aggregates including weighted text scores and recalls
- missing_files_report.json
  - filenames missing from GT or prediction side

Useful columns in per_page_metrics.csv:
- mean_iou
- mean_lev_sim
- mean_chrf
- mean_end2end_lev
- mean_end2end_chrf
- recall_iou_03
- recall_iou_05
- weighted_lev_sim
- weighted_chrf
- page_concat_lev_sim
- page_concat_chrf

## Notebooks

The base_jupyter folder contains development notebooks for:
- OCR experiments (ocr_new.ipynb, ocr_tests.ipynb, pp_ocrv5*.ipynb)
- Translation experiments (translate_test.ipynb, translate_eval.ipynb, translate_panel.ipynb)
- End-to-end and typesetting tests (jp_to_en_typeset.ipynb)
- Evaluation experiments (ocr_eval.ipynb)

These notebooks are useful for ablations, debugging, and visual validation. Production-like runs should use scripts in src and run_all.bat.

## Troubleshooting

### PaddleOCR fails to initialize
- Verify paddleocr and paddlepaddle(-gpu) installation compatibility.
- If GPU issues occur, switch OCR scripts to CPU by changing device="gpu" to device="cpu".

### Missing fonts in rendering
- Pass --font-path to typeset.py with a valid .ttf file.
- Ensure NotoSansCJKjp-VF.ttf is present for OCR visualization text drawing.

### Translation model download errors
- Check internet access and Hugging Face availability.
- Re-run after clearing partial cache if needed.

### Evaluation says no matching files
- Ensure GT and prediction directories have identical JSON filenames.

### Unexpected empty translation selections
- Inspect translation_best_debug/render_debug_best.json and run_summary_best.csv to see why boxes were skipped or merged.

## Known Limitations

- OCR scripts are currently hardcoded to GPU mode.
- Translation stage loads all three large models in one process, which can be memory-heavy.
- Best-translation selection is heuristic (English-likeness + score), not semantic quality scoring.
- Typesetting is axis-aligned box based; curved balloons and stylized layouts may still fail.
- requirements.txt is a broad environment snapshot and may include unrelated packages.

## Suggested Next Improvements

- Add a minimal requirements file specifically for src scripts.
- Expose OCR device selection as CLI flags.
- Add optional batching and half-precision for translation speed/memory.
- Add semantic reranking (for example COMET/BERTScore) when selecting best translation.
- Add unit tests for JSON schema validation and matching logic.
