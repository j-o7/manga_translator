# COMET Manga Translation Pipeline

COMET is an end-to-end pipeline for manga translation and re-typesetting. It takes Japanese manga pages, detects and merges text regions, translates them into English, renders the translated text back into the page, and evaluates the result against English reference pages.

The pipeline is designed as a modular baseline. Each stage saves intermediate outputs, which makes it easier to debug OCR errors, translation failures, and rendering issues.

## Pipeline Overview

1. **Preprocess images**  
   `src/preprocess.py`  
   Applies image preprocessing methods such as adaptive thresholding and saves resized originals for later rendering.

2. **Run OCR on Japanese pages**  
   `src/pp_ocrv5_jap.py`  
   Uses PaddleOCR PP-OCRv5 and a custom merging pipeline to create translation-ready Japanese text regions.

3. **Run OCR on English pages**  
   `src/pp_ocrv5_en.py`  
   Uses PaddleOCR PP-OCRv5 to create English reference OCR JSON files for evaluation.

4. **Translate Japanese OCR text**  
   `src/translate_jp_to_en.py`  
   Runs three JA→EN models:
   - `facebook/m2m100_418M`
   - `Helsinki-NLP/opus-mt-ja-en`
   - `facebook/nllb-200-distilled-600M`

5. **Select best translation and render**  
   `src/typeset.py`  
   Chooses the best English candidate per text region, fits the text into the region, and renders it back into the page.

6. **Evaluate predictions**  
   `src/eval.py`  
   Compares predicted JSON files against English reference JSON files using layout and text metrics.

## Main Features

- Separate OCR pipelines for Japanese source pages and English reference pages
- Custom Japanese OCR merging with reading-order-aware cleanup and furigana handling
- English OCR merging and filtering for cleaner reference regions
- Multi-model translation with per-region candidate selection
- Box-based text rendering with font fitting and overlap handling
- Evaluation with IoU, Levenshtein similarity, chrF, and end-to-end weighted metrics

## Repository Layout

- `run_all.bat` — full pipeline runner
- `src/preprocess.py` — preprocessing
- `src/pp_ocrv5_jap.py` — Japanese OCR
- `src/pp_ocrv5_en.py` — English OCR
- `src/translate_jp_to_en.py` — translation
- `src/typeset.py` — best-model selection and rendering
- `src/eval.py` — evaluation
- `src/utils/merging_ocr_jap.py` — Japanese box merging
- `src/utils/merging_ocr_en.py` — English box merging

## Quick Start

### Full pipeline
Edit paths in `run_all.bat`, then run:

```bash
run_all.bat