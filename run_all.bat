@echo off
REM Full pipeline: preprocess -> OCR -> translate -> typeset -> eval
REM All directories are configurable via variables at the top of this file

setlocal enabledelayedexpansion

REM ========================================
REM CONFIGURATION - Customize these paths
REM ========================================

REM Input directories
set JAP_INPUT_DIR=./data/onepiece/jap
set EN_INPUT_DIR=./data/onepiece/en

REM Preprocessing output directories
set JAP_PREPROCESS_OUT=outputs/jap
set EN_PREPROCESS_OUT=outputs/en
set PREPROCESS_TECHNIQUE=adaptive

REM OCR output directories
set JAP_OCR_OUT=outputs/jap
set EN_OCR_OUT=outputs/en
set JAP_OCR_INPUT=%JAP_PREPROCESS_OUT%/%PREPROCESS_TECHNIQUE%
set EN_OCR_INPUT=%EN_PREPROCESS_OUT%/%PREPROCESS_TECHNIQUE%

REM Translation directories (created automatically within JAP_OCR_OUT)
set FB_TRANSLATION_DIR=translation_fb
set HL_TRANSLATION_DIR=translation_hl
set NLLB_TRANSLATION_DIR=translation_nllb

REM Typeset configuration
set TYPESET_INPUT_DIR=%JAP_OCR_OUT%
set TYPESET_BASE_JSON_DIR=json
set TYPESET_OG_DIR=resized_original
set TYPESET_BEST_JSON_DIR=translation_best
set TYPESET_BEST_RENDER_DIR=translation_best_render
set TYPESET_DEBUG_DIR=translation_best_debug
set TYPESET_PAD=1
set TYPESET_MAX_FONT=42
set TYPESET_MIN_FONT=10

REM Evaluation directories
set EVAL_GT_DIR=%EN_OCR_OUT%/json
set EVAL_PRED_DIR=%TYPESET_INPUT_DIR%/%TYPESET_BEST_JSON_DIR%
set EVAL_OUT_DIR=outputs/csv_metrics
set EVAL_IOU_THRESHOLD=0.1

REM ========================================
REM END CONFIGURATION
REM ========================================

echo.
echo ========================================
echo  STEP 1: PREPROCESS (ADAPTIVE)
echo ========================================
echo.
echo Config:
echo   Japanese input: %JAP_INPUT_DIR%
echo   English input: %EN_INPUT_DIR%
echo   Technique: %PREPROCESS_TECHNIQUE%
echo.

echo Preprocessing Japanese images with %PREPROCESS_TECHNIQUE% technique...
python src/preprocess.py --in_dir %JAP_INPUT_DIR% --out_dir %JAP_PREPROCESS_OUT% --technique %PREPROCESS_TECHNIQUE%
if errorlevel 1 (
    echo ERROR: Japanese preprocessing failed
    exit /b 1
)

echo Preprocessing English images with %PREPROCESS_TECHNIQUE% technique...
python src/preprocess.py --in_dir %EN_INPUT_DIR% --out_dir %EN_PREPROCESS_OUT% --technique %PREPROCESS_TECHNIQUE%
if errorlevel 1 (
    echo ERROR: English preprocessing failed
    exit /b 1
)

echo Preprocessing complete.
echo.

REM ========================================
echo.
echo ========================================
echo  STEP 2: OCR (PADDLE OCR)
echo ========================================
echo.
echo Config:
echo   Japanese input: %JAP_OCR_INPUT%
echo   English input: %EN_OCR_INPUT%
echo.

echo Running OCR on Japanese adaptive images...
python src/pp_ocrv5_jap.py --in_dir %JAP_OCR_INPUT% --out_dir %JAP_OCR_OUT%
if errorlevel 1 (
    echo ERROR: Japanese OCR failed
    exit /b 1
)

echo Running OCR on English adaptive images...
python src/pp_ocrv5_en.py --in_dir %EN_OCR_INPUT% --out_dir %EN_OCR_OUT%
if errorlevel 1 (
    echo ERROR: English OCR failed
    exit /b 1
)

echo OCR complete.
echo.

REM ========================================
echo.
echo ========================================
echo  STEP 3: TRANSLATE (JA-^>EN)
echo ========================================
echo.
echo Config:
echo   Input: %JAP_OCR_OUT%
echo   Outputs: %FB_TRANSLATION_DIR%, %HL_TRANSLATION_DIR%, %NLLB_TRANSLATION_DIR%
echo.

echo Translating Japanese OCR results to English...
python src/translate_jp_to_en.py --in_dir %JAP_OCR_OUT%/json
if errorlevel 1 (
    echo ERROR: Translation failed
    exit /b 1
)

echo Translation complete.
echo.

REM ========================================
echo.
echo ========================================
echo  STEP 4: TYPESET (RENDER WITH BEST TRANSLATION)
echo ========================================
echo.
echo Config:
echo   Root: %TYPESET_INPUT_DIR%
echo   Base JSON: %TYPESET_BASE_JSON_DIR%
echo   Translation dirs: %FB_TRANSLATION_DIR%, %HL_TRANSLATION_DIR%, %NLLB_TRANSLATION_DIR%
echo   Output JSON: %TYPESET_BEST_JSON_DIR%
echo   Output images: %TYPESET_BEST_RENDER_DIR%
echo   Debug output: %TYPESET_DEBUG_DIR%
echo   Font settings: min=%TYPESET_MIN_FONT%, max=%TYPESET_MAX_FONT%, pad=%TYPESET_PAD%
echo.

echo Typesetting with best translation selection...
python src/typeset.py ^
  --root %TYPESET_INPUT_DIR% ^
  --base-json-dir %TYPESET_BASE_JSON_DIR% ^
  --fb-dir %FB_TRANSLATION_DIR% ^
  --hl-dir %HL_TRANSLATION_DIR% ^
  --nllb-dir %NLLB_TRANSLATION_DIR% ^
  --og-dir %TYPESET_OG_DIR% ^
  --out-best-json-dir %TYPESET_BEST_JSON_DIR% ^
  --out-render-dir %TYPESET_BEST_RENDER_DIR% ^
  --out-debug-dir %TYPESET_DEBUG_DIR% ^
  --pad %TYPESET_PAD% ^
  --max-font %TYPESET_MAX_FONT% ^
  --min-font %TYPESET_MIN_FONT%
if errorlevel 1 (
    echo ERROR: Typesetting failed
    exit /b 1
)

echo Typesetting complete.
echo.

REM ========================================
echo.
echo ========================================
echo  STEP 5: EVALUATE (METRICS)
echo ========================================
echo.
echo Config:
echo   Ground truth: %EVAL_GT_DIR%
echo   Predictions: %EVAL_PRED_DIR%
echo   Output: %EVAL_OUT_DIR%
echo   IoU threshold: %EVAL_IOU_THRESHOLD%
echo.

echo Evaluating results...
python src/eval.py ^
  --gt_dir %EVAL_GT_DIR% ^
  --pred_dir %EVAL_PRED_DIR% ^
  --out_dir %EVAL_OUT_DIR% ^
  --iou_threshold %EVAL_IOU_THRESHOLD%
if errorlevel 1 (
    echo ERROR: Evaluation failed
    exit /b 1
)

echo Evaluation complete.
echo.

REM ========================================
echo.
echo ========================================
echo  PIPELINE COMPLETE
echo ========================================
echo.
echo Results saved in:
echo   - Preprocessed images: %JAP_PREPROCESS_OUT%\%PREPROCESS_TECHNIQUE%, %EN_PREPROCESS_OUT%\%PREPROCESS_TECHNIQUE%
echo   - OCR results: %JAP_OCR_OUT%, %EN_OCR_OUT%
echo   - Translations: %JAP_OCR_OUT%\%FB_TRANSLATION_DIR%, %HL_TRANSLATION_DIR%, %NLLB_TRANSLATION_DIR%
echo   - Best typeset: %TYPESET_INPUT_DIR%\%TYPESET_BEST_JSON_DIR%
echo   - Rendered images: %TYPESET_INPUT_DIR%\%TYPESET_BEST_RENDER_DIR%
echo   - Debug logs: %TYPESET_INPUT_DIR%\%TYPESET_DEBUG_DIR%
echo   - Metrics: %EVAL_OUT_DIR%\per_box_metrics.csv, per_page_metrics.csv
echo.
echo ========================================
echo.

endlocal