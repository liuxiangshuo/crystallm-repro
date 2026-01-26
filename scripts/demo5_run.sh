#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-$HOME/miniconda3/envs/myenv/bin/python}"
REPRO=~/projects/crystallm-repro
CRY=$REPRO/external/CrystaLLM

START_PROMPT="${START_PROMPT:-data_Na2Cl2}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
TOP_K="${TOP_K:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2000}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-123}"
TARGET="${TARGET:-file}"

# IMPORTANT: separate outputs by NUM_SAMPLES to avoid overwriting
OUTROOT="${OUTROOT:-$REPRO/reports/demo5_compare_n${NUM_SAMPLES}}"
mkdir -p "$OUTROOT"

run_one () {
  local tag="$1"
  local model_dir="$2"
  local rundir="$OUTROOT/$tag"
  mkdir -p "$rundir/raw" "$rundir/cifs"

  # clear previous artifacts under this tag
  rm -f "$rundir/raw"/*.cif 2>/dev/null || true
  rm -f "$rundir/cifs"/*.cif 2>/dev/null || true

  echo "==================== $tag ===================="
  echo "[INFO] model_dir(out_dir): $model_dir"
  echo "[INFO] OUTROOT: $OUTROOT"
  echo "[INFO] prompt=$START_PROMPT num_samples=$NUM_SAMPLES top_k=$TOP_K max_new_tokens=$MAX_NEW_TOKENS seed=$SEED"

  ( cd "$rundir/raw" && \
    "$PY" "$CRY/bin/sample.py" \
      out_dir="$model_dir" \
      start="$START_PROMPT" \
      num_samples="$NUM_SAMPLES" \
      top_k="$TOP_K" \
      max_new_tokens="$MAX_NEW_TOKENS" \
      seed="$SEED" \
      device="$DEVICE" \
      target="$TARGET" \
  )

  "$PY" "$CRY/bin/postprocess.py" "$rundir/raw" "$rundir/cifs"
  "$PY" "$REPRO/scripts/eval_cifs.py" --cif_dir "$rundir/cifs" --out_csv "$OUTROOT/${tag}_eval.csv"

  echo "[OK] $tag -> $OUTROOT/${tag}_eval.csv"
}

BASELINE_DIR="$CRY/crystallm_v1_small"
NACL_DIR="$REPRO/out/nacl_ft_small"
MIX154_DIR="$REPRO/out/mix154_ft_small"

run_one baseline "$BASELINE_DIR"
run_one nacl_ft_small "$NACL_DIR"
run_one mix154_ft_small "$MIX154_DIR"

echo "[DONE] CSVs:"
ls -lh "$OUTROOT/"*_eval.csv
