#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   PYTHON="$CONDA_PREFIX/bin/python" bash scripts/demo6_eval_all.sh reports/demo6_prompt_sweep_n100
#
# Output:
#   reports/.../processed/<prompt>/<model>/eval.csv
#   reports/.../processed/<prompt>/<model>/eval.log

ROOT="${1:-}"
if [[ -z "$ROOT" ]]; then
  echo "[ERROR] missing ROOT arg, e.g. reports/demo6_prompt_sweep_n100"
  exit 1
fi

PROCESSED_ROOT="${ROOT}/processed"
if [[ ! -d "$PROCESSED_ROOT" ]]; then
  echo "[ERROR] processed_root not found: $PROCESSED_ROOT"
  exit 1
fi

PY="${PYTHON:-python}"

run_eval() {
  local cif_dir="$1"
  local out_csv="$2"
  local log_file="$3"

  # IMPORTANT: your eval_cifs.py requires --cif_dir and --out_csv
  PYTHONWARNINGS=ignore $PY scripts/eval_cifs.py \
    --cif_dir "$cif_dir" \
    --out_csv "$out_csv" \
    >"$log_file" 2>&1
}

echo "[EvalAll] root=$ROOT"
echo "[EvalAll] processed_root=$PROCESSED_ROOT"
echo

fail_cnt=0
total_cnt=0
skip_cnt=0

for prompt_dir in "$PROCESSED_ROOT"/*; do
  [[ -d "$prompt_dir" ]] || continue
  prompt_name="$(basename "$prompt_dir")"

  for model_dir in "$prompt_dir"/*; do
    [[ -d "$model_dir" ]] || continue
    model_name="$(basename "$model_dir")"

    out_csv="$model_dir/eval.csv"
    log_file="$model_dir/eval.log"

    if [[ -f "$out_csv" ]]; then
      echo "[SKIP] $prompt_name × $model_name (eval.csv exists)"
      skip_cnt=$((skip_cnt+1))
      continue
    fi

    if ! ls "$model_dir"/*.cif >/dev/null 2>&1; then
      echo "[SKIP] $prompt_name × $model_name (no .cif)"
      skip_cnt=$((skip_cnt+1))
      continue
    fi

    echo "[EVAL] $prompt_name × $model_name"
    total_cnt=$((total_cnt+1))
    if ! run_eval "$model_dir" "$out_csv" "$log_file"; then
      echo "[ERROR] eval failed: $model_dir"
      echo "  see: $log_file"
      fail_cnt=$((fail_cnt+1))
    fi
  done
done

echo
echo "[EvalAll] done. total=$total_cnt skip=$skip_cnt fail=$fail_cnt"
if [[ $fail_cnt -ne 0 ]]; then
  exit 2
fi
