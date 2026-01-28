#!/usr/bin/env bash
set -euo pipefail

# Quick eval: baseline vs mix154_ft_small vs mix10000_ft_small_v2
# Output:
#   reports/demo6_plus10k_n${NUM_SAMPLES}/processed/<prompt>/<model>/eval.csv
#   reports/demo6_plus10k_n${NUM_SAMPLES}/summary.{csv,md}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NUM_SAMPLES="${NUM_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1000}"
TOP_K="${TOP_K:-5}"
TEMPERATURE="${TEMPERATURE:-0.8}"
SEED="${SEED:-123}"
DEVICE="${DEVICE:-cuda}"

ROOT="reports/demo6_plus10k_n${NUM_SAMPLES}"
PROMPT_DIR="${ROOT}/prompts"
PROC_ROOT="${ROOT}/processed"

mkdir -p "$PROMPT_DIR" "$PROC_ROOT"

PROMPTS=("Na2Cl2" "BaTiO3" "LiFePO4")

MODEL_baseline="${REPO_ROOT}/external/CrystaLLM/crystallm_v1_small"
MODEL_mix154="${REPO_ROOT}/out/mix154_ft_small"
MODEL_mix10000="${REPO_ROOT}/out/mix10000_ft_small_v2"

MAKE_PROMPT="${REPO_ROOT}/external/CrystaLLM/bin/make_prompt_file.py"
SAMPLE_PY="${REPO_ROOT}/external/CrystaLLM/bin/sample.py"
POSTPROCESS_PY="${REPO_ROOT}/external/CrystaLLM/bin/postprocess.py"
EVAL_PY="${REPO_ROOT}/scripts/eval_cifs.py"
SUM_PY="${REPO_ROOT}/scripts/demo6_summarize_from_eval.py"

for p in "${PROMPTS[@]}"; do
  pf="${PROMPT_DIR}/${p}.txt"
  if [[ ! -f "$pf" ]]; then
    python "$MAKE_PROMPT" "$p" "$pf"
  fi
done

run_one() {
  local prompt="$1"
  local model_name="$2"
  local model_dir="$3"

  local prompt_file="${PROMPT_DIR}/${prompt}.txt"
  local start_prompt
  start_prompt="$(cat "$prompt_file")"

  local out_dir="${PROC_ROOT}/${prompt}/${model_name}"
  mkdir -p "$out_dir"

  local raw_dir="${out_dir}/raw"
  local proc_dir="${out_dir}/cifs"
  mkdir -p "$raw_dir" "$proc_dir"

  echo "[RUN] ${prompt} × ${model_name}"

  (
    cd "$raw_dir"
    python "$SAMPLE_PY" \
      out_dir="$model_dir" \
      start="$start_prompt" \
      num_samples="$NUM_SAMPLES" \
      max_new_tokens="$MAX_NEW_TOKENS" \
      top_k="$TOP_K" \
      temperature="$TEMPERATURE" \
      seed="$SEED" \
      device="$DEVICE" \
      target=file \
      > sample.log 2>&1
  )

  python "$POSTPROCESS_PY" "$raw_dir" "$proc_dir" >> "$raw_dir/sample.log" 2>&1

  mkdir -p "${proc_dir}/__too_large"
  find "$proc_dir" -type f -name "*.cif" -size +1M -exec mv -t "${proc_dir}/__too_large" {} + 2>/dev/null || true

  PYTHONWARNINGS=ignore python "$EVAL_PY" \
    --cif_dir "$proc_dir" \
    --out_csv "${out_dir}/eval.csv" \
    > "${out_dir}/eval.log" 2>&1
}

for p in "${PROMPTS[@]}"; do
  run_one "$p" "baseline"   "$MODEL_baseline"
  run_one "$p" "mix154"     "$MODEL_mix154"
  run_one "$p" "mix10000"   "$MODEL_mix10000"
done

python "$SUM_PY" \
  --root "$ROOT" \
  --out_csv "${ROOT}/summary.csv" \
  --out_md  "${ROOT}/summary.md" \
  --topk 8

echo "[OK] wrote ${ROOT}/summary.csv and summary.md"
