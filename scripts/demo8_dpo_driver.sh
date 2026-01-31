#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/demo8_dpo_driver.sh LiFePO4
#
# Assumes:
# - dpo-crystallm lives at:   ~/projects/dpo-crystallm
# - CrystaLLM repo lives at:  ~/projects/crystallm-repro/external/CrystaLLM
# - conda envs: myenv, matgl_env, dpo_crystallm
#
# Output:
# - All artifacts live under ~/projects/dpo-crystallm

TARGET="${1:-LiFePO4}"
ROOT=~/projects/dpo-crystallm
CRYST=~/projects/crystallm-repro/external/CrystaLLM
BASE_DIR="$CRYST/crystallm_v1_small"
PKG_DIR="$CRYST/crystallm"

# knobs
TOP_K=10
TEMP=1.0
SEED=123
N=200
MAX_NEW_TOKENS=2000
PAIR_TOP=50
PAIR_BOT=50
DPO_STEPS=300
BETA=0.1
LR=1e-6

STAMP="$(date +%Y%m%d_%H%M%S)"
BATCH="demo8_${TARGET}_k${TOP_K}_t${TEMP}_seed${SEED}_n${N}_${STAMP}"

echo "[1/7] Sample baseline CIFs (official sampler)"
source ~/miniconda3/bin/activate myenv
OUT_BASE="$ROOT/data/raw_cifs_baseline_official/${BATCH}"
mkdir -p "$OUT_BASE"
cd "$OUT_BASE"
python "$CRYST/bin/make_prompt_file.py" "$TARGET" prompt.txt
python "$CRYST/bin/sample.py" out_dir="$BASE_DIR" start=FILE:prompt.txt \
  num_samples="$N" top_k="$TOP_K" temperature="$TEMP" max_new_tokens="$MAX_NEW_TOKENS" seed="$SEED" \
  device=cuda target=file > sample.log 2>&1

echo "[2/7] Validate baseline CIFs"
cd "$ROOT"
VAL_BASE="$ROOT/data/scored/validated_${BATCH}_baseline"
mkdir -p "$VAL_BASE"
python scripts/11_validate_cifs.py --in_dir "$OUT_BASE" --out_dir "$VAL_BASE" > "$VAL_BASE/validate.log" 2>&1

echo "[3/7] Score baseline (MatGL)"
source ~/miniconda3/bin/activate matgl_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
SCO_BASE="$ROOT/data/scored/${BATCH}_baseline_scores.csv"
python "$ROOT/scripts/35_score_dir_matgl.py" --in_dir "$VAL_BASE/valid_cifs" --out_csv "$SCO_BASE" \
  |& tee "$ROOT/data/scored/${BATCH}_baseline_score_matgl.log"

echo "[4/7] Build DPO pairs (top vs bottom)"
source ~/miniconda3/bin/activate myenv
PAIRS="$ROOT/data/dpo_pairs/${BATCH}_pairs.jsonl"
python - <<PY
import csv, json
from pathlib import Path
scores_csv = Path("$SCO_BASE")
valid_dir = Path("$VAL_BASE")/"valid_cifs"
out_path = Path("$PAIRS")
TOP_N=$PAIR_TOP; BOT_N=$PAIR_BOT; MAXLEN=1024
# load
rows=[]
with scores_csv.open() as f:
    r=csv.DictReader(f)
    for row in r:
        if row.get("error"): continue
        row["score"]=float(row["score_e_per_atom"])
        rows.append(row)
rows.sort(key=lambda x:x["score"])  # lower is better
good=rows[:TOP_N]; bad=rows[-BOT_N:]
def read(p): return p.read_text(errors="ignore")
with out_path.open("w") as fout:
    for i,g in enumerate(good):
        b=bad[i%len(bad)]
        ex={"prompt":"$TARGET",
            "chosen": read(valid_dir/g["file"]),
            "rejected": read(valid_dir/b["file"]),
            "chosen_file": g["file"],
            "rejected_file": b["file"],
            "chosen_score_e_per_atom": g["score"],
            "rejected_score_e_per_atom": b["score"]}
        fout.write(json.dumps(ex)+"\n")
print("Wrote pairs:", out_path, "n=", TOP_N)
PY

echo "[5/7] Filter pairs to max 1024 tokens"
source ~/miniconda3/bin/activate dpo_crystallm
PAIRS_F="$ROOT/data/dpo_pairs/${BATCH}_pairs_max1024.jsonl"
PAIRS="$PAIRS" PAIRS_F="$PAIRS_F" python - <<'PY'
import json, sys, types, importlib.util
from pathlib import Path
pkg_dir = Path("~/projects/crystallm-repro/external/CrystaLLM/crystallm").expanduser()
def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod
tok_mod = load_module(pkg_dir / "_tokenizer.py", "tok")
stub = types.ModuleType("crystallm"); stub.CIFTokenizer = tok_mod.CIFTokenizer
sys.modules["crystallm"] = stub
tokenizer = tok_mod.CIFTokenizer()
import os
in_path = Path(os.environ["PAIRS"])
out_path = Path(os.environ["PAIRS_F"])
MAXLEN=1024
kept=dropped=0
with in_path.open() as fin, out_path.open("w") as fout:
    for line in fin:
        ex=json.loads(line)
        prompt=ex["prompt"]+"\n"
        c=tokenizer.encode(tokenizer.tokenize_cif(prompt+ex["chosen"]))
        r=tokenizer.encode(tokenizer.tokenize_cif(prompt+ex["rejected"]))
        if len(c)<=MAXLEN and len(r)<=MAXLEN:
            fout.write(json.dumps(ex)+"\n"); kept+=1
        else:
            dropped+=1
print("Wrote:", out_path, "kept:", kept, "dropped:", dropped)
PY

echo "[6/7] Train DPO (CrystaLLM-specific)"
OUT_DPO="$ROOT/runs/${BATCH}_dpo"
mkdir -p "$OUT_DPO"
python "$ROOT/scripts/32_train_dpo_crystallm.py" \
  --pairs "$PAIRS_F" \
  --ckpt_dir "$BASE_DIR" \
  --pkg_dir "$PKG_DIR" \
  --out_dir "$OUT_DPO" \
  --steps "$DPO_STEPS" \
  --beta "$BETA" \
  --lr "$LR" \
  --device cuda \
  --seed "$SEED" \
  |& tee "$ROOT/runs/${BATCH}_dpo.log"

echo "[7/7] Sample DPO model (official sampler) + validate + score"
source ~/miniconda3/bin/activate myenv
DPO_DIR="$ROOT/models/crystallm_v1_small_dpo_${BATCH}"
mkdir -p "$DPO_DIR"
rsync -a --delete --exclude 'ckpt.pt' "$BASE_DIR/" "$DPO_DIR/"
cp -f "$OUT_DPO/ckpt.pt" "$DPO_DIR/ckpt.pt"

OUT_DPO_SAMP="$ROOT/data/raw_cifs_dpo_official/${BATCH}_dpo"
mkdir -p "$OUT_DPO_SAMP"
cd "$OUT_DPO_SAMP"
python "$CRYST/bin/make_prompt_file.py" "$TARGET" prompt.txt
python "$CRYST/bin/sample.py" out_dir="$DPO_DIR" start=FILE:prompt.txt \
  num_samples="$N" top_k="$TOP_K" temperature="$TEMP" max_new_tokens="$MAX_NEW_TOKENS" seed="$SEED" \
  device=cuda target=file > sample.log 2>&1

cd "$ROOT"
VAL_DPO="$ROOT/data/scored/validated_${BATCH}_dpo"
mkdir -p "$VAL_DPO"
python scripts/11_validate_cifs.py --in_dir "$OUT_DPO_SAMP" --out_dir "$VAL_DPO" > "$VAL_DPO/validate.log" 2>&1

source ~/miniconda3/bin/activate matgl_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
SCO_DPO="$ROOT/data/scored/${BATCH}_dpo_scores.csv"
python "$ROOT/scripts/35_score_dir_matgl.py" --in_dir "$VAL_DPO/valid_cifs" --out_csv "$SCO_DPO" \
  |& tee "$ROOT/data/scored/${BATCH}_dpo_score_matgl.log"

echo "DONE"
echo "Baseline scores: $SCO_BASE"
echo "DPO scores     : $SCO_DPO"
echo "DPO ckpt       : $OUT_DPO/ckpt.pt"
