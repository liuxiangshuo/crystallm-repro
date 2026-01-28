CrystaLLM Reproduction

This repository documents my reproduction and extensions of CrystaLLM on a university GPU server (CIF generation + evaluation + finetuning + prompt generalization), including a 10k-sample continued-training experiment and a paper-grade evaluation.

Overview

What’s in this repo?

Demo1 (Pretrained): Generate CIFs with the official CrystaLLM v1 small checkpoint and evaluate validity/uniqueness.

Demo2 (Minimal finetune loop): Finetune on 10 NaCl CIFs to close the loop: train → generate → postprocess → evaluate.

Demo5: Compare pretrained vs NaCl-only FT vs Mix154(strict) FT under identical sampling (n=200).

Demo6: Prompt sweep generalization across multiple prompts (eval.csv → summary.md/summary.csv).

10k training (“mix10000”): Build a 10k tokenized subset from the official token stream, continue training from the pretrained checkpoint, then evaluate with:

temperature ablation (n=200, t=0.8/0.6/0.4)

paper-grade eval (n=1000, t=0.6, 2 prompts)

Links

Blog: blog/crystallm-nacl-min-finetune.md

Environment

Remote dev: VS Code Remote-SSH

OS: Ubuntu 20.04 (server)

GPU: RTX 4090D (CUDA available)

Python: 3.10 (conda env: myenv)

PyTorch: 2.0.1 + CUDA 11.8

Evaluation: pymatgen (eval outputs parse_ok, reduced_formula, spacegroup_number/symbol, error)

Repository layout

External upstream code (NOT tracked)

external/CrystaLLM/ (downloaded/zip+scp)

Pretrained small checkpoint: external/CrystaLLM/crystallm_v1_small/ckpt.pt

Finetune outputs

out/nacl_ft_small/

out/mix154_ft_small/

out/mix10000_ft_small_v2/ (10k continued training; contains ckpt.pt)

Key report artifacts

Demo5 (n=200): reports/demo5_compare_n200/summary.md

Demo6 (n=100): reports/demo6_prompt_sweep_n100/summary.md and summary.csv

mix10000 temperature sweep (n=200): reports/demo6_plus10k_n200/summary_t0.{8,6,4}.md/.csv

Paper-grade eval (n=1000, t=0.6): reports/demo6_plus10k_n1000/summary.md and summary.csv

Core pipeline

Generation → Postprocess → Eval → Summary

sample.py generates sample_*.cif (writes into the current working directory)

postprocess.py cleans/standardizes CIFs into a processed directory

scripts/eval_cifs.py parses processed CIFs → eval.csv with parse_ok, reduced_formula, spacegroup_number/symbol, error

scripts/demo6_summarize_from_eval.py reads eval.csv only → summary.md/summary.csv (fast and stable; avoids parsing CIFs during summarization)

Metrics

validity = parse_ok ratio

uniqueness(valid) = uniq_valid / num_valid (estimated from available eval fields; no structural hash)

unique_formulas = number of distinct reduced_formula among valid samples

unique_spacegroups = number of distinct spacegroup_number among valid samples

top_formula / top_spacegroup = top-k frequency summaries

Quickstart

Activate environment
cd ~/projects/crystallm-repro
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv
python -V

Demo6 (n=100) — Prompt Sweep

Artifacts

reports/demo6_prompt_sweep_n100/summary.md

reports/demo6_prompt_sweep_n100/summary.csv

Re-summarize from eval.csv
python scripts/demo6_summarize_from_eval.py
--root reports/demo6_prompt_sweep_n100
--out_csv reports/demo6_prompt_sweep_n100/summary.csv
--out_md reports/demo6_prompt_sweep_n100/summary.md
--topk 8

10k tokenized subset → mix10000 training

A) Official tokenized data extracted path
data/official/tokens_v1_train_val/mp_oqmd_nomad_cifs_semisymm_Z_props_2/{train.bin,val.bin,meta.pkl}

B) Build a 10k subset by slicing the token stream at the data_ marker
Script: scripts/subset_tokenized_by_data_prefix.py

IN_DIR=data/official/tokens_v1_train_val/mp_oqmd_nomad_cifs_semisymm_Z_props_2
OUT_DIR=data/datasets/mix10000_tokenized

python scripts/subset_tokenized_by_data_prefix.py
--in_dir "$IN_DIR"
--out_dir "$OUT_DIR"
--n_train 9000
--n_val 1000
--min_len 32

C) Continue training on the 10k subset (mix10000)

Key gotchas (critical)

block_size must be 1024 (matches the pretrained small checkpoint)

use init_from: resume and ensure ckpt.pt exists inside out_dir

pretrained ckpt reports iter=100000; to train +10k steps set max_iters=110000 (absolute)

ensure checkpoint saving; verify ckpt.pt mtime changes before evaluating

Training prep
mkdir -p out/mix10000_ft_small_v2
cp -f external/CrystaLLM/crystallm_v1_small/ckpt.pt out/mix10000_ft_small_v2/ckpt.pt

Train
python external/CrystaLLM/bin/train.py --config config/mix10000_ft_small.yaml

Verify ckpt updated (critical)
stat -c "%y %s %n" out/mix10000_ft_small_v2/ckpt.pt

Demo6 + mix10000 evaluation (baseline vs mix154 vs mix10000)

Script

scripts/demo6_run_plus10k.sh

Temperature ablation (n=200; saved summaries)

reports/demo6_plus10k_n200/summary_t0.8.md/.csv

reports/demo6_plus10k_n200/summary_t0.6.md/.csv

reports/demo6_plus10k_n200/summary_t0.4.md/.csv

Temperature sweep summary (mix10000, 3 prompts, n=200, top_k=5, max_new_tokens=1000)

t=0.8: validity 0.8667, uniqueness 0.2210, formula_unique 21.33, sg_unique 6.33

t=0.6: validity 0.9150, uniqueness 0.1934, formula_unique 22.67, sg_unique 3.00

t=0.4: validity 0.9450, uniqueness 0.1371, formula_unique 18.33, sg_unique 2.67
Takeaway: lower temperature improves validity but reduces diversity; t=0.6 is a better compromise.

Paper-grade evaluation (n=1000, t=0.6; prompts: LiFePO4 + Na2Cl2)
Artifacts

reports/demo6_plus10k_n1000/summary.md

reports/demo6_plus10k_n1000/summary.csv

Macro-average (2 prompts, n=1000, t=0.6)

baseline: validity 1.0000, uniqueness 0.0230, unique_formulas 7, unique_SG 14.0

mix154: validity 0.9955, uniqueness 0.0473, unique_formulas 32, unique_SG 14.5

mix10000: validity 0.8785, uniqueness 0.1493, unique_formulas 87, unique_SG 5.5
Interpretation: mix154 is a stable upgrade; mix10000 increases diversity strongly but reduces validity and collapses SG diversity.

Known pitfalls

conda not found / python2 issues: source conda.sh then conda activate myenv

bash shows “>” and appears to hang: unfinished multiline command; Ctrl+C to cancel

sample.py writes to cwd: always cd into raw_dir; use absolute paths in scripts

eval_cifs.py CLI requires --cif_dir and --out_csv

block_size mismatch: pretrained ckpt uses block_size=1024

checkpoint saving: verify out/mix10000_ft_small_v2/ckpt.pt mtime updates before evaluating

License & citation

This repo is a reproduction log. Please follow upstream CrystaLLM license and citation requirements.