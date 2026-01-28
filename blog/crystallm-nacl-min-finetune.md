CrystaLLM Reproduction: From a Minimal NaCl Finetune Loop to Prompt-Sweep Evaluation (Engineering-Focused)

Scope: This post is an engineering reproduction of the CrystaLLM workflow—not a paper-level performance reproduction. The goal is to build a re-runnable, auditable, end-to-end pipeline:
train → generate → postprocess → parse/evaluate → summarize.

TL;DR

On a restricted university GPU server (no sudo, unstable internet), I pushed CrystaLLM from “demo runs” to a complete pipeline that can:

finetune on a dataset (even tiny)

generate CIF samples from prompts

postprocess to valid CIFs

parse & evaluate with pymatgen and export CSV

summarize results across multiple prompts/models

Minimal NaCl finetune (10 CIFs):

validity: 10/10

NaCl hit rate: 10/10

uniqueness(valid): 4/10
This is expected behavior for tiny data (fast overfit + mode collapse).

Then I extended the workflow to:

Demo5: n=200 comparison (baseline vs NaCl-only FT vs Mix154 FT)

Demo6: prompt sweep (generalization under multiple composition prompts)

Demo7: continued training on a 10k subset + temperature ablation + n=1000 “paper-grade” runs

1. Environment & Constraints

Remote dev: VS Code Remote-SSH

Server: Ubuntu 20.04 (node01), RTX 4090D

Permissions: no sudo; system Python lacks pip/venv

Python: Miniconda ~/miniconda3, conda env myenv (Python 3.10.19)

PyTorch: 2.0.1 + CUDA 11.8 (CUDA available)

Networking: GitHub/Zenodo unreliable → download ZIP/weights locally, then scp to server

2. Repo Layout (Recommended)

Repro repo: ~/projects/crystallm-repro

Upstream code (not committed): external/CrystaLLM

Artifacts: out/ (avoid committing large checkpoints)

Shareable evidence: reports/ (logs, CSVs, markdown summaries)

crystallm-repro/
  external/CrystaLLM/
  data/
    nacl_min/
      raw_cifs/
      tokens/
        train.bin
        val.bin
        meta.pkl
        tokens.tar.gz
        starts.pkl
  config/
    nacl_ft_small.yaml
    mix10000_ft_small.yaml
  scripts/
    eval_cifs.py
    subset_tokenized_by_data_prefix.py
    demo5_summarize.py
    demo6_summarize.py
  out/
    nacl_ft_small/
      ckpt.pt
      train.log
    nacl_ft_small_gen_raw/
    nacl_ft_small_gen_processed/
    nacl_ft_small_eval.csv
  reports/
    demo5_compare_n200/
    demo6_prompt_sweep_n100/
    demo6_plus10k_n200/
    demo6_plus10k_n1000/

3. Goal: A Reproducible, Auditable Loop

This project is not about retraining the paper’s best model. It’s about making the workflow robust and repeatable:

Build a dataset from parseable CIFs

CIF → tokens → finetune

Generate CIFs from prompts using the trained checkpoint

Postprocess, parse/evaluate with pymatgen, export CSV

Summarize results across runs/models/prompts

4. Minimal Dataset: “NaCl-10” to Validate the Pipeline

Under unstable network constraints, the fastest path to test the training pipeline is:

generate a tiny dataset locally from a known-working demo path

verify CIFs are parseable (pymatgen)

finetune and ensure the entire loop completes end-to-end

I generated 10 NaCl CIFs, verified parseability, then used them as a minimal training set.

5. Key Engineering Fixes (What Broke, and What I Changed)
5.1 block_size vs tiny token streams

Default block_size=1024 fails on tiny datasets because sampling can require len(data) - block_size >= 0. With very few tokens you hit:

len(data) - block_size < 0 → batch sampling breaks

Fix: set block_size=256 for the minimal dataset experiments.

5.2 Attention implementation mismatch (attn.bias)

Some code paths assumed attn.bias exists for cropping, but certain attention implementations don’t have it.

Fix: guard with hasattr(attn, "bias") before using it.

5.3 “resume” vs “finetune” when shapes change

After cropping / changing block size, parameter shapes can change. If you try to resume, optimizer states (AdamW momentum tensors) can conflict with new shapes.

Fix: implement a true finetune mode:

load model weights only

do not restore optimizer/scaler state

reset iteration counter to 0

5.4 Parameterize the evaluator

Original evaluator scripts were demo-specific (hardcoded paths like demo_processed and fixed output names).

Fix: add argparse options such as:

--cif_dir

--out_csv

This makes evaluation reusable across all experiments.

6. Training Setup (Minimal NaCl Finetune)

Init: small checkpoint weights (finetune mode)

block_size=256, dtype=float16, compile=false

max_iters=1000, save ckpt & compute train/val estimates every 100 iters

7. Minimal Loop Results (NaCl-10)

This validates the pipeline: train → generate → postprocess → parse/evaluate.

Observations

Training: train loss quickly drops to ~0.008 while val loss rises to ~2.x
→ classic overfit, expected for tiny data

Validity: postprocess produces no warnings; pymatgen parses 10/10

Target hit: reduced formula is NaCl for 10/10

Diversity: uniqueness(valid) 4/10 (repetition / collapse)

Real log excerpt:

step 0:    train loss 0.6435, val loss 0.7154
...
step 1000: train loss 0.0080, val loss 2.6899


Real eval summary:

validity: 10 / 10
uniqueness(all): 4 / 10
uniqueness(valid): 4 / 10
formula counts: {'NaCl': 10}

8. Discussion: Why This Overfits (and Why That’s Fine)

This minimal experiment targets pipeline stability, not generalization:

train↓, val↑: memorization dominates with tiny data

NaCl 10/10: target bias is strong but composition collapses

uniqueness 4/10: repetition indicates mode collapse

To improve diversity:

enlarge dataset (more compositions / space groups / cell variants)

tune sampling (temperature, top_k/top_p)

add stronger de-dup/filter metrics

Rounding warnings during parsing are typically numerical safeguards and don’t necessarily invalidate parse_ok.

9. Repro Commands (Minimal Loop)

When internet is unreliable: download code/weights locally, then scp to the server.

Train
cd $REPRO/external/CrystaLLM
python bin/train.py --config $REPRO/config/nacl_ft_small.yaml 2>&1 | tee $REPRO/out/nacl_ft_small/train.log

Generate + Postprocess
cd $REPRO/external/CrystaLLM

rm -f sample_*.cif
python bin/sample.py \
  out_dir=$REPRO/out/nacl_ft_small \
  start=$'data_Na2Cl2\n' \
  num_samples=10 \
  top_k=5 \
  max_new_tokens=2000 \
  device=cuda \
  dtype=float16 \
  target=file

mkdir -p $REPRO/out/nacl_ft_small_gen_raw
mv sample_*.cif $REPRO/out/nacl_ft_small_gen_raw/

python bin/postprocess.py \
  $REPRO/out/nacl_ft_small_gen_raw \
  $REPRO/out/nacl_ft_small_gen_processed

Evaluate
cd $REPRO
python scripts/eval_cifs.py \
  --cif_dir $REPRO/out/nacl_ft_small_gen_processed \
  --out_csv $REPRO/out/nacl_ft_small_eval.csv

10. Demo5 (n=200): Baseline vs NaCl-only FT vs Mix154 FT

Prompt: data_Na2Cl2
Sampling: top_k=5, max_new_tokens=2000, seed=123, device=cuda, target=file
Generate 200 samples per model. Same pipeline: generate → postprocess → pymatgen parse/eval.

Model	Validity	Formula counts (valid, top8)	Uniqueness (valid)	Avg sites (valid)	Unique SG (valid)
baseline	200/200 (100.0%)	{'NaCl': 159, 'NaClO3': 12, 'NaClO2': 8, 'NaClF2': 6, 'NaClO': 6, 'NaClO4': 4, 'NaClF4': 2, 'NaClF': 2, 'others': 1}	200	4.99	21
nacl_ft_small	196/200 (98.0%)	{'NaCl': 195, 'Na': 1}	196	3.98	7
mix154_ft_small	199/200 (99.5%)	{'NaCl': 64, 'NaClO3': 59, 'NaClO2': 35, 'NaClF': 11, 'NaClO10': 9, 'NaClF2': 6, 'NaClO': 3, 'NaClF3': 2, 'others': 10}	199	10.08	12
Key takeaways

Validity: baseline 200/200, NaCl-only 196/200, Mix154 199/200

Collapse vs diversity: NaCl-only collapses almost entirely to NaCl (195/196 valid). Mix154 spreads probability mass across multiple formulas.

Space-group diversity: baseline 21, NaCl-only 7, Mix154 12 → NaCl-only FT sharply reduces SG coverage.

Repro
NUM_SAMPLES=200 bash scripts/demo5_run.sh DEMO5_ROOT=reports/demo5_compare_n200 TOPK=8
python scripts/demo5_summarize.py

11. Demo6: Prompt Sweep (Generalization Across Composition Prompts)

In Demo6, I evaluate baseline / NaCl-only FT / Mix154(strict) FT under identical sampling across multiple composition prompts (e.g., Na2Cl2, Al2O3, BaTiO3, LiFePO4, MgO, SiO2, TiO2, …), generating N=100 per prompt and using CIF parsing parse_ok as the primary validity metric.

High-level pattern:

NaCl-only FT: excellent hit rate on NaCl-related prompts, but on non-NaCl prompts it frequently drifts back to NaCl → strong mode collapse, weak generalization

baseline: tends to follow prompts more faithfully, but diversity can be limited

Mix154(strict): maintains high validity while improving formula and space-group diversity across prompts

(See reports/demo6_prompt_sweep_n100/summary.md for the full table.)

12. Demo7: 10k Training Subset + Temperature Ablation + n=1000 Runs

This section extends the workflow beyond curated small sets by continuing training on a 10k-sample tokenized subset derived from the official token stream.

What changed vs Demo6

Training data: small curated set → 10k subset from official token stream

New model: mix10000 (continued training on 10k subset)

Evaluation: same CSV → summary.md workflow, then compare:

baseline (pretrained)

mix154 (strict finetune)

mix10000 (10k-subset continued training)

12.1 Getting official tokenized data onto the server

Download tokens_v1_train_val.tar.gz locally, then upload to server:

# server
mkdir -p ~/projects/crystallm-repro/data/official
# scp tokens_v1_train_val.tar.gz into data/official/


Extract:

cd ~/projects/crystallm-repro
mkdir -p data/official/tokens_v1_train_val
tar -xzf data/official/tokens_v1_train_val.tar.gz -C data/official/tokens_v1_train_val

find data/official/tokens_v1_train_val -maxdepth 3 -type f \
  \( -name "train.bin" -o -name "val.bin" -o -name "meta.pkl" \) -print


Example extracted path:

data/official/tokens_v1_train_val/mp_oqmd_nomad_cifs_semisymm_Z_props_2/{train.bin,val.bin,meta.pkl}

12.2 Building a 10k subset from the token stream (train=9000, val=1000)

The official meta.pkl contains vocab metadata (stoi/itos/vocab_size), but not sample start indices.
So I built a subset by scanning for the data_ marker token that denotes sample boundaries.

Script:

scripts/subset_tokenized_by_data_prefix.py

Run:

IN_DIR=data/official/tokens_v1_train_val/mp_oqmd_nomad_cifs_semisymm_Z_props_2
OUT_DIR=data/datasets/mix10000_tokenized

python scripts/subset_tokenized_by_data_prefix.py \
  --in_dir "$IN_DIR" \
  --out_dir "$OUT_DIR" \
  --n_train 9000 \
  --n_val 1000 \
  --min_len 32

ls -lh "$OUT_DIR"/{train.bin,val.bin,meta.pkl}


Result:

data/datasets/mix10000_tokenized/{train.bin,val.bin,meta.pkl}

12.3 Training mix10000 (continued training from pretrained checkpoint)
Pitfalls I hit (worth calling out)

Checkpoint block_size mismatch: pretrained ckpt uses block_size=1024. If you set 2048, training can crash.

init_from semantics: use init_from: resume and place ckpt.pt inside the output directory.

Absolute vs delta iters: pretrained ckpt might report iter=100000. To train “+10k steps”, set max_iters: 110000.

Checkpoint saving: if ckpt isn’t updating, evaluation silently uses old weights.

Minimal config example:

# config/mix10000_ft_small.yaml
out_dir: out/mix10000_ft_small_v2
init_from: resume
dataset: data/datasets/mix10000_tokenized

device: cuda
dtype: float16
compile: false

batch_size: 16
gradient_accumulation_steps: 8

block_size: 1024
max_iters: 110000
eval_interval: 250
log_interval: 10
validate: true
always_save_checkpoint: true


Prepare output dir + ckpt:

mkdir -p out/mix10000_ft_small_v2
cp -f external/CrystaLLM/crystallm_v1_small/ckpt.pt out/mix10000_ft_small_v2/ckpt.pt


Train:

python external/CrystaLLM/bin/train.py --config config/mix10000_ft_small.yaml


Sanity check ckpt updated:

stat -c "%y %s %n" out/mix10000_ft_small_v2/ckpt.pt

12.4 Prompt-sweep evaluation (baseline vs mix154 vs mix10000)

Small prompt suite:

Na2Cl2

LiFePO4

BaTiO3 (also used for ablations)

Automation:

scripts/demo6_run_plus10k.sh

Outputs:

reports/demo6_plus10k_n{N}/summary.md

reports/demo6_plus10k_n{N}/summary.csv

12.5 Temperature ablation (n=200)

I evaluated mix10000 with temperature ∈ {0.8, 0.6, 0.4} at n=200, and saved:

reports/demo6_plus10k_n200/summary_t0.8.md

reports/demo6_plus10k_n200/summary_t0.6.md

reports/demo6_plus10k_n200/summary_t0.4.md

Macro-average results (3 prompts, n=200):

temp	validity_mean	uniqueness_mean	formula_unique_mean	sg_unique_mean
0.8	0.8667	0.2210	21.33	6.33
0.6	0.9150	0.1934	22.67	3.00
0.4	0.9450	0.1371	18.33	2.67

Takeaway: lower temperature improves validity, but reduces diversity—especially space-group diversity.
Among these, t=0.6 is the best balance.

12.6 “Paper-grade” run (n=1000, t=0.6) on 2 prompts

Final run: n=1000, t=0.6, prompts {LiFePO4, Na2Cl2}. Macro-average:

model	validity_mean	uniqueness_mean	formula_unique_mean	sg_unique_mean
baseline	1.0000	0.0230	7	14.0
mix154	0.9955	0.0473	32	14.5
mix10000	0.8785	0.1493	87	5.5

Interpretation

mix154 is the “safe upgrade”: near-baseline validity with moderate diversity gains; SG diversity remains high.

mix10000 shows a strong validity–diversity trade-off: much higher formula diversity + uniqueness, but lower validity and collapsed SG diversity.

Prompt-level highlights:

LiFePO4: mix10000 diversity increases, but validity drops substantially

Na2Cl2: mix10000 stays highly valid but can collapse strongly toward NaCl and a dominant space group

13. What I’d Tune Next

To make mix10000 more usable in practice:

lower learning rate (e.g., 1e-4 or 2e-4) and try fewer additional steps first (e.g., +2k) to avoid breaking parseability

keep t=0.6 as default; also try reducing max_new_tokens (e.g., 800) to improve validity on hard prompts

add stricter quality filters / constraints to penalize malformed structures and encourage SG diversity

14. Repro Checklist (Things I Can Re-run)

Build 10k tokenized subset: scripts/subset_tokenized_by_data_prefix.py

Train continued model: external/CrystaLLM/bin/train.py --config config/mix10000_ft_small.yaml

Evaluate + summarize: scripts/demo6_run_plus10k.sh and saved summaries (summary_t0.*.md)

Final report: reports/demo6_plus10k_n1000/summary.md