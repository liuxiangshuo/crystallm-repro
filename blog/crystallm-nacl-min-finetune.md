# CrystaLLM Reproduction: From a Minimal NaCl Finetune Loop to Prompt-Sweep Evaluation (Engineering-Focused)

**Scope.** This is an _engineering reproduction_ of the CrystaLLM workflow — not a paper-level performance reproduction. The goal is to build a **re-runnable, auditable end-to-end pipeline**:

**train → generate → postprocess → parse/evaluate → summarize**

---

## Table of contents

- TL;DR
    
- Environment and constraints
    
- Repo layout
    
- Experiment 1: Minimal NaCl finetune loop (NaCl-10)
    
- Engineering fixes
    
- Experiment 2: Demo5 comparison (n=200)
    
- Experiment 3: Demo6 prompt sweep (n=100)
    
- Experiment 4: 10k subset continued training (mix10000)
    
- Experiment 5: Temperature ablation for mix10000 (n=200)
    
- Experiment 6: Paper-grade run (n=1000, t=0.6)
    
- What I would tune next
    
- Repro checklist
    

---

## TL;DR

On a restricted university GPU server (no sudo, unstable internet), I pushed CrystaLLM from “demo runs” to a complete pipeline that can:

- finetune on a dataset (even tiny)
    
- generate CIF samples from prompts
    
- postprocess to parseable CIFs
    
- parse & evaluate with pymatgen and export CSV
    
- summarize results across multiple prompts/models
    

### Minimal NaCl finetune (10 CIFs)

- validity: **10/10**
    
- NaCl hit rate: **10/10**
    
- uniqueness(valid): **4/10**
    

This is expected behavior for tiny data (fast overfit + mode collapse).

Then I extended the workflow to:

- **Demo5**: n=200 comparison (baseline vs NaCl-only FT vs Mix154 FT)
    
- **Demo6**: prompt sweep (generalization under multiple composition prompts)
    
- **Demo7**: continued training on a **10k** subset + temperature ablation + n=1000 “paper-grade” runs
    

---

## Environment and constraints

### Setup

- Remote dev: VS Code Remote-SSH
    
- Server: Ubuntu 20.04 (`node01`), RTX 4090D
    
- Permissions: no sudo; system Python lacks pip/venv
    
- Python: Miniconda `~/miniconda3`, conda env `myenv` (Python 3.10.19)
    
- PyTorch: 2.0.1 + CUDA 11.8 (CUDA available)
    
- Networking: GitHub/Zenodo unreliable on server  
    → download ZIP/weights locally, then `scp` to server
    

---

## Repo layout

Recommended high-level structure:

- Repro repo: `~/projects/crystallm-repro`
    
- Upstream code (not committed): `external/CrystaLLM`
    
- Artifacts: `out/` (avoid committing large checkpoints)
    
- Shareable evidence: `reports/` (logs, CSVs, markdown summaries)
    

Example layout (conceptual):

crystallm-repro/  
external/CrystaLLM/ # upstream (not tracked)  
config/ # finetune configs  
scripts/ # eval/summarize/subset tools  
out/ # checkpoints and training runs (large)  
reports/ # shareable summaries and CSV outputs  
blog/ # long-form notes (this file)

---

## Experiment 1: Minimal NaCl finetune loop (NaCl-10)

### Goal

Prove the full loop is stable end-to-end (not generalization):

- build a minimal dataset
    
- tokenize
    
- finetune from pretrained weights
    
- generate from prompt
    
- postprocess
    
- parse/evaluate with pymatgen
    
- export CSV evidence
    

### Setup

- Init checkpoint: CrystaLLM v1 small (`external/CrystaLLM/crystallm_v1_small/ckpt.pt`)
    
- Dataset: 10 NaCl CIFs (tiny on purpose)
    
- Important settings: `dtype=float16`, `compile=false`, and a smaller `block_size` for tiny token streams
    

### Commands

Train:

`cd $REPRO/external/CrystaLLM python bin/train.py --config $REPRO/config/nacl_ft_small.yaml 2>&1 | tee $REPRO/out/nacl_ft_small/train.log`

Generate + postprocess:

`cd $REPRO/external/CrystaLLM  rm -f sample_*.cif python bin/sample.py \   out_dir=$REPRO/out/nacl_ft_small \   start=$'data_Na2Cl2\n' \   num_samples=10 \   top_k=5 \   max_new_tokens=2000 \   device=cuda \   dtype=float16 \   target=file  mkdir -p $REPRO/out/nacl_ft_small_gen_raw mv sample_*.cif $REPRO/out/nacl_ft_small_gen_raw/  python bin/postprocess.py \   $REPRO/out/nacl_ft_small_gen_raw \   $REPRO/out/nacl_ft_small_gen_processed`

Evaluate:

`cd $REPRO python scripts/eval_cifs.py \   --cif_dir $REPRO/out/nacl_ft_small_gen_processed \   --out_csv $REPRO/out/nacl_ft_small_eval.csv`

### Outputs

- Training log: `out/nacl_ft_small/train.log`
    
- Generated CIFs: `out/nacl_ft_small_gen_raw/`
    
- Postprocessed CIFs: `out/nacl_ft_small_gen_processed/`
    
- Eval CSV: `out/nacl_ft_small_eval.csv`
    

### Results

- validity: 10 / 10
    
- reduced_formula NaCl: 10 / 10
    
- uniqueness(valid): 4 / 10
    

Training behavior (expected with tiny data):

- train loss decreases rapidly (memorization)
    
- val loss increases (overfit)
    

### Interpretation

This experiment is intentionally “too small to generalize.” Its value is that the **pipeline is runnable and auditable**.

---

## Engineering fixes

These were the main issues I hit while making the pipeline robust and reusable.

### 1) block_size vs tiny token streams

**Problem:** default `block_size=1024` can fail on tiny datasets (not enough tokens to sample a full block).  
**Fix:** use smaller block size (e.g., 256) for the minimal dataset runs.

### 2) Attention implementation mismatch (attn.bias)

Some code paths assume `attn.bias` exists for cropping. Certain attention implementations do not have it.  
**Fix:** guard usage with `hasattr(attn, "bias")`.

### 3) “resume” vs “finetune” when shapes change

If you change shapes (e.g., block size), resuming optimizer/scaler state can fail.  
**Fix:** implement a finetune mode:

- load model weights only
    
- do not restore optimizer/scaler state
    
- reset iteration counter
    

### 4) Parameterize the evaluator

The original evaluator was demo-specific.  
**Fix:** make `scripts/eval_cifs.py` reusable via argparse:

- `--cif_dir`
    
- `--out_csv`
    

This made it reusable across Demo5/Demo6/Demo7 without rewriting scripts.

---

## Experiment 2: Demo5 comparison (n=200)

### Goal

Compare three checkpoints under identical sampling:

- baseline (pretrained)
    
- NaCl-only finetune
    
- Mix154(strict) finetune
    

### Setup

- Prompt: `data_Na2Cl2`
    
- Sampling: `top_k=5`, `max_new_tokens=2000`, `seed=123`, `device=cuda`, `target=file`
    
- n=200 per model
    

### Outputs

- Summary: `reports/demo5_compare_n200/summary.md`
    

### Results (n=200, from summary)

- baseline: 200/200 valid; NaCl 159/200; Unique SG 21
    
- nacl_ft_small: 196/200 valid; NaCl 195/196; Unique SG 7
    
- mix154_ft_small: 199/200 valid; NaCl 64/199; Unique SG 12
    

### Interpretation

- NaCl-only finetune increases NaCl hit rate but collapses diversity (SG coverage drops).
    
- Mix154 spreads probability mass across multiple formulas and retains better diversity.
    

---

## Experiment 3: Demo6 prompt sweep (n=100)

### Goal

Evaluate generalization across multiple composition prompts under identical sampling:

- baseline vs NaCl-only vs Mix154(strict)
    

### Setup

- Multiple prompts (e.g., Na2Cl2, Al2O3, BaTiO3, LiFePO4, MgO, SiO2, TiO2, …)
    
- n=100 per prompt per model
    
- Validity metric: `parse_ok` from CIF parsing
    

### Outputs

- `reports/demo6_prompt_sweep_n100/summary.md`
    
- `reports/demo6_prompt_sweep_n100/summary.csv`
    

### High-level pattern observed

- NaCl-only: strong collapse toward NaCl on non-NaCl prompts → weak generalization
    
- baseline: generally follows prompts but diversity is limited
    
- Mix154(strict): high validity and improved formula/spacegroup diversity
    

---

## Experiment 4: 10k subset continued training (mix10000)

### Goal

Move beyond small curated sets by continuing training on a **10k-sample** tokenized subset derived from the official token stream, then evaluate prompt behavior and the validity–diversity trade-off.

### Setup

- Official tokenized archive: `tokens_v1_train_val.tar.gz` (download locally → scp to server)
    
- Extracted dataset path:
    
    - `data/official/tokens_v1_train_val/mp_oqmd_nomad_cifs_semisymm_Z_props_2/`
        
- Build a 10k subset:
    
    - train=9000, val=1000
        
    - slice by scanning for `data_` token boundaries in the token stream
        
- Continue training from pretrained ckpt:
    
    - pretrained ckpt reports `iter=100000`
        
    - to train +10k steps, set `max_iters=110000`
        
    - **block_size must match checkpoint** (`block_size=1024`)
        

### Commands

Extract tokenized archive:

`cd ~/projects/crystallm-repro mkdir -p data/official/tokens_v1_train_val tar -xzf data/official/tokens_v1_train_val.tar.gz -C data/official/tokens_v1_train_val  find data/official/tokens_v1_train_val -maxdepth 3 -type f \   \( -name "train.bin" -o -name "val.bin" -o -name "meta.pkl" \) -print`

Build 10k tokenized subset:

`IN_DIR=data/official/tokens_v1_train_val/mp_oqmd_nomad_cifs_semisymm_Z_props_2 OUT_DIR=data/datasets/mix10000_tokenized  python scripts/subset_tokenized_by_data_prefix.py \   --in_dir "$IN_DIR" \   --out_dir "$OUT_DIR" \   --n_train 9000 \   --n_val 1000 \   --min_len 32  ls -lh "$OUT_DIR"/{train.bin,val.bin,meta.pkl}`

Train mix10000:

`mkdir -p out/mix10000_ft_small_v2 cp -f external/CrystaLLM/crystallm_v1_small/ckpt.pt out/mix10000_ft_small_v2/ckpt.pt  python external/CrystaLLM/bin/train.py --config config/mix10000_ft_small.yaml`

Critical sanity check (don’t skip):

`stat -c "%y %s %n" out/mix10000_ft_small_v2/ckpt.pt`

### Pitfalls that mattered

- If checkpoint saving is not enabled/triggered, evaluation may silently use the old pretrained ckpt.  
    Fix: ensure the config includes `always_save_checkpoint: true` (and validate saving by checking ckpt mtime).
    

---

## Experiment 5: Temperature ablation for mix10000 (n=200)

### Goal

Control the validity–diversity trade-off by sweeping temperature for mix10000 while keeping everything else fixed.

### Setup

- Prompts: Na2Cl2, BaTiO3, LiFePO4
    
- n=200 per prompt per model
    
- Fixed: `top_k=5`, `max_new_tokens=1000`
    
- Temperature: t ∈ {0.8, 0.6, 0.4}
    
- Script: `scripts/demo6_run_plus10k.sh`
    
- Saved summaries:
    
    - `reports/demo6_plus10k_n200/summary_t0.8.*`
        
    - `reports/demo6_plus10k_n200/summary_t0.6.*`
        
    - `reports/demo6_plus10k_n200/summary_t0.4.*`
        

### Macro-average results (mix10000; 3 prompts; n=200)

temp | validity_mean | uniqueness_mean | formula_unique_mean | sg_unique_mean  
0.8 | 0.8667 | 0.2210 | 21.33 | 6.33  
0.6 | 0.9150 | 0.1934 | 22.67 | 3.00  
0.4 | 0.9450 | 0.1371 | 18.33 | 2.67

### Interpretation

- Lower temperature increases validity but reduces diversity (especially SG diversity).
    
- t=0.6 is a reasonable compromise for follow-up evaluations.
    

---

## Experiment 6: Paper-grade run (n=1000, t=0.6)

### Goal

Run a larger evaluation (n=1000) at the selected compromise temperature (t=0.6), for stable conclusions.

### Setup

- Prompts: LiFePO4 and Na2Cl2
    
- n=1000 per prompt
    
- temperature=0.6, top_k=5, max_new_tokens=1000
    
- Compared:
    
    - baseline
        
    - mix154
        
    - mix10000
        

### Outputs

- `reports/demo6_plus10k_n1000/summary.md`
    
- `reports/demo6_plus10k_n1000/summary.csv`
    

### Macro-average (2 prompts, n=1000)

model | validity_mean | uniqueness_mean | formula_unique_mean | sg_unique_mean  
baseline | 1.0000 | 0.0230 | 7 | 14.0  
mix154 | 0.9955 | 0.0473 | 32 | 14.5  
mix10000 | 0.8785 | 0.1493 | 87 | 5.5

### Interpretation

- mix154 is a “stable upgrade”: near-baseline validity, moderate diversity gains, SG diversity preserved.
    
- mix10000 shows a strong validity–diversity trade-off: large increase in formula diversity/uniqueness but reduced validity and collapsed SG diversity (notably on Na2Cl2).
    

---

## What I would tune next

To make mix10000 more usable:

- reduce learning rate (e.g., 1e-4 or 2e-4) and train fewer additional steps first (+2k) to preserve parseability
    
- keep temperature around 0.6; try reducing max_new_tokens (e.g., 800) to improve validity on hard prompts
    
- add stricter filtering or structural hashing to make uniqueness(valid) more rigorous
    

---

## Repro checklist

Things I can re-run:

- build 10k tokenized subset: `scripts/subset_tokenized_by_data_prefix.py`
    
- continued training: `external/CrystaLLM/bin/train.py --config config/mix10000_ft_small.yaml`
    
- eval + summarize: `scripts/demo6_run_plus10k.sh` and saved summaries `summary_t0.*`
    
- final report: `reports/demo6_plus10k_n1000/summary.md`