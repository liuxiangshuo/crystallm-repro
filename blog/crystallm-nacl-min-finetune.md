# CrystaLLM 复现：NaCl 最小微调闭环（中英双语）
# CrystaLLM Reproduction: Minimal NaCl Finetune Loop (Bilingual)

---

## 0. 摘要 / TL;DR

### 中文
我在学校服务器（无 sudo、外网不稳定）上，把 CrystaLLM 从“能跑 demo”推进到“能训练 + 生成 + 解析评估”的完整闭环。  
用 10 个 NaCl CIF 作为最小训练集微调后，生成样本评估结果为：**validity 10/10、NaCl 命中 10/10、uniqueness(valid) 4/10**。  
这不是论文级性能复现，而是**工程管线闭环复现**：可重复、可审计、可继续扩展。

### English
On a restricted university GPU server (no sudo, unstable internet), I pushed CrystaLLM from a runnable demo to a full **train → generate → postprocess → parse/evaluate** loop.  
With a minimal dataset of 10 NaCl CIFs, evaluation shows: **validity 10/10, NaCl hit rate 10/10, uniqueness(valid) 4/10**.  
This is not paper-level performance reproduction; it is an **engineering reproduction of the end-to-end pipeline**.

---

## 1. 环境 / Environment

### 中文
- 远程开发：VS Code Remote-SSH  
- 服务器：Ubuntu 20.04（node01），GPU：RTX 4090D  
- 权限限制：无 sudo；系统 Python 无 pip/venv  
- Python：Miniconda `~/miniconda3`，conda env `myenv`（Python 3.10.19）  
- PyTorch：2.0.1 + CUDA 11.8（CUDA 可用）  
- 外网不稳：GitHub/Zenodo 访问不稳定 → 本机下载 ZIP/权重 + scp 上传到服务器

### English
- Remote dev: VS Code Remote-SSH  
- Server: Ubuntu 20.04 (node01), GPU: RTX 4090D  
- Constraints: no sudo; system Python lacks pip/venv  
- Python: Miniconda `~/miniconda3`, conda env `myenv` (Python 3.10.19)  
- PyTorch: 2.0.1 + CUDA 11.8 (CUDA available)  
- Unstable internet: GitHub/Zenodo unreliable → download ZIP/weights locally + scp upload to server

---

## 2. 仓库与目录 / Repo Layout

### 中文
复现仓库：`~/projects/crystallm-repro`  
官方代码（不建议提交大文件）：`external/CrystaLLM`  
训练/生成/评估输出：`out/`（通常不提交 ckpt）  
可提交的证据文件建议放：`reports/`

推荐结构：

```text
crystallm-repro/
  external/CrystaLLM/
  data/nacl_min/
    raw_cifs/
    tokens/
      train.bin
      val.bin
      meta.pkl
      tokens.tar.gz
      starts.pkl
  config/
    nacl_ft_small.yaml
  scripts/
    eval_cifs.py
  out/
    nacl_ft_small/
      ckpt.pt
      train.log
    nacl_ft_small_gen_raw/
    nacl_ft_small_gen_processed/
    nacl_ft_small_eval.csv
```
### English

Repo: `~/projects/crystallm-repro`  
Upstream code (not committed): `external/CrystaLLM`  
Artifacts: `out/` (do not commit large ckpt)  
Shareable evidence: `reports/`

---

## 3. 目标 / Goal

### 中文

目标不是重训论文级模型，而是走通可审计闭环：

1. 用可解析的 CIF 构造最小训练集
    
2. CIF → tokens → train（微调）
    
3. 用训练后的 checkpoint 再生成
    
4. postprocess 后用 pymatgen 解析评估，并输出 CSV
    

### English

The goal is not paper-level retraining, but a reproducible pipeline:

1. build a minimal dataset from parseable CIFs
    
2. CIF → tokens → train (finetune)
    
3. generate with the trained checkpoint
    
4. postprocess, parse/evaluate with pymatgen, export CSV
    

---

## 4. 最小数据集 / Minimal Dataset

### 中文

我先用 CrystaLLM demo 生成 10 个 NaCl CIF，确认可被 pymatgen 解析，然后作为最小训练集。  
在外网不稳定的服务器环境下，这种“自生成最小数据集”很适合先验证训练管线可用性。

### English

I generated 10 NaCl CIFs via the demo, verified they are parseable by pymatgen, and used them as a minimal training set.  
This is practical for validating the training pipeline under unstable network constraints.

---

## 5. 关键工程修复 / Key Engineering Fixes

### 中文

1. **block_size 与极小 tokens 不匹配**：默认 `block_size=1024` 会导致 `len(data)-block_size<0`（batch 采样失败），因此将 `block_size` 调整为 `256`。
    
2. **注意力实现差异（attn.bias）**：裁剪 block_size 时假设 `attn.bias` 存在，但某些实现路径下不存在；用 `hasattr(attn, "bias")` 做兼容。
    
3. **resume vs finetune**：裁剪后参数形状变化会与 AdamW 动量状态张量冲突；因此使用（并补丁实现）`finetune`：加载权重、不恢复 optimizer/scaler，并将 iter 归零。
    
4. **评估脚本参数化**：原脚本写死 `demo_processed` 和 `eval_demo.csv`，改为 argparse 支持 `--cif_dir/--out_csv`。
    

### English

1. **block_size vs tiny tokens**: default `block_size=1024` breaks when `len(data)-block_size<0`, so set `block_size=256`.
    
2. **Attention differences (attn.bias)**: block cropping assumed `attn.bias` exists; guarded with `hasattr(attn, "bias")`.
    
3. **resume vs finetune**: cropped params conflict with AdamW momentum tensors; implement `finetune` (load weights only, skip optimizer/scaler, reset iter to 0).
    
4. **Parameterized evaluator**: replaced demo-only hardcoded paths with argparse `--cif_dir/--out_csv`.
    

---

## 6. 训练设置 / Training Setup

### 中文

- 初始化：small checkpoint 权重（finetune 模式）
    
- `block_size=256`、`dtype=float16`、`compile=false`
    
- `max_iters=1000`，每 `100` iter 做一次 train/val 估计并保存 checkpoint
    

### English

- Init: small checkpoint weights (finetune mode)
    
- `block_size=256`, `dtype=float16`, `compile=false`
    
- `max_iters=1000`, evaluate every `100` iters and save checkpoint
    

---

## 7. 结果 / Results

### 中文

在 10 个 NaCl CIF 的最小训练集上完成微调，并跑通 “训练→生成→后处理→解析评估” 闭环。

- **训练现象**：train loss 很快降到 ~0.008，而 val loss 上升到 ~2.x（典型过拟合，符合极小数据预期）。
    
- **生成有效性**：postprocess 无 WARNING，pymatgen 解析成功率 **10/10**。
    
- **目标命中**：reduced formula 全部为 **NaCl（10/10）**。
    
- **多样性**：有效样本 uniqueness(valid) **4/10**（存在重复/坍缩）。
    

训练日志节选（真实输出）：

`step 0:    train loss 0.6435, val loss 0.7154 ... step 1000: train loss 0.0080, val loss 2.6899`

评估输出（真实输出）：

`validity: 10 / 10 uniqueness(all): 4 / 10 uniqueness(valid): 4 / 10 formula counts: {'NaCl': 10}`

### English

Finetuning on 10 NaCl CIFs successfully closes the loop: train → generate → postprocess → parse/evaluate.

- **Training behavior**: train loss quickly drops to ~0.008 while val loss rises to ~2.x (classic overfitting expected for tiny data).
    
- **Validity**: no postprocess WARNING; pymatgen parses **10/10** samples.
    
- **Target hit**: reduced formula is **NaCl for 10/10** samples.
    
- **Diversity**: uniqueness(valid) is **4/10**, indicating repetition / mode collapse.
    

Log excerpt (real output):

`step 0:    train loss 0.6435, val loss 0.7154 ... step 1000: train loss 0.0080, val loss 2.6899`

Eval summary (real output):

`validity: 10 / 10 uniqueness(all): 4 / 10 uniqueness(valid): 4 / 10 formula counts: {'NaCl': 10}`

---

## 8. 讨论 / Discussion

### 中文

这个实验的目标是验证工程闭环稳定性，而不是泛化性能。  
仅 10 条样本会导致：

- **train↓、val↑**：模型快速记住训练分布，泛化变差（过拟合）。
    
- **NaCl 10/10**：目标命中很高，但分布坍缩到单一组成。
    
- **uniqueness 4/10**：样本重复明显，体现小数据微调下的模式坍缩。
    

要提升多样性：扩大训练集（多化学式/空间群/不同晶胞）、调采样（temperature、top_k/top_p）、或加入更严格去重/筛选指标。  
解析时出现的 rounding warnings 属于数值精度处理，不影响 parse_ok（仍为 True）。

### English

This experiment targets pipeline stability rather than generalization. With only 10 samples:

- **train↓, val↑**: memorization and degraded generalization (overfitting).
    
- **NaCl 10/10**: strong target bias collapses the composition distribution.
    
- **uniqueness 4/10**: heavy repetition / mode collapse.
    

To improve diversity: scale up dataset (more compositions/space groups/cell variants), tune sampling (temperature, top_k/top_p), and add stricter dedup/filter metrics.  
Rounding warnings are numerical safeguards and do not invalidate parse_ok.

---

## 9. 复现命令 / Repro Commands

### 中文

核心命令如下（外网不稳时，代码/权重通过本机下载 + scp 上传）：

### English

Core commands below (when internet is unreliable, use local download + scp upload).

训练 / Train:

`cd $REPRO/external/CrystaLLM python bin/train.py --config $REPRO/config/nacl_ft_small.yaml 2>&1 | tee $REPRO/out/nacl_ft_small/train.log`

生成与后处理 / Generate & Postprocess:

`cd $REPRO/external/CrystaLLM rm -f sample_*.cif  python bin/sample.py \   out_dir=$REPRO/out/nacl_ft_small \   start=$'data_Na2Cl2\n' \   num_samples=10 \   top_k=5 \   max_new_tokens=2000 \   device=cuda \   dtype=float16 \   target=file  mkdir -p $REPRO/out/nacl_ft_small_gen_raw mv sample_*.cif $REPRO/out/nacl_ft_small_gen_raw/  python bin/postprocess.py \   $REPRO/out/nacl_ft_small_gen_raw \   $REPRO/out/nacl_ft_small_gen_processed`

评估 / Evaluate:

`cd $REPRO python scripts/eval_cifs.py \   --cif_dir $REPRO/out/nacl_ft_small_gen_processed \   --out_csv $REPRO/out/nacl_ft_small_eval.csv`

---

## 10. 局限与下一步 / Limitations & Next Steps

### 中文

这是“最小闭环验证”，数据量极少，过拟合与低多样性属于预期现象。  
下一步建议做一个对比：**微调前（baseline small） vs 微调后**，并扩展训练集到多化学式/多空间群。

### English

This is a minimal loop validation; tiny data inevitably overfits and collapses diversity.  
Next: run **baseline (pre-finetune) vs post-finetune** comparison and scale dataset to multiple compositions/space groups.