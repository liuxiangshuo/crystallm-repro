CrystaLLM Reproduction / 复现记录
本仓库记录我在学校 GPU 服务器上复现 CrystaLLM（生成晶体结构 CIF）的过程与结果。 This repo records my reproduction of CrystaLLM on a university GPU server (CIF generation + evaluation).

Overview / 总览
What’s in this repo? / 仓库里有什么
Demo 1 (Pretrained): 使用官方 CrystaLLM v1 small 在 GPU 上生成 NaCl CIF，并评估 validity/uniqueness

Demo 2 (Finetune, minimal loop): 用 10 个 NaCl CIF 构造最小训练集，完成 train → generate → postprocess → evaluate 闭环，并记录证据

Links / 链接
Blog (bilingual) / 博客长文（中英双语）: blog/crystallm-nacl-min-finetune.md

Evidence / 证据: reports/nacl_min/train.log, reports/nacl_min/nacl_ft_small_eval.csv

Finetune config / 微调配置: config/nacl_ft_small.yaml

Evaluator / 评估脚本: scripts/eval_cifs.py

Environment / 环境
CN
Remote dev: VS Code Remote-SSH 远程开发

OS: Ubuntu 20.04（server）

GPU: NVIDIA GeForce RTX 4090 D（CUDA 可用）

Python: 3.10（conda env：myenv）

PyTorch: 2.0.1 + CUDA 11.8（conda 安装）

Evaluation: pymatgen

EN
Remote dev: VS Code Remote-SSH

OS: Ubuntu 20.04 (server)

GPU: NVIDIA GeForce RTX 4090 D (CUDA available)

Python: 3.10 (conda env: myenv)

PyTorch: 2.0.1 + CUDA 11.8 (installed via conda)

Evaluation: pymatgen

Model / 模型权重
CN
官方 small checkpoint：external/CrystaLLM/crystallm_v1_small/ckpt.pt （服务器外网不稳定时建议本机下载后 scp 上传）

EN
Official small checkpoint: external/CrystaLLM/crystallm_v1_small/ckpt.pt (When internet is unstable, download locally and scp to server)

Demo 1 — Generate NaCl with CrystaLLM v1 small (GPU)
Goal / 目标
CN: 使用官方预训练模型 CrystaLLM v1 small 在 GPU 上生成晶体结构（CIF），并用 pymatgen 验证可解析性（validity）与去重（uniqueness）。

EN: Use the official pretrained CrystaLLM v1 small on GPU to generate CIFs, then evaluate validity (pymatgen parse) and uniqueness (file hash).

Quickstart / 一键复现（生成 + 后处理 + 评估）
Run inside external/CrystaLLM/ 在 external/CrystaLLM/ 目录下运行

0) Activate env / 激活环境
Bash
conda activate myenv
1) Generate CIFs (GPU) / 生成 CIF（GPU）
CN：prompt 使用 data_Na2Cl2；采样：top_k=5、max_new_tokens=2000、num_samples=10 EN：prompt data_Na2Cl2; sampling: top_k=5, max_new_tokens=2000, num_samples=10

Bash
python bin/sample.py \
  out_dir=crystallm_v1_small \
  start=$'data_Na2Cl2\n' \
  num_samples=10 \
  top_k=5 \
  max_new_tokens=2000 \
  device=cuda \
  dtype=float16 \
  target=file
会在当前目录生成 sample_1.cif ... sample_10.cif。

2) Postprocess / 后处理（规范化 CIF）
Bash
rm -rf demo_raw demo_processed
mkdir -p demo_raw demo_processed
mv sample_*.cif demo_raw/

python bin/postprocess.py demo_raw demo_processed
3) Validate one sample (pymatgen) / 单样本解析验证
Bash
python -c "from pymatgen.core import Structure; s=Structure.from_file('demo_processed/sample_1.cif'); print('OK parsed'); print('formula:', s.composition.formula); print('reduced:', s.composition.reduced_formula); print('sites:', len(s))"
Expected example output / 期望示例输出：

Plaintext
formula: Na2 Cl2
reduced: NaCl
sites: 4
4) Evaluate validity & uniqueness / 批量评估
使用本仓库的参数化评估脚本：scripts/eval_cifs.py

Bash
cd ../..  # back to repo root
python scripts/eval_cifs.py \
  --cif_dir out/demo_processed_or_your_dir \
  --out_csv out/eval_demo1.csv
Demo 2 — Minimal NaCl Finetune Loop / 最小 NaCl 微调闭环
Goal / 目标
CN: 用 10 个 NaCl CIF 构造“最小可训练数据集”，在 small checkpoint 上微调，完成 train → generate → postprocess → evaluate 的闭环，并输出可审计证据。

EN: Build a minimal trainable dataset from 10 NaCl CIFs, finetune from the small checkpoint, and close the loop: train → generate → postprocess → evaluate, with auditable artifacts.

Key Results / 关键结果（真实输出）
validity: 10 / 10

reduced formula NaCl: 10 / 10

uniqueness(valid): 4 / 10

Evidence / 证据：

training log: reports/nacl_min/train.log

eval csv: reports/nacl_min/nacl_ft_small_eval.csv

blog post: blog/crystallm-nacl-min-finetune.md

Quickstart / 一键复现（微调 + 生成 + 评估）
Notes / 注意
external/CrystaLLM/ 不纳入 git（需要你自己准备上游代码与 checkpoint）

训练配置文件：config/nacl_ft_small.yaml

评估脚本：scripts/eval_cifs.py

1) Train (finetune) / 微调训练
Bash
cd external/CrystaLLM
python bin/train.py --config ../../config/nacl_ft_small.yaml 2>&1 | tee ../../out/nacl_ft_small/train.log
2) Generate + postprocess / 生成 + 后处理
Bash
rm -f sample_*.cif

python bin/sample.py \
  out_dir=../../out/nacl_ft_small \
  start=$'data_Na2Cl2\n' \
  num_samples=10 \
  top_k=5 \
  max_new_tokens=2000 \
  device=cuda \
  dtype=float16 \
  target=file

mkdir -p ../../out/nacl_ft_small_gen_raw
mv sample_*.cif ../../out/nacl_ft_small_gen_raw/

python bin/postprocess.py \
  ../../out/nacl_ft_small_gen_raw \
  ../../out/nacl_ft_small_gen_processed
3) Evaluate / 评估
Bash
cd ../..
python scripts/eval_cifs.py \
  --cif_dir out/nacl_ft_small_gen_processed \
  --out_csv out/nacl_ft_small_eval.csv
Notes / 备注
Demo 2 的最小数据集会强烈过拟合（train loss ↓，val loss ↑），这是极小样本的预期现象；其意义在于验证工程闭环可用。

若要提升生成多样性与泛化，需要扩大数据集（多化学式/空间群/晶胞变化）并调整采样策略（temperature、top_k/top_p 等）。

License / 许可证与致谢
This repo is a reproduction log. Please follow the upstream CrystaLLM license and citation requirements. 本仓库为复现记录，请遵循上游 CrystaLLM 的许可证与引用要求。
