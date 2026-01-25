CrystaLLM Reproduction / 复现记录

本仓库记录我在学校 GPU 服务器上复现 CrystaLLM（生成晶体结构 CIF）的过程与结果。
This repo records my reproduction of CrystaLLM on a university GPU server (CIF generation + evaluation).

Demo 1 — Generate NaCl with CrystaLLM v1 small (GPU) / 用 CrystaLLM v1 small 生成 NaCl（GPU）
Goal / 目标

CN：使用官方预训练模型 CrystaLLM v1 small 在 GPU 上生成晶体结构（CIF），并用 pymatgen 验证可解析性（validity）与去重（uniqueness）。

EN：Use the official pretrained CrystaLLM v1 small on GPU to generate CIFs, then evaluate validity (pymatgen parse) and uniqueness (file hash).

Environment / 环境

CN

VS Code Remote-SSH 远程开发

OS：Ubuntu 20.04（server）

GPU：NVIDIA GeForce RTX 4090 D（CUDA 可用）

Python：3.10（conda env：myenv）

PyTorch：2.0.1 + CUDA 11.8（conda 安装）

评估：pymatgen

EN

Remote dev: VS Code Remote-SSH

OS: Ubuntu 20.04 (server)

GPU: NVIDIA GeForce RTX 4090 D (CUDA available)

Python: 3.10 (conda env: myenv)

PyTorch: 2.0.1 + CUDA 11.8 (installed via conda)

Evaluation: pymatgen

Model / 模型权重

CN：模型权重路径：crystallm_v1_small/ckpt.pt（由 crystallm_v1_small.tar.gz 解压得到）

EN：Checkpoint: crystallm_v1_small/ckpt.pt (extracted from crystallm_v1_small.tar.gz)

Quickstart / 一键复现（生成 + 后处理 + 评估）

Run the following inside external/CrystaLLM/
在 external/CrystaLLM/ 目录下运行以下命令

0) Activate env / 激活环境
conda activate myenv

1) Generate CIFs (GPU) / 生成 CIF（GPU）

CN：prompt 使用 data_Na2Cl2；采样参数：top_k=5、max_new_tokens=2000、num_samples=10

EN：prompt data_Na2Cl2; sampling: top_k=5, max_new_tokens=2000, num_samples=10

python bin/sample.py \
  out_dir=crystallm_v1_small \
  start=$'data_Na2Cl2\n' \
  num_samples=10 \
  top_k=5 \
  max_new_tokens=2000 \
  device=cuda \
  target=file


This writes sample_1.cif ... sample_10.cif in the current directory.
会在当前目录生成 sample_1.cif ... sample_10.cif。

2) Postprocess / 后处理（规范化 CIF）
rm -rf demo_raw demo_processed
mkdir -p demo_raw demo_processed
mv sample_*.cif demo_raw/
python bin/postprocess.py demo_raw demo_processed

3) Validate one sample (pymatgen) / 单样本解析验证（pymatgen）
python -c "from pymatgen.core import Structure; s=Structure.from_file('demo_processed/sample_1.cif'); print('OK parsed'); print('formula:', s.composition.formula); print('reduced:', s.composition.reduced_formula); print('sites:', len(s))"


Expected example output / 期望示例输出：

formula: Na2 Cl2

reduced: NaCl

sites: 4

4) Evaluate validity & uniqueness / 批量评估（validity / uniqueness）

CN：使用本仓库脚本 scripts/eval_cifs.py 统计：

validity：可被 pymatgen 解析的比例

uniqueness：基于文件内容 hash 的去重比例

EN：Run scripts/eval_cifs.py for:

validity: pymatgen parse success rate

uniqueness: dedup by file hash

python scripts/eval_cifs.py


Outputs / 输出：

eval_demo.csv（每个样本一行：parse_ok / formula / sites / hash）

terminal stats: validity / uniqueness

My result (NaCl demo) / 本次结果（NaCl demo）

validity: 10/10

uniqueness: 10/10 (by file hash)

Notes / 备注（踩坑记录）

CN：服务器可能无法直连 GitHub / Zenodo（网络/证书/DNS 污染）。常用解决方案：在本机下载后 scp 上传服务器。

EN：Server may fail to access GitHub/Zenodo due to network/TLS/DNS issues. A robust workaround is: download on local machine and upload via scp.

CN：若 postprocess.py 报 SymmOp ... as_xyz_string 错误，原因通常是 pymatgen API 版本差异（新版本是 as_xyz_str）。需要对脚本做兼容补丁（建议记录在 patch 文档中）。

EN：If postprocess.py fails with SymmOp ... as_xyz_string, it is typically a pymatgen API mismatch (as_xyz_str in newer versions). Apply a small compatibility patch and document it.

Repository Layout / 仓库结构

scripts/ — my evaluation scripts / 我写的评估脚本

results/ — small reproducible outputs (optional) / 小规模可复现产出（可选）

external/ — ignored in git (.gitignore) / 外部代码与大文件不进仓库（已忽略）


From scratch: Server setup (VS Code + conda + CUDA PyTorch) / 从零搭建服务器环境（VS Code + conda + CUDA PyTorch）
1) Connect to server via VS Code Remote-SSH / 用 VS Code Remote-SSH 连接服务器

CN

VS Code 安装扩展：Remote - SSH

Ctrl+Shift+P → Remote-SSH: Add New SSH Host...

填写（示例）：

ssh -p <PORT> <USER>@<HOST>


Remote-SSH: Connect to Host... 连接后左下角显示 SSH: <HOST>

EN

Install extension: Remote - SSH

Ctrl+Shift+P → Remote-SSH: Add New SSH Host...

Add host:

ssh -p <PORT> <USER>@<HOST>


Connect and ensure bottom-left shows SSH: <HOST>

2) Install Miniconda without sudo / 无 sudo 安装 Miniconda（装到 home 目录）

CN：在服务器终端执行（不需要 sudo）：

cd ~
mkdir -p ~/installers && cd ~/installers
curl -L -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh || \
wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash miniconda.sh -b -p ~/miniconda3
source ~/miniconda3/bin/activate


EN (no sudo required):

cd ~
mkdir -p ~/installers && cd ~/installers
curl -L -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh || \
wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash miniconda.sh -b -p ~/miniconda3
source ~/miniconda3/bin/activate


If conda create fails with ToS prompts,
CN：按提示接受 ToS 或切换 conda-forge；EN：accept ToS or switch to conda-forge.

3) Create environment / 创建环境

CN/EN

conda create -n myenv python=3.10 -y
conda activate myenv
python -V

4) Install PyTorch (CUDA) / 安装 PyTorch（CUDA 版）

CN：先确认 GPU 可见：

nvidia-smi


EN: confirm GPU:

nvidia-smi


Install (CUDA 11.8 runtime) / 安装（CUDA 11.8 运行时）：

conda install -y pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia


Verify / 验证：

python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

5) Known fixes / 常见问题修复记录
(a) iJIT_NotifyEvent ImportError (MKL/OpenMP mismatch) / torch 导入报 iJIT_NotifyEvent

CN：常见于 MKL/Intel OpenMP 版本过新，尝试降级到 <2024.1（需要 conda-forge）：

conda config --add channels conda-forge
conda config --set channel_priority strict
conda install -y "mkl<2024.1" "intel-openmp<2024.1"


EN: often caused by too-new MKL/Intel OpenMP; downgrade:

conda config --add channels conda-forge
conda config --set channel_priority strict
conda install -y "mkl<2024.1" "intel-openmp<2024.1"

(b) numpy.dtype size changed (binary incompatibility) / numpy 与 pymatgen 二进制不兼容

CN：把 numpy 固定到 <2，用 conda-forge 重新装 pymatgen：

conda install -y -c conda-forge "numpy<2" pymatgen


EN:

conda install -y -c conda-forge "numpy<2" pymatgen


Networking notes (GitHub/Zenodo) / 网络下载问题（GitHub/Zenodo）

CN：服务器可能无法直接访问 GitHub/Zenodo（DNS 污染/网络限制）。推荐：

本机下载 → scp 上传服务器

或让 curl 走本机代理（socks5h://127.0.0.1:<PORT> 确保 DNS 也走代理）

EN: server may not access GitHub/Zenodo (DNS/network). Recommended:

download locally → upload via scp

or use proxy with DNS (socks5h://127.0.0.1:<PORT>)