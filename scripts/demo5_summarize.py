from pathlib import Path
import os
import pandas as pd

def summarize(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "parse_ok" not in df.columns:
        raise RuntimeError(f"parse_ok not found in {csv_path}; columns={list(df.columns)}")

    n = len(df)
    vdf = df[df["parse_ok"] == True]
    valid = len(vdf)
    vr = valid / n if n else 0.0

    fc = vdf["reduced_formula"].value_counts().to_dict() if "reduced_formula" in vdf.columns else {}

    # uniqueness(valid): prefer a stable id column if exists; fallback to file
    uniq_valid = None
    for col in ["struct_hash", "cif_hash", "fingerprint", "file", "filename"]:
        if col in vdf.columns:
            uniq_valid = int(vdf[col].nunique())
            break

    sites_mean = float(vdf["sites"].mean()) if "sites" in vdf.columns and valid else None

    # spacegroup diversity if eval provides it
    uniq_sg = None
    if "spacegroup_number" in vdf.columns and valid:
        sg_df = vdf.dropna(subset=["spacegroup_number"])
        uniq_sg = int(sg_df["spacegroup_number"].nunique())

    return n, valid, vr, fc, uniq_valid, sites_mean, uniq_sg

def pct(x): return f"{x*100:.1f}%"
def fnum(x):
    if x is None: return "-"
    if isinstance(x, float): return f"{x:.2f}"
    return str(x)

def main():
    repo = Path("~/projects/crystallm-repro").expanduser()

    # resolve root
    demo5_root = os.environ.get("DEMO5_ROOT", "").strip()
    if demo5_root:
        root = Path(demo5_root).expanduser()
        if not root.is_absolute():
            root = (repo / demo5_root).resolve()
    else:
        num = os.environ.get("NUM_SAMPLES", "").strip()
        cand = repo / "reports" / (f"demo5_compare_n{num}" if num else "demo5_compare")
        root = cand

    out = root / "summary.md"

    rows = []
    for base in ["baseline", "nacl_ft_small", "mix154_ft_small"]:
        p = root / f"{base}_eval.csv"
        n, valid, vr, fc, uniq, sites, uniq_sg = summarize(p)
        rows.append((base, n, valid, vr, fc, uniq, sites, uniq_sg))

    lines = []
    lines.append("# Demo5: Baseline vs NaCl-only FT vs Mix154 FT (same prompt & sampling)\n")
    lines.append("Prompt: `data_Na2Cl2`  |  Sampling: `top_k=5, max_new_tokens=2000, seed=123, device=cuda, target=file`\n")
    lines.append("| Model | Validity | Formula counts (valid) | Uniqueness (valid) | Avg sites (valid) | Unique SG (valid) |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for model, n, valid, vr, fc, uniq, sites, uniq_sg in rows:
        lines.append(f"| {model} | {valid}/{n} ({pct(vr)}) | {fc} | {fnum(uniq)} | {fnum(sites)} | {fnum(uniq_sg)} |")

    lines.append("\n## Repro commands\n```bash\nNUM_SAMPLES=200 bash scripts/demo5_run.sh\nDEMO5_ROOT=reports/demo5_compare_n200 python scripts/demo5_summarize.py\n```")

    out.write_text("\n".join(lines), encoding="utf-8")
    print("[OK] wrote", out)

if __name__ == "__main__":
    main()
