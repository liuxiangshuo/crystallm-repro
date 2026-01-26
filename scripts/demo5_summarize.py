from pathlib import Path
import os
import pandas as pd

TOPK = int(os.environ.get("TOPK", "8"))

def topk_dict(d: dict, k: int):
    if not d:
        return {}
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    top = items[:k]
    rest = items[k:]
    out = dict(top)
    if rest:
        out["others"] = sum(v for _, v in rest)
    return out

def summarize(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "parse_ok" not in df.columns:
        raise RuntimeError(f"parse_ok not found in {csv_path}; columns={list(df.columns)}")

    n = len(df)
    vdf = df[df["parse_ok"] == True]
    valid = len(vdf)
    vr = valid / n if n else 0.0

    # formula counts
    if valid and "reduced_formula" in vdf.columns:
        fc_full = vdf["reduced_formula"].value_counts().to_dict()
        fc = topk_dict(fc_full, TOPK)
    else:
        fc_full, fc = {}, {}

    # uniqueness(valid)
    uniq_valid = None
    for col in ["struct_hash", "cif_hash", "fingerprint", "file", "filename"]:
        if col in vdf.columns:
            uniq_valid = int(vdf[col].nunique())
            break

    sites_mean = float(vdf["sites"].mean()) if valid and "sites" in vdf.columns else None

    # spacegroup diversity
    uniq_sg = None
    if valid and "spacegroup_number" in vdf.columns:
        sg_df = vdf.dropna(subset=["spacegroup_number"])
        uniq_sg = int(sg_df["spacegroup_number"].nunique())

    return n, valid, vr, fc, fc_full, uniq_valid, sites_mean, uniq_sg

def pct(x): return f"{x*100:.1f}%"
def fnum(x):
    if x is None: return "-"
    if isinstance(x, float): return f"{x:.2f}"
    return str(x)

def main():
    repo = Path("~/projects/crystallm-repro").expanduser()

    demo5_root = os.environ.get("DEMO5_ROOT", "").strip()
    if demo5_root:
        root = Path(demo5_root).expanduser()
        if not root.is_absolute():
            root = (repo / demo5_root).resolve()
    else:
        num = os.environ.get("NUM_SAMPLES", "").strip()
        root = repo / "reports" / (f"demo5_compare_n{num}" if num else "demo5_compare")

    out = root / "summary.md"

    rows = []
    models = ["baseline", "nacl_ft_small", "mix154_ft_small"]
    for m in models:
        p = root / f"{m}_eval.csv"
        n, valid, vr, fc, fc_full, uniq, sites, uniq_sg = summarize(p)
        rows.append((m, n, valid, vr, fc, fc_full, uniq, sites, uniq_sg))

    # build markdown
    lines = []
    lines.append("# Demo5: Baseline vs NaCl-only FT vs Mix154 FT (same prompt & sampling)\n")
    lines.append("Prompt: `data_Na2Cl2`  |  Sampling: `top_k=5, max_new_tokens=2000, seed=123, device=cuda, target=file`\n")

    lines.append(f"| Model | Validity | Formula counts (valid, top{TOPK}) | Uniqueness (valid) | Avg sites (valid) | Unique SG (valid) |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for m, n, valid, vr, fc, fc_full, uniq, sites, uniq_sg in rows:
        lines.append(f"| {m} | {valid}/{n} ({pct(vr)}) | {fc} | {fnum(uniq)} | {fnum(sites)} | {fnum(uniq_sg)} |")

    # key takeaways (auto-fill with your numbers)
    # pull baseline/nacl/mix
    d = {m: (n, valid, vr, fc_full, uniq, sites, uniq_sg) for m, n, valid, vr, fc, fc_full, uniq, sites, uniq_sg in rows}
    bn = d["baseline"]; nn = d["nacl_ft_small"]; mn = d["mix154_ft_small"]

    def get_count(full_counts, key):
        return int(full_counts.get(key, 0))

    lines.append("\n## Key takeaways\n")
    lines.append(f"- **Validity**: baseline {bn[1]}/{bn[0]}, NaCl-only FT {nn[1]}/{nn[0]}, Mix154 FT {mn[1]}/{mn[0]}.")
    lines.append(f"- **Collapse vs diversity**: NaCl-only FT is highly concentrated on NaCl (NaCl={get_count(nn[3],'NaCl')}/{nn[1]} valid), while Mix154 FT spreads mass across multiple formulas (NaCl={get_count(mn[3],'NaCl')}/{mn[1]} valid).")
    lines.append(f"- **Space-group diversity (valid)**: baseline {bn[6]}, NaCl-only FT {nn[6]}, Mix154 FT {mn[6]}.\n")

    lines.append("## Repro commands\n```bash\nNUM_SAMPLES=200 bash scripts/demo5_run.sh\nDEMO5_ROOT=reports/demo5_compare_n200 TOPK=8 python scripts/demo5_summarize.py\n```")

    out.write_text("\n".join(lines), encoding="utf-8")
    print("[OK] wrote", out)

if __name__ == "__main__":
    main()
