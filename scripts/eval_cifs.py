import argparse
from pathlib import Path
import pandas as pd

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def safe_spacegroup(struct: Structure):
    try:
        sga = SpacegroupAnalyzer(struct, symprec=0.1)
        return int(sga.get_space_group_number()), str(sga.get_space_group_symbol())
    except Exception:
        return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", required=True, help="Directory containing CIF files")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    args = ap.parse_args()

    cif_dir = Path(args.cif_dir)
    paths = sorted(list(cif_dir.rglob("*.cif")))

    rows = []
    for p in paths:
        row = {
            "file": p.name,
            "parse_ok": False,
            "reduced_formula": None,
            "sites": None,
            "spacegroup_number": None,
            "spacegroup_symbol": None,
            "error": None,
        }
        try:
            s = Structure.from_file(p)
            row["parse_ok"] = True
            row["reduced_formula"] = s.composition.reduced_formula
            row["sites"] = len(s)
            sg_num, sg_sym = safe_spacegroup(s)
            row["spacegroup_number"] = sg_num
            row["spacegroup_symbol"] = sg_sym
        except Exception as e:
            row["error"] = str(e)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"written: {out_csv}")

    n = len(df)
    vdf = df[df["parse_ok"] == True]
    valid = len(vdf)

    print(f"validity: {valid} / {n}")

    # uniqueness: all vs valid (by filename here; if you later add struct_hash, update accordingly)
    print(f"uniqueness(all): {df['file'].nunique()} / {n}")
    print(f"uniqueness(valid): {vdf['file'].nunique()} / {valid if valid else 1}")

    # formula counts
    if valid:
        fc = vdf["reduced_formula"].value_counts().to_dict()
    else:
        fc = {}
    print(f"formula counts: {fc}")

    # spacegroup stats
    if valid:
        sg_valid = vdf.dropna(subset=["spacegroup_number"])
        uniq_sg = int(sg_valid["spacegroup_number"].nunique())
        sg_counts = sg_valid["spacegroup_number"].value_counts().head(20).to_dict()
    else:
        uniq_sg = 0
        sg_counts = {}
    print(f"spacegroup unique(valid): {uniq_sg}")
    print(f"spacegroup counts(top20): {sg_counts}")

if __name__ == "__main__":
    main()
