from pathlib import Path
from pymatgen.core import Structure
import csv, hashlib
from collections import Counter
import argparse

def file_hash16(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", required=True, help="Directory containing CIF files")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--pattern", default="*.cif", help="Glob pattern, default: *.cif")
    args = ap.parse_args()

    indir = Path(args.cif_dir).expanduser().resolve()
    out = Path(args.out_csv).expanduser().resolve()

    if not indir.exists():
        raise SystemExit(f"[ERROR] cif_dir not found: {indir}")

    files = sorted(indir.glob(args.pattern))
    # also accept .CIF if pattern is default
    if args.pattern == "*.cif":
        files += sorted(indir.glob("*.CIF"))

    if len(files) == 0:
        raise SystemExit(f"[ERROR] No files matched under {indir} with pattern {args.pattern}")

    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    ok = 0
    formula_counter = Counter()

    for p in files:
        rec = {"file": p.name, "hash16": file_hash16(p), "parse_ok": False,
               "reduced_formula": "", "sites": "", "error": ""}
        try:
            s = Structure.from_file(str(p))
            rec["parse_ok"] = True
            rec["reduced_formula"] = s.composition.reduced_formula
            rec["sites"] = len(s)
            ok += 1
            formula_counter[rec["reduced_formula"]] += 1
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        rows.append(rec)

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file","hash16","parse_ok","reduced_formula","sites","error"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    total = len(rows)
    unique_all = len({r["hash16"] for r in rows})
    unique_valid = len({r["hash16"] for r in rows if r["parse_ok"]})

    print("written:", out)
    print("validity:", ok, "/", total)
    print("uniqueness(all):", unique_all, "/", total)
    print("uniqueness(valid):", unique_valid, "/", max(ok, 1))
    print("formula counts:", dict(formula_counter))

if __name__ == "__main__":
    main()
