from pathlib import Path
from pymatgen.core import Structure
import csv, hashlib
from collections import Counter

indir = Path("demo_processed")
out = Path("eval_demo.csv")

def file_hash16(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]

rows = []
ok = 0
formula_counter = Counter()

files = sorted(indir.glob("sample_*.cif"))
for p in files:
    rec = {"file": p.name, "hash16": file_hash16(p), "parse_ok": False, "reduced_formula": "", "sites": ""}
    try:
        s = Structure.from_file(p)
        rec["parse_ok"] = True
        rec["reduced_formula"] = s.composition.reduced_formula
        rec["sites"] = len(s)
        ok += 1
        formula_counter[rec["reduced_formula"]] += 1
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    rows.append(rec)

with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file","hash16","parse_ok","reduced_formula","sites","error"])
    w.writeheader()
    for r in rows:
        if "error" not in r: r["error"] = ""
        w.writerow(r)

total = len(rows)
unique_all = len({r["hash16"] for r in rows})
unique_valid = len({r["hash16"] for r in rows if r["parse_ok"]})

print("written:", out)
print("validity:", ok, "/", total)
print("uniqueness(all):", unique_all, "/", total)
print("uniqueness(valid):", unique_valid, "/", max(ok, 1))
print("formula counts:", dict(formula_counter))
