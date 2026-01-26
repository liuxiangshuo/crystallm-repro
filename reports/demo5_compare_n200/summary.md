# Demo5: Baseline vs NaCl-only FT vs Mix154 FT (same prompt & sampling)

Prompt: `data_Na2Cl2`  |  Sampling: `top_k=5, max_new_tokens=2000, seed=123, device=cuda, target=file`

| Model | Validity | Formula counts (valid, top8) | Uniqueness (valid) | Avg sites (valid) | Unique SG (valid) |
|---|---:|---|---:|---:|---:|
| baseline | 200/200 (100.0%) | {'NaCl': 159, 'NaClO3': 12, 'NaClO2': 8, 'NaClF2': 6, 'NaClO': 6, 'NaClO4': 4, 'NaClF4': 2, 'NaClF': 2, 'others': 1} | 200 | 4.99 | 21 |
| nacl_ft_small | 196/200 (98.0%) | {'NaCl': 195, 'Na': 1} | 196 | 3.98 | 7 |
| mix154_ft_small | 199/200 (99.5%) | {'NaCl': 64, 'NaClO3': 59, 'NaClO2': 35, 'NaClF': 11, 'NaClO10': 9, 'NaClF2': 6, 'NaClO': 3, 'NaClF3': 2, 'others': 10} | 199 | 10.08 | 12 |

## Key takeaways

- **Validity**: baseline 200/200, NaCl-only FT 196/200, Mix154 FT 199/200.
- **Collapse vs diversity**: NaCl-only FT is highly concentrated on NaCl (NaCl=195/196 valid), while Mix154 FT spreads mass across multiple formulas (NaCl=64/199 valid).
- **Space-group diversity (valid)**: baseline 21, NaCl-only FT 7, Mix154 FT 12.

## Repro commands
```bash
NUM_SAMPLES=200 bash scripts/demo5_run.sh
DEMO5_ROOT=reports/demo5_compare_n200 TOPK=8 python scripts/demo5_summarize.py
```