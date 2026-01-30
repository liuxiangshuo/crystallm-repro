## Baseline vs DPO (MatGL proxy energy per atom)
Lower (more negative) is better.

| metric | baseline | DPO | DPO − baseline |
|---|---:|---:|---:|
| n | 201 | 200 | -1 |
| min(best) | -6.23358 | -6.24511 | -0.011528 |
| p10 | -5.4476 | -5.4465 | 0.00109918 |
| median | -4.3611 | -4.67015 | -0.309053 |
| mean | -2.76747 | -4.07087 | -1.3034 |
| p90 | 0.397786 | -2.02373 | -2.42151 |
| max(worst) | 31.095 | 20.3838 | -10.7112 |

**Takeaway:** DPO improves the distribution strongly (mean −1.30 eV/atom; 90th percentile −2.42 eV/atom) and reduces extreme high-energy outliers.
