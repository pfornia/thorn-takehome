# Example run

Output of a full three-complaint search over all five provided base images:

```bash
python -m autosearch --uf all --images images/*.jpg images/*.jpeg --out results/
```

2,388 candidates scored in 6m07s (CPU/MPS). All three targets met.

| complaint | best `dog` | stage 1 | crossings | stage 2 |
|---|---|---|---|---|
| uf1 | 0.9991 | 150 | 59 | 193 |
| uf2 | 1.0000 | 1620 | 136 | 89 |
| uf3 | 0.9998 | 220 | 64 | 101 |

`report.json` holds the full record: baseline scores for every unmodified input, per-stage counts,
the winning parameter set, and the top-K for each complaint.

Images are not committed (they regenerate deterministically from the recorded parameters and seeds).
