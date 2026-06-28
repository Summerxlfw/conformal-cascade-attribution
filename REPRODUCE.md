# Reproduction Notes

## Environment

The scripts were developed with Python 3.10+ on macOS/Linux. `requirements.txt`
pins the versions used for the public-release smoke test on 2026-06-28. Install:

```bash
pip install -r requirements.txt
```

## Figure-Level Reproduction

The committed `results/*.csv` files are sufficient to regenerate the evidence figures:

```bash
python figures/make_evidence_figures.py
```

Expected outputs:

- `figures/figure_outputs/F2_decomposition.pdf`
- `figures/figure_outputs/F3_partial_rho_repair.pdf`
- `figures/figure_outputs/F4_generality.pdf`
- Matching `.svg` and `*_data.csv` files

## Full Raw-Data Rerun

Full reruns require a local `data/slurp_real/` directory with:

```text
slurp_asr.parquet
slurp_asr_logprob.parquet
features_train.npz
features_devel.npz
features_test.npz
```

These files are derived from the official SLURP release, Whisper transcripts, faster-whisper word log-probabilities, and sentence-transformer embeddings. Raw SLURP data and external model weights are not redistributed here.

Once prepared:

```bash
python configs/per_stage_conformal.py
python configs/baselines.py
python configs/recompute_partial_rho_bootstrap.py
python configs/robustness.py
python configs/slurpF_generality.py
```

## Split Hash Discipline

If you prepare local splits, record stable IDs and hashes:

```bash
python scripts/hash_splits.py data/slurp_real
```

If an official SLURP manifest is unavailable in your local checkout, do not fabricate one. Record the available IDs and the data source/version instead.
