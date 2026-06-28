# Conformal Cascade Attribution

Code and result artifacts for:

**Per-stage conformal inefficiency in a speech-to-intent cascade: a validated decomposition and the limits of gold-free repair targeting**

This repository contains the clean public release for the P13 Neurocomputing submission. It reproduces the fixed-threshold conformal decomposition, selective-repair audit, baseline positioning, and evidence figures from the derived result tables.

## What Is Included

- First-party Python code for split conformal prediction, Mondrian baselines, per-stage decomposition, repair audit, robustness checks, and figure rendering.
- Derived CSV result tables used by the manuscript.
- Protocol and evidence-audit notes needed to interpret the numbers.
- Figure rendering script for Figs. 2-4.

## What Is Not Included

- Raw SLURP audio or transcripts. Please obtain SLURP from the official source and cite Bastianelli et al. (EMNLP 2020).
- Whisper or sentence-transformer weights. These are loaded from their official packages/providers.
- Manuscript source, submission forms, reviewer packets, handoff files, AI figure blueprints, or local harness files.
- Any gated/private data.

## Repository Layout

```text
configs/      Core analysis scripts and conformal primitives
results/      Derived CSV result tables supporting the manuscript
figures/      Evidence-figure rendering script
protocols/    Study protocol and metric definition
docs/         Claim/evidence and novelty audit notes
```

## Quick Check

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python figures/make_evidence_figures.py
```

This regenerates evidence-figure PDFs/SVGs under `figures/figure_outputs/` from the committed CSVs. Full raw-data reruns require preparing `data/slurp_real/` as described in `REPRODUCE.md`.

## License

First-party code in this repository is released under the MIT License. Third-party datasets, models, and libraries retain their own licenses; see `THIRD_PARTY_LICENSES.md`.
