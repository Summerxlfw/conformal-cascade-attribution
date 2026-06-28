# Study Protocol

## Study Objective

- Primary objective: Quantify and validate the per-stage split of conformal prediction-set inflation in the SLURP ASR→NLU (speech→intent) cascade under a fixed deployment threshold, and test whether any deployable gold-free signal improves selective-repair efficiency beyond the trivial largest-set-first policy.
- Secondary objective: (a) generality of the decomposition under a second, stronger ASR (medium.en); (b) robustness across α / target / conformal-score / training-seed; (c) coverage preservation under selective repair.

## Data Sources

| Dataset | Role | N | Label source | Access / license | Notes |
|---|---|---:|---|---|---|
| SLURP train | train (NLU fit on gold) | 11,514 | SLURP gold transcript + gold intent | public (CC BY 4.0) | Whisper small.en/medium.en ASR; all-MiniLM-L6-v2 384-d L2-normalized emb |
| SLURP devel | calibration (conformal threshold on ASR-devel) | 2,033 | SLURP gold | public | deployment calibration口径 |
| SLURP test | test (held out, ASR run once) | 2,974 | SLURP gold | public | gold used only as ground-truth counterfactual (Δ), never fed to predictor |
| Speech-MASSIVE de-DE | follow-up boundary check | 16,521 | AmazonScience/MASSIVE de-DE intent/scenario labels + FBK-MT/Speech-MASSIVE audio | public train/validation + gated official test accepted by user | True gated test, no carved split. ASR packet complete, but cross-corpus decisive threshold failed because ASR-attributable share was 0.183% (<8%). Boundary evidence only, not support for multi-corpus generality. |
| Speech-MASSIVE ar-SA | attempted follow-up | N/A | AmazonScience/MASSIVE labels available; FBK audio incomplete | public train_115/validation + gated test | Protocol blocker: public audio train has only train_115=115 rows, not the locked 11,514-row train split. ASR was not run; no package was produced. |

## Split Plan

- Split unit: utterance (SLURP official train / devel / test splits; no re-splitting).
- Train: 11,514 utterances; NLU classifier fit on gold-transcript embeddings only.
- Validation / Calibration: devel 2,033; conformal threshold calibrated on **ASR-devel** (deployment口径), fixed thereafter.
- Test: 2,974; ASR decoded once; no threshold/model tuning on test.
- External / hidden holdout: none beyond official test split.
- Leakage risks: (1) NLU trained on gold then evaluated on ASR — this is the standard SLU train-clean/test-noisy setting, not leakage; (2) test gold embeddings define the counterfactual Δ = |C_asr|−|C_gold| — they are the ground-truth target, never inputs to the predictor or calibrator.
- Leakage checks: NLU never sees test data; threshold calibrated on devel not test; gold-free signals use only deployment-available info (ASR text / ASR-internal confidence / set size); the gold-requiring WER reference is explicitly flagged non-deployable; 4/2974 test transcript micro-changes (word-timestamp decoding) disclosed, not silently dropped.

## Model / Method

- Method under test: per-stage conformal inefficiency decomposition (fixed deployment threshold) + conditional selective-repair audit. NLU = LogisticRegression(C=10, max_iter=1000) over MiniLM embeddings; conformal = split-conformal LAC (score = 1 − p(true class)), α = 0.10.
- Frozen components: Whisper ASR (small.en / medium.en), all-MiniLM-L6-v2 embedder.
- Trainable components: logistic-regression intent classifier (fit once on gold train).
- Pretraining: all-MiniLM-L6-v2 (HF), Whisper small.en / medium.en (faster-whisper 1.2.1).
- Initialization: LR deterministic (lbfgs); robustness uses 0.8 train-subsample × 3 seeds.

## Baselines

See `baseline_registry.yaml`. (Standard conformal comparators + the decisive trivial largest-set-first targeting baseline + a gold-requiring WER reference + falsified gold-free probes.)

## Metrics

See `metric_definition.md`.

## Statistical Plan

- Primary endpoint: per-stage inflation decomposition (I_ASR, I_NLU shares) AND partial Spearman ρ(gold-free signal, Δ | sz_asr) — the gold-free attribution power beyond the trivial set-size baseline.
- Confidence interval: bootstrap (3000 resamples) over test utterances, 95% percentile.
- Significance test: bootstrap CI excludes 0; pre-specified operational GO threshold for the gold-free signal = partial rho > +0.15 AND within-stratum lift clearly > random.
- Multiple comparison handling: all candidate gold-free signals reported (no selective reporting); the single pre-specified signal verdict governs; learned composites flagged explicitly as post-hoc.
- Seed policy: ASR decoded once (holdout discipline); NLU deterministic; robustness via 0.8 train-subsample × 3 seeds; CIs by bootstrap over test, not seed cherry-picking.

## Fairness Rules

- Same data access: every compared deployable signal sees only deployment-available info (ASR text, ASR-internal confidence, deployed set size); the gold-requiring WER attributor is labeled non-deployable and used only as an upper reference.
- Same augmentation policy: none (no augmentation).
- Same early stopping policy: N/A (LR fit to convergence; no early stopping).
- Same hyperparameter budget: C=10 and α=0.10 fixed a priori; no per-method or test-set tuning.
- Same post-processing: identical fixed conformal threshold (calibrated on ASR-devel) used for all method/baseline comparisons.

## Change Control

Protocol changes after first result must be logged here:

| Date | Change | Reason | Affected results | Approved by |
|---|---|---|---|---|
| 2026-06-25 | Reframe target objective from "actionable gold-free targeting beats WER" to "validated decomposition + measured deployability limit" | gold-free targeting collapsed to ≈ trivial sz_asr baseline under rigorous test (partial/lift) | headline / main claim (C002 now negative-leaning); no experiment numbers changed | summer |
| 2026-06-25 | Decomposition framed under FIXED deployment threshold (not per-condition recalibration) | per-condition recalibration bundled re-calibration benefit into ASR (inflated to ~98%); fixed threshold is deployment-correct (ASR ~31%) | I_ASR / I_NLU shares | summer |
| 2026-06-27 | Speech-MASSIVE cross-corpus expansion closed as boundary evidence, not positive generality | de-DE complete true-test run failed the pre-specified ASR-share threshold (0.183% < 8%); ar-SA lacked a full 11,514-row audio train split (only train_115=115) and hit the stop condition before ASR | multi-corpus claim removed/kept forbidden; SLURP remains the sole main corpus; de-DE/ar-SA become limitation/boundary evidence | summer |
