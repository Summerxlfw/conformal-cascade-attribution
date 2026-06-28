# Metric Definition

## Primary Metric

- Name: Conformal prediction-set inflation I and its exact per-stage decomposition (I_ASR, I_NLU)
- Formula: For a fixed conformal threshold t calibrated on ASR-devel (LAC, α): C(x) = {k : 1 − p_k(x) ≤ t}. Inflation I = mean_x |C(x)| − 1 (relative to ideal singleton). Per-stage (telescoping identity): I_ASR = mean_x(|C(x_asr)| − |C(x_gold)|), I_NLU = mean_x |C(x_gold)| − 1; I = I_ASR + I_NLU exactly. Per-utterance ASR-attributable repair benefit Δ_i = |C(x_asr,i)| − |C(x_gold,i)|.
- Unit: prediction-set cardinality (count), relative to ideal singleton 1.0.
- Direction: lower I is better (less inefficiency); the decomposition is descriptive (shares).
- Aggregation level: per-utterance set size, macro-averaged over the SLURP test split (2,974 utterances).
- Confidence interval: nonparametric bootstrap, 3000 resamples over test utterances, 95% percentile.

## Secondary Metrics

| Metric | Formula | Purpose | Direction |
|---|---|---|---|
| Marginal coverage | mean_x 1[y ∈ C(x)] | conformal validity (must hold) | ≥ 1 − α (target 0.90) |
| Partial Spearman ρ(signal, Δ ∣ sz_asr) | rank-partial correlation of a gold-free signal with true repair benefit Δ, controlling deployed set size sz_asr | does a gold-free signal add attribution power beyond the trivial baseline? (the decisive test) | higher = more usable; pre-reg GO = > +0.15 |
| Within-stratum / top-budget repair lift | mean Δ over top-b by signal ÷ mean Δ (and set-size reduction at fixed budget under fixed threshold) | actionable repair efficiency vs trivial / random | higher = better targeting |
| Repair-induced-drop slack | fraction of repaired utterances covered under ASR but not under gold | boundary of the empirical coverage-preserving repair property | lower (reported, ≤0.05) |
| Word Error Rate (WER) | edit distance(gold, asr) / len(gold) | context + the gold-requiring upper-reference attributor | lower; reported, not optimized |

## Exclusion Rules

- Invalid sample: audio decode failure → excluded from ASR run (none material).
- Missing label: none (all SLURP utterances have gold transcript + intent).
- Ambiguous label: SLURP scenario / intent_full as provided; not re-adjudicated.
- Failed inference: none; the 4/2974 test utterances whose transcript micro-changed under word-timestamp decoding are **disclosed, not dropped**.

## Reporting Rules

- Decimal places: 3 for set sizes / correlations / coverage; CIs to 3.
- CI format: "X.XXX [lo, hi]" (95% bootstrap percentile).
- Per-dataset reporting: main corpus = SLURP; per-(α / target / CP-score / ASR system) reported in robustness + same-corpus ASR check. Speech-MASSIVE follow-up is reported only as boundary status (de-DE threshold fail; ar-SA protocol blocker), not as positive per-dataset validation.
- Macro / micro averaging: macro over test utterances.
- Patient-level vs image-level: N/A (utterance-level; no patient grouping).

## Metric Anti-cheating Checks

- No test-set threshold tuning: conformal threshold calibrated on ASR-devel, fixed; ASR decoded once.
- No metric switching after seeing results: primary endpoint (decomposition + partial ρ) pre-registered; the earlier "conf>WER targeting" headline was retracted *because* the pre-registered trivial-baseline comparison failed it, not by switching metrics.
- No cherry-picking best seed: ASR decoded once; NLU deterministic; CIs by bootstrap; robustness via 0.8 train-subsample × 3 seeds.
- Exploratory labeling: the learned deployable composite (Ridge on devel) is labeled post-hoc/exploratory; APS undercoverage (0.875, non-randomized) and α=0.20 degeneration are flagged as known limitations, not hidden.
