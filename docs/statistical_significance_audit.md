# Statistical Significance Audit: P13 conformal_cascade

This packet is the project-level statistical significance gate. It covers the
two main empirical claims (C001 decomposition, C002 gold-free targeting). The
support claims C003 (coverage property) and C004 (single-corpus causal check)
are not main empirical claims and are not given per-claim significance blocks.

Validate with: `scripts/check_statistical_significance.sh <project_dir>`.

## Claims Covered
- C001 : Exact per-stage decomposition of conformal set inflation under the fixed deployment threshold (ASR 31.17% / NLU-intrinsic 68.83%).
- C002 : The best deployable gold-free repair signal adds only a small, below-bar within-stratum refinement over the trivial set-size policy.

## Per-Claim Block

### Claim C001
Metric: Conformal prediction-set inflation I and its exact per-stage decomposition (I_ASR, I_NLU)
Point estimate: I_ASR 0.139 (31.17%), I_NLU 0.307 (68.83%); I_total 0.446
Confidence interval: 95% bootstrap (3000 resamples): I_ASR [0.115, 0.164], I_NLU [0.282, 0.330]
N seeds: 3
Seed list reference: E006
Test or justification: The point estimate and 95% bootstrap percentile intervals are from the canonical single deployment run (E001, results/decomposition_summary.csv); both stage intervals exclude zero. The telescoping residual is exact (floating-point precision), so the decomposition is an accounting identity rather than a fitted quantity. The decomposition structure (ASR a minority near 31% of inflation) was independently reproduced across the three training-subsample seeds of E006 (per-seed I_ASR 0.148-0.167, ASR share 31-32%), which is the basis for the N=3 seed reference.
Decision: pass
Boundary: Holds under the fixed ASR-calibrated threshold on SLURP scenario intents (LAC, alpha in [0.05,0.10]); the bootstrap interval quantifies test-utterance resampling uncertainty, while the three-seed reference quantifies training-subsample stability of the structure, not of the exact point estimate.

### Claim C002
Metric: Partial Spearman ρ(signal, Δ ∣ sz_asr)
Point estimate: +0.135 (recognizer-internal word-confidence signal, controlling deployed set size)
Confidence interval: 95% bootstrap (3000 resamples): [0.098, 0.169]
N seeds: 3
Seed list reference: E006
Test or justification: The point estimate and 95% bootstrap percentile interval are from the decisive-test run (E002, results/slurpE_decisive.csv); the interval excludes zero but the point estimate is below the pre-specified operational go bar of +0.15, so the signal is statistically positive yet practically small. The downstream NLU-nonconformity (-0.152) and pure-text proxies do not exceed the trivial set-size baseline, and the only strong attributor (alignment WER, +0.232) requires gold transcripts. The positivity of the word-confidence partial correlation was reproduced across the three training-subsample seeds of E006 (per-seed +0.12 to +0.13), which is the basis for the N=3 seed reference.
Decision: pass
Boundary: This is a precision-bounded characterization on SLURP under the fixed-threshold endpoint: the gold-free refinement is statistically positive but below the pre-specified operational bar, and it does not establish that gold-free ASR-side targeting is actionable or that it beats WER.

## Overall verdict: pass

Derivation: any reject -> reject; any revise -> revise; else pass.

## Boundary
This project reports a single real corpus (SLURP) with two recognizers (Whisper small.en and medium.en). The primary endpoint uncertainty is quantified by nonparametric bootstrap over test utterances (3000 resamples, 95% percentile), and training-subsample stability is quantified by the three-seed robustness sweep (E006). The two main empirical claims are descriptive/attributional under a fixed deployment conformal threshold; no clinical, deployment-performance, cross-dataset, or cross-domain significance is claimed.
