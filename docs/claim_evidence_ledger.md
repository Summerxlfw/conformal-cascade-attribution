# Claim Evidence Ledger

Purpose: every important sentence in the manuscript must either appear here or be downgraded.

## Claim Table

| Claim ID | Exact claim sentence | Strength | Claim type | Evidence ID/source | Citation class | Source check | Manuscript location | Boundary sentence | Status |
|---|---|---|---|---|---|---|---|---|---|
| C001 | Under a fixed (ASR-calibrated) deployment threshold, the conformal prediction-set inflation of the speech-to-intent cascade decomposes exactly into an ASR-attributable component and an NLU-intrinsic component, with the ASR stage accounting for 31% of the inflation (I_ASR = 0.139, 95% CI [0.115, 0.164]) and NLU-intrinsic uncertainty for 69% (I_NLU = 0.307, 95% CI [0.282, 0.330]). | main | empirical | E001, E005, E006 | none | result CSV bit-exact | Abstract / Methods / Results | This holds under a fixed ASR-calibrated threshold on SLURP scenario intents (LAC, alpha in [0.05, 0.10]); it does not claim the ASR stage is unimportant in general, and the share shifts under per-condition recalibration. | supported |
| C002 | The best deployable gold-free selective-repair policy is approximately the trivial largest-prediction-set-first policy; ASR-internal word-level confidence adds only a small but statistically robust within-stratum refinement (partial Spearman rho with the ASR-attributable repair benefit, controlling deployed set size, = +0.135, 95% CI [0.098, 0.169]), while downstream NLU-nonconformity and pure-text proxies add no usable signal and the only strong attributor (alignment WER, +0.232) requires gold transcripts. | main | empirical | E002, E003 | none | result CSV bit-exact | Abstract / Results / Discussion | This is a precision-bounded characterization on SLURP under the fixed-threshold endpoint: the gold-free refinement is statistically positive but practically small (below the pre-specified operational +0.15 bar), and does not establish that gold-free ASR-side targeting is a useful actionable method or that it beats WER. | supported |
| C003 | The selective-repair audit maintains marginal coverage at or above 1-alpha across all tested budgets (minimum 0.900), with a quantified repair-induced-drop slack of at most 0.051. | support | system | E003 | none | result CSV bit-exact | Methods / Results | This is an empirical, conditional property (holding when repair does not increase nonconformity), not an unconditional coverage theorem; slack is reported, not eliminated. | supported |
| C004 | Replacing the ASR system with a stronger one (medium.en, lower WER) is accompanied by a smaller ASR-attributable inflation (I_ASR 0.139 to 0.076) while the decomposition structure and the gold-free signal persist, providing a same-corpus recognizer-strength check of the attribution. | support | empirical | E005 | none | result CSV bit-exact | Results | This uses two ASR systems on a single corpus (SLURP); it is a directional recognizer-strength check, not multi-dataset or cross-domain generality. | supported |
| C005 | Existing conformal methods that appear closest occupy different objectives or axes: cascaded-module calibration targets regression interval reliability, PASC targets joint multi-stage coverage, modular residual-decomposition and ConformaDecompose methods decompose regression interval uncertainty, fairness work decomposes group-level classification set-size disparity, agent-attribution work locates failure steps, and MiRD decomposes miscoverage risk; none performs fixed-threshold per-stage classification-set inefficiency attribution with gold-free repair. | background | empirical | E004, E007, E008, E009, E010, E011, E014, E015 | evidence-bearing | PDF-fulltext / arXiv source-level audit pass; VOR metadata locked where available; arXiv-only 2025/2026 preprints still need final Zotero ownership check | Related Work | Differentiation is by objective and decomposition axis, not a benchmark win or a first-CP-on-pipelines claim. | supported |
| C006 | Under the same data split, target error rate, and no-test-retuning evaluation protocol, the deployed split-conformal predictor (coverage 0.894, mean set size 1.446) has similar realized coverage and set size to a Mondrian class-conditional variant (0.898, 1.595) and a simplified PASC-style joint-maximum adaptation using an ASR-confidence proxy for the upstream stage (audited joint coverage 0.894, 1.450), but none of these reproduced alternatives apportions the set-size inflation across stages or repairs the input. | support | empirical | E004 | none | result CSV bit-exact | Results | This is an objective-differentiation reproduction on SLURP under the same split/alpha/no-retuning protocol; the joint-maximum predictor is a simplified PASC-style adaptation, and the comparison is not a benchmark performance claim. External attribution of these objectives to prior work is carried by C005 in Related Work. | supported |
| C007 | With the conformal threshold fixed, the per-stage decomposition is an algebraic telescoping identity, and coverage loss after selective repair can only come from repaired instances whose true-label nonconformity score increases under the repaired input; the observed repair-induced-drop slack quantifies this residual case. | support | theoretical / system | E001, E003 | none | algebraic proof plus result CSV audit | Methods | This is a conditional accounting and coverage-audit statement, not an unconditional finite-sample guarantee for arbitrary repair operators. | supported |
| C008 | Two Speech-MASSIVE follow-up checks did not provide positive second-corpus support: de-DE completed on the official test split but failed the pre-specified ASR-share threshold (0.183% < 8%), while ar-SA hit a protocol blocker because the public audio train split contains only train_115=115 rows rather than the required 11,514 rows. | support | empirical boundary | E012, E013 | none | result CSV + protocol check | Discussion / Limitations | These follow-up checks are boundary evidence and stopping evidence for language expansion; they do not invalidate the SLURP decomposition, and they must not be described as multi-corpus confirmation or as ar-SA ASR failure. | supported |

## Evidence Table

| Evidence ID | Evidence source | Evidence type | Result / citation | Reliability | Limit |
|---|---|---|---|---|---|
| E001 | results/decomposition_summary.csv | experiment | I_ASR=0.139 CI[0.115,0.164]; I_NLU=0.307 CI[0.282,0.330]; ASR 31% / NLU 69%; coverage 0.894; residual 1.1e-16 (telescoping exact). Figure F2 from this CSV. | high | fixed ASR-calibrated threshold, SLURP scenario, alpha=0.10; descriptive shares |
| E002 | results/slurpE_decisive.csv | experiment | partial rho(signal, Delta given sz_asr): word_lp_std +0.135 CI[0.098,0.169]; NLU-nonconf -0.152 CI[-0.177,-0.127]; min_word_lp -0.126 CI[-0.161,-0.091]; WER (needs gold) +0.232 CI[0.193,0.272]; random ~0.017 CI[-0.020,0.053]. Bootstrap CIs are persisted in results/partial_rho_bootstrap.csv and generated by configs/recompute_partial_rho_bootstrap.py; Figure F3 from this CSV pair. | high | pre-specified operational +0.15 bar not met by point estimate; precision-bounded |
| E003 | results/method_results.csv | experiment | repair policies, budgets 0.10/0.25/0.50: min coverage 0.900; slack <=0.051; combined - sz_asr paired-bootstrap mean = +0.001 CI[-0.003,+0.006] from results/repair_diff_bootstrap.csv (within-resample paired estimator, != full-set point diff 0.003) | high | empirical conditional coverage; gold counterfactual repair |
| E004 | results/baselines_results.csv | experiment | vanilla cov 0.894 set 1.446; Mondrian 0.898 set 1.595; simplified PASC-style joint-max adaptation joint-cov 0.894 set 1.450; none does per-stage attribution | high | PASC stage-1 = simplified ASR-confidence adaptation |
| E005 | results/slurpF_generality.csv | experiment | WER 0.194->0.167; I_ASR 0.139->0.076; coverage 0.903; word_std partial +0.125. Figure F4 from this CSV. | high | same-corpus recognizer-strength check; two ASR systems, single corpus |
| E006 | results/robustness_summary.csv | experiment | ASR-minority share + partial>0 across alpha{.05,.10}/target{scenario,intent_full}/seed under LAC; ASR share 27-32% across granularities (31% scenario / 27% intent_full). alpha=.20 (cov 0.802) and non-randomized APS (cov 0.875, undercovers) are reported as degenerate/undercovering BOUNDARIES, not part of the holds-across set | medium | alpha>0.10 out of scope; APS undercovers -> excluded from robustness claim |
| E007 | https://arxiv.org/abs/2605.18812 | citation | PASC = joint multi-stage coverage via joint-max score; no per-stage attribution/repair (litread card 05_literature/literature/pasc2026.md) | medium | needs Zotero verification |
| E008 | https://arxiv.org/abs/2510.04406 | citation | Decomposition-Based Modular CP for Two-Stage Modeling = sequential-model residual / interval-width decomposition with component scaling and recalibration; not fixed-threshold classification set-size attribution | medium | arXiv-only; needs final Zotero ownership check |
| E009 | https://arxiv.org/abs/2602.16794 | citation | Beyond Procedure = classification set-size disparity decomposition along protected-group / label-cluster fairness axes; not pipeline-stage attribution | medium | needs Zotero verification |
| E010 | https://arxiv.org/abs/2605.06788 | citation | Conformal Agent Error Attribution = conformal localization of failure steps in agent trajectories; prediction set indexes trajectory positions, not class-label set-size inflation | medium | needs Zotero verification |
| E011 | https://arxiv.org/abs/2605.27091 | citation | MiRD = miscoverage-risk decomposition for open-ended QA set-valued prediction; risk decomposition rather than stage-wise set-size decomposition | medium | needs Zotero verification |
| E014 | https://arxiv.org/abs/2309.12510 | citation | Confidence Calibration for Systems with Cascaded Predictive Modules calibrates prediction intervals for cascaded regression systems using module-level validation; it targets system-level interval reliability rather than classification-set inefficiency attribution or repair. | medium | arXiv-only; needs final Zotero ownership check |
| E015 | https://arxiv.org/abs/2604.27149 | citation | ConformaDecompose explains reducible calibration-induced uncertainty through progressive calibration localization for regression intervals; not stage-wise classification-set inflation attribution. | medium | arXiv-only; accepted author version according to arXiv comment, final Springer metadata pending |
| E012 | results/speech_massive_followup_status.csv | experiment | Speech-MASSIVE de-DE true gated test analysis (details in results/massive_de_de/threshold_verdicts.csv, decomposition_summary.csv, and results/cross_corpus_summary.csv): WER(test corpus)=0.1521, coverage=0.913, telescoping residual=1.1e-16, word_std partial +0.065 CI[0.029,0.101], but I_ASR=0.001 and ASR_pct=0.183%, failing the pre-specified ASR_pct>=8% threshold. | high | boundary negative only; multilingual corpus/labels/embeddings differ from SLURP; does not support multi-corpus generality |
| E013 | protocol check summarized in results/speech_massive_followup_status.csv | protocol check | Speech-MASSIVE ar-SA public audio train has only train_115=115 rows, while validation=2033 and official test=2974 are available; the locked train=11514 prerequisite failed, so no ASR package/parquet/npz/DONE was produced. | high | data availability blocker, not ASR performance evidence; no conformal/statistical result exists for ar-SA |

## Template Safety

The default `C001` row is `draft` by design. It prevents a new project from
appearing prose-ready before evidence exists. Do not change a claim to
`supported` until the exact claim sentence, evidence source, result/citation,
source check, manuscript location, and boundary sentence are filled.

If a ledger contains only unfilled placeholder rows (the bracketed
"input-needed" markers from this template) or no parseable `C...` / `E...`
IDs, it is semantically empty even if `check_writing_inputs.sh` reports no ID
conflicts. Writing agents must treat that state as `Input resolution: reject`
for claim-bearing prose.

## Status Values

- `draft`: claim is proposed but not yet checked.
- `supported`: evidence supports the exact wording.
- `weaken`: wording is too strong and must be softened.
- `unsupported`: remove or mark as future work / hypothesis.
- `citation_needed`: needs source-level citation check.
- `forbidden`: contradicts paper spine or protocol.

## Use In Manuscript

Only `supported` claims may enter manuscript prose.

Rules:

- `draft`: do not use in prose.
- `weaken`: rewrite claim and re-check before use.
- `citation_needed`: do not use in prose unless the sentence is explicitly marked `[CITATION NEEDED]` in a draft-only artifact.
- `unsupported`: remove or keep outside manuscript.
- `forbidden`: never use.

If a section needs a non-supported claim, stop and update this ledger first.

## Citation Class Rules

- `none`: claim is supported by project protocol/result rather than external citation.
- `background`: citation gives context but is not load-bearing.
- `evidence-bearing`: citation supports a gap, novelty, comparison, clinical implication, or other load-bearing claim.

Evidence-bearing citation requirements:

- Source check must be `PDF-fulltext` or equivalent source-level verification.
- Metadata-only checks are insufficient.
- Until verified, status must remain `citation_needed`.

## Strength Rules

- `main`: supported by primary evidence under the locked protocol.
- `support`: supports the story but is not the central contribution.
- `exploratory`: hypothesis-generating only.
- `background`: external context; does not prove this paper's contribution.

## Required Boundary Sentence

Every main or clinical/application claim must have a boundary sentence.

Pattern:

> This supports [narrow conclusion] under [setting/assumption], but does not establish [stronger claim].

## Audit Questions

For each claim:

1. Is the sentence exactly supported, or only loosely related to the evidence?
2. Does the evidence measure the object named in the claim?
3. Is the claim type clear: method, data, empirical, clinical, or system?
4. What stronger claim might a reader incorrectly infer?
5. What reviewer attack would invalidate this sentence?
6. Should this claim appear in Abstract, Introduction, Results, Discussion, or nowhere?
