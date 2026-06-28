# Speech-MASSIVE de-DE Analysis Readout

Run class: formal cross-corpus analysis on the official de-DE test split.

Input packet: local prepared `data/massive_de_de` directory (not redistributed).

Primary de-DE result:

- coverage: 0.913
- I_ASR: 0.001
- I_NLU: 0.549
- ASR share: 0.2%
- word_std partial rho: 0.065 [0.029, 0.101]
- telescoping residual: 1.11e-16
- min repair coverage: 0.909
- max repair slack: 0.114

Threshold verdict: FAIL

Notes:

- The de-DE test split is the official Speech-MASSIVE test split (2974 rows), not a carved pseudo-test.
- Confidence missing values come from empty ASR outputs and are filled with a fixed worst-confidence rule fitted on train only.
- `DONE` in the input packet means ASR packet complete; the CSVs here are the Mac-side conformal/statistical analysis.

Cross-corpus summary:

| corpus         | language   | model                                                              |   wer_mean_utt |   wer_corpus |   I_ASR |   I_NLU |   ASR_pct |   coverage |   word_std_partial |   word_std_ci_lo |   word_std_ci_hi |   residual | notes                                                                  |
|:---------------|:-----------|:-------------------------------------------------------------------|---------------:|-------------:|--------:|--------:|----------:|-----------:|-------------------:|-----------------:|-----------------:|-----------:|:-----------------------------------------------------------------------|
| SLURP          | en         | Whisper small.en + all-MiniLM-L6-v2                                |       0.193961 |     nan      |   0.139 |   0.307 |    31.17  |      0.894 |              0.135 |            0.099 |            0.169 |    1.1e-16 | SLURP WER column from slurpF_generality.csv is mean per-utterance WER. |
| Speech-MASSIVE | de-DE      | Whisper multilingual small + paraphrase-multilingual-MiniLM-L12-v2 |       0.1699   |       0.1521 |   0.001 |   0.549 |     0.183 |      0.913 |              0.065 |            0.029 |            0.101 |    1.1e-16 | True gated test audio; no carved test; id dtype=str.                   |
