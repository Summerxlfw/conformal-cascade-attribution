#!/usr/bin/env python3
"""Recompute SLURP partial-rho bootstrap rows from results/slurpE_decisive.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
N_BOOT = 3000


def partial_spearman(x, y, z):
    rx = st.rankdata(x)
    ry = st.rankdata(y)
    rz = st.rankdata(z)
    x_res = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    y_res = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(st.spearmanr(x_res, y_res).correlation)


def r3(value):
    return float(f"{value:.3f}")


def main():
    decisive = pd.read_csv(RESULTS / "slurpE_decisive.csv")
    delta = decisive["delta"].to_numpy()
    sz_asr = decisive["sz_asr"].to_numpy()
    signals = [
        (
            "word_lp_std",
            decisive["asr_word_logprob_std"].to_numpy(),
            "yes (ASR-internal, no gold)",
        ),
        (
            "conf_nlu_downstream",
            decisive["conf_nlu"].to_numpy(),
            "yes (downstream NLU-nonconf)",
        ),
        (
            "min_word_lp",
            -decisive["asr_min_word_logprob"].to_numpy(),
            "yes (ASR-internal, no gold)",
        ),
        (
            "wer_gold_reference",
            decisive["wer"].to_numpy(),
            "no (gold transcript required)",
        ),
        (
            "random",
            np.random.default_rng(0).random(len(decisive)),
            "reference only",
        ),
    ]

    rows = []
    for name, values, deployable in signals:
        boot = []
        for b in range(N_BOOT):
            idx = np.random.default_rng(b).integers(0, len(decisive), len(decisive))
            boot.append(partial_spearman(values[idx], delta[idx], sz_asr[idx]))
        lo, hi = np.percentile(np.asarray(boot), [2.5, 97.5])
        rows.append(
            {
                "signal": name,
                "partial_rho": r3(partial_spearman(values, delta, sz_asr)),
                "ci_lo": r3(lo),
                "ci_hi": r3(hi),
                "n_bootstrap": N_BOOT,
                "seed_scheme": (
                    "per-resample np.random.default_rng(b) b=0..2999; "
                    "percentile 2.5/97.5"
                ),
                "deployable": deployable,
            }
        )

    out = RESULTS / "partial_rho_bootstrap.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
