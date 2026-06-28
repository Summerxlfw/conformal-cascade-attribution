"""
First-party (MIT) reimplementations of the standard split / class-conditional
(Mondrian) / weighted split-conformal routines used by this study.

Written from the standard definitions so the repository carries no third-party
code:
  * LAC nonconformity score  s = 1 - p(true class)        (Sadinle et al., 2019)
  * split conformal threshold = (1-alpha) higher-quantile  (Vovk et al., 2005)
  * class-conditional / Mondrian: a separate threshold per group
                                                          (Romano et al., 2020)
  * weighted split conformal: cumulative-weight quantile  (Tibshirani et al., 2019)

These functions are verified to produce identical quantiles and prediction
sets to the reference baseline implementation on the bundled feature CSVs
(see tests/test_conformal_primitives.py for the standalone property tests).

API matches the call sites in protocol_5x50_all_baselines_2026-05-29.py exactly.
"""
import numpy as np


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---- vanilla split conformal (LAC) ----

def split_cp_calibrate_multiclass(calib_logits, calib_y, alpha):
    probs = softmax_np(calib_logits)
    true_p = probs[np.arange(len(calib_y)), calib_y]
    nonconf = 1.0 - true_p
    return np.quantile(nonconf, 1.0 - alpha, method="higher")


def split_cp_predict_set_multiclass(test_logits, q):
    probs = softmax_np(test_logits)
    C = (1.0 - probs) <= q
    return C, C.sum(axis=1)


# ---- class-conditional / Mondrian split conformal ----

def mondrian_split_cp_calibrate(calib_logits, calib_y, calib_groups, alpha):
    probs = softmax_np(calib_logits)
    quantiles = {}
    for g in np.unique(calib_groups):
        mask = (calib_groups == g)
        if mask.sum() == 0:
            continue
        nonconf = 1.0 - probs[mask, calib_y[mask]]
        quantiles[int(g)] = float(np.quantile(nonconf, 1.0 - alpha, method="higher"))
    return quantiles


def mondrian_split_cp_predict(test_logits, test_groups, quantiles):
    probs = softmax_np(test_logits)
    N, K = probs.shape
    C = np.zeros((N, K), dtype=bool)
    fallback = max(quantiles.values()) if quantiles else 1.0
    for i in range(N):
        q = quantiles.get(int(test_groups[i]), fallback)
        C[i] = (1.0 - probs[i]) <= q
    return C, C.sum(axis=1)


# ---- weighted split conformal ----

def weighted_split_cp_calibrate(calib_logits, calib_y, calib_weights, alpha):
    probs = softmax_np(calib_logits)
    nonconf = 1.0 - probs[np.arange(len(calib_y)), calib_y]
    order = np.argsort(nonconf)
    sorted_scores = nonconf[order]
    cum = np.cumsum(calib_weights[order])
    threshold = (1.0 - alpha) * cum[-1]
    idx = min(np.searchsorted(cum, threshold, side="right"), len(sorted_scores) - 1)
    return float(sorted_scores[idx])


def weighted_split_cp_predict(test_logits, quantile):
    probs = softmax_np(test_logits)
    C = (1.0 - probs) <= quantile
    return C, C.sum(axis=1)


def estimate_density_ratio(X_train, X_test, method="logistic"):
    """Importance weights w(x)=p_test(x)/p_train(x) via a train-vs-test
    classifier (clipped to [0.1, 10], renormalised to sum to N)."""
    if method == "none":
        return np.ones(len(X_train))
    from sklearn.linear_model import LogisticRegression
    X = np.vstack([X_train, X_test])
    z = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X, z)
    p_test = clf.predict_proba(X_train)[:, 1]
    w = np.clip(p_test / (1.0 - p_test + 1e-10), 0.1, 10.0)
    return w * (len(w) / w.sum())


# ---- per-group diagnostics ----

def compute_coverage_by_group(C, y_true, groups):
    out = {}
    for g in np.unique(groups):
        mask = (groups == g)
        if mask.sum() == 0:
            continue
        out[int(g)] = float(C[mask, y_true[mask]].mean())
    return out


def compute_set_size_by_group(set_size, groups):
    out = {}
    for g in np.unique(groups):
        mask = (groups == g)
        if mask.sum() == 0:
            continue
        out[int(g)] = float(set_size[mask].mean())
    return out
