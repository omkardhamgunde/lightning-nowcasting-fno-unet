"""Pooled pixel-hour occurrence scores, not individual-flash counts."""
import math
import numpy as np


def occurrence_metrics(probability, target, threshold):
    p = np.asarray(probability, dtype=np.float32).ravel()
    y = np.asarray(target).ravel()
    if not 0 <= threshold <= 1 or not len(p) or p.shape != y.shape:
        raise ValueError("Invalid threshold or prediction/target dimensions")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected finite probabilities and binary targets")
    pred, truth = p > np.float32(threshold), y == 1
    tp = int((pred & truth).sum()); tn = int((~pred & ~truth).sum())
    fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    def ratio(a, b): return a / b if b else None
    denominator = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    correlation = None
    if np.std(p.astype(np.float64)) > 0 and np.std(y.astype(np.float64)) > 0:
        correlation = float(np.corrcoef(p.astype(np.float64), y.astype(np.float64))[0, 1])
    return {"counts": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            "metrics": {"CSI": ratio(tp, tp + fp + fn), "POD": ratio(tp, tp + fn),
                        "FAR": ratio(fp, tp + fp), "MCC": ratio(float(tp) * tn - float(fp) * fn, denominator),
                        "Pattern_Correlation": correlation, "Brier": float(np.mean((p.astype(np.float64) - y) ** 2))},
            "threshold": threshold, "operator": ">", "total_cells": len(p)}
