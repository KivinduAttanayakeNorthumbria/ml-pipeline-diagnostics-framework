
# Fault-detection metrics and significance tests.
#
# Framing A (per-prediction): the positive class is "the prediction is WRONG".The flagging system is treated as a detector of wrong predictions:
# flagged unreliable = RED or YELLOW
# flagged reliable = GREEN
# ground truth positive (should be flagged) = (correct == 0)
#
# For each experiment and UQ method under each ablation condition:
# precision, recall, F1  - of the flag-as-unreliable decision
# AUROC - over a continuous risk score
#
# Tests:
# McNemar - per experiment: does UQ+XAI flag a different set than UQ-only?
# Wilcoxon - per dataset: across experiments, is GREEN-accuracy difference between UQ-only and UQ+XAI significant?


import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


PER_SAMPLE_DIR = os.path.join('results', 'per_sample')
OUTPUT_DIR = os.path.join('results', 'analysis')
XAI_THRESHOLD = 0.5
KNOWN_MODELS = ['simple_cnn', 'resnet18', 'fcnn', 'rf', 'xgb']


def parse_experiment_id(filename):
    stem = filename[:-4] if filename.endswith('.csv') else filename
    for model in KNOWN_MODELS:
        token = f"_{model}_"
        if token in stem:
            dataset = stem.split(token)[0]
            remainder = stem.split(token)[1]
            parts = remainder.split('_', 1)
            return {
                'dataset': dataset,
                'model': model,
                'test_case': parts[0],
                'condition': parts[1] if len(parts) > 1 else 'baseline',
            }
    return None


def normalise(x):
    """Min-max normalise to [0, 1]. Flat arrays map to zeros."""
    lo, hi = np.min(x), np.max(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def assign_flags_vec(uncertainty, consistency, uq_threshold, xai_threshold):
    high_uncertainty = uncertainty > uq_threshold
    low_consistency = consistency < xai_threshold
    flags = np.full(len(uncertainty), 'GREEN', dtype=object)
    flags[high_uncertainty | low_consistency] = 'YELLOW'
    flags[high_uncertainty & low_consistency] = 'RED'
    return flags


def detection_metrics(flags, correct):
# Precision/recall/F1 of the 'flag as unreliable' decision.
# positive (y_true) = prediction is wrong  = (correct == 0)
# positive (y_pred) = flagged unreliable   = (flag != GREEN)

    y_true = (correct == 0).astype(int)
    y_pred = (flags != 'GREEN').astype(int)

    # If only one class present in truth, precision/recall are ill-defined
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None, None, None

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    return round(p, 4), round(r, 4), round(f, 4)


def safe_auroc(risk, correct):
# AUROC of risk score vs (prediction is wrong). None if single-class.
    y_true = (correct == 0).astype(int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None
    try:
        return round(roc_auc_score(y_true, risk), 4)
    except ValueError:
        return None


def process_method(df, unc_col, uq_threshold):
    uncertainty = df[unc_col].to_numpy()
    consistency = df['consistency'].to_numpy()
    correct = df['correct'].to_numpy()

    if np.all(np.isnan(uncertainty)):
        return None
    uncertainty = np.nan_to_num(uncertainty, nan=0.0)

    ones = np.ones(len(consistency))
    zeros = np.zeros(len(uncertainty))

    norm_unc = normalise(uncertainty)
    inv_cons = 1.0 - consistency

    # Flags for each condition
    uq_flags = assign_flags_vec(uncertainty, ones, uq_threshold, XAI_THRESHOLD)
    xai_flags = assign_flags_vec(zeros, consistency, uq_threshold, XAI_THRESHOLD)
    both_flags = assign_flags_vec(uncertainty, consistency, uq_threshold, XAI_THRESHOLD)

    # Detection metrics
    uq_p, uq_r, uq_f = detection_metrics(uq_flags, correct)
    xai_p, xai_r, xai_f = detection_metrics(xai_flags, correct)
    both_p, both_r, both_f = detection_metrics(both_flags, correct)

    # AUROC over risk scores
    auroc_uq = safe_auroc(norm_unc, correct)
    auroc_xai = safe_auroc(inv_cons, correct)
    auroc_both = safe_auroc(norm_unc + inv_cons, correct)

    # McNemar: UQ+XAI vs UQ-only flag decisions on the same samples.
    # Table of (uq_flagged?, both_flagged?) disagreements.
    uq_flagged = (uq_flags != 'GREEN').astype(int)
    both_flagged = (both_flags != 'GREEN').astype(int)
    n01 = int(np.sum((uq_flagged == 0) & (both_flagged == 1)))
    n10 = int(np.sum((uq_flagged == 1) & (both_flagged == 0)))
    n00 = int(np.sum((uq_flagged == 0) & (both_flagged == 0)))
    n11 = int(np.sum((uq_flagged == 1) & (both_flagged == 1)))

    if (n01 + n10) > 0:
        table = [[n00, n01], [n10, n11]]
        # exact test when discordant counts are small
        exact = (n01 + n10) < 25
        mc_result = mcnemar(table, exact=exact)
        mc_pvalue = round(float(mc_result.pvalue), 4)
        mc_discordant = n01 + n10
    else:
        mc_pvalue = None
        mc_discordant = 0

    return {
        'uq_precision': uq_p, 'uq_recall': uq_r, 'uq_f1': uq_f,
        'xai_precision': xai_p, 'xai_recall': xai_r, 'xai_f1': xai_f,
        'both_precision': both_p, 'both_recall': both_r, 'both_f1': both_f,
        'auroc_accuracy_only': 0.5,
        'auroc_uq': auroc_uq,
        'auroc_xai': auroc_xai,
        'auroc_uq_plus_xai': auroc_both,
        'mcnemar_pvalue': mc_pvalue,
        'mcnemar_discordant': mc_discordant,
    }


def green_accuracy(flags, correct):
    green = (flags == 'GREEN')
    if np.sum(green) == 0:
        return None
    return float(np.mean(correct[green]))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join('results', 'reports', 'thresholds.json')) as f:
        thresholds = json.load(f)

    files = sorted(glob.glob(os.path.join(PER_SAMPLE_DIR, '*.csv')))
    print(f"Found {len(files)} per-sample files")

    rows = []
    paired = {}

    for filepath in files:
        filename = os.path.basename(filepath)
        meta = parse_experiment_id(filename)
        if meta is None:
            continue
        df = pd.read_csv(filepath)

        for method, unc_col in [('mc', 'mc_uncertainty'), ('de', 'de_uncertainty')]:
            key = f"{meta['dataset']}_{meta['model']}_{method}"
            if key not in thresholds:
                continue
            uq_threshold = thresholds[key]

            result = process_method(df, unc_col, uq_threshold)
            if result is None:
                continue

            rows.append({
                'dataset': meta['dataset'], 'model': meta['model'],
                'test_case': meta['test_case'], 'condition': meta['condition'],
                'uq_method': method, 'n_samples': len(df), **result,
            })

            # paired green accuracies for Wilcoxon
            uncertainty = np.nan_to_num(df[unc_col].to_numpy(), nan=0.0)
            consistency = df['consistency'].to_numpy()
            correct = df['correct'].to_numpy()
            ones = np.ones(len(consistency))
            uq_green = green_accuracy(
                assign_flags_vec(uncertainty, ones, uq_threshold, XAI_THRESHOLD), correct)
            both_green = green_accuracy(
                assign_flags_vec(uncertainty, consistency, uq_threshold, XAI_THRESHOLD), correct)
            if uq_green is not None and both_green is not None:
                paired.setdefault((meta['dataset'], method), []).append((uq_green, both_green))

    per_experiment = pd.DataFrame(rows)
    per_exp_path = os.path.join(OUTPUT_DIR, 'metrics_per_experiment.csv')
    per_experiment.to_csv(per_exp_path, index=False)
    print(f"Wrote {per_exp_path} ({len(per_experiment)} rows)")

    # Wilcoxon per dataset+method on paired (uq_only, uq+xai) GREEN accuracies
    sig_rows = []
    for (dataset, method), pairs in sorted(paired.items()):
        uq_vals = np.array([p[0] for p in pairs])
        both_vals = np.array([p[1] for p in pairs])
        diffs = both_vals - uq_vals

        if len(diffs) >= 5 and np.any(diffs != 0):
            try:
                stat, pvalue = wilcoxon(both_vals, uq_vals)
                pvalue = round(float(pvalue), 4)
            except ValueError:
                pvalue = None
        else:
            pvalue = None

        sig_rows.append({
            'dataset': dataset,
            'uq_method': method,
            'n_experiments': len(pairs),
            'mean_uq_only_green': round(float(uq_vals.mean()), 4),
            'mean_uq_plus_xai_green': round(float(both_vals.mean()), 4),
            'mean_difference': round(float(diffs.mean()), 4),
            'wilcoxon_pvalue': pvalue,
        })

    significance = pd.DataFrame(sig_rows)
    sig_path = os.path.join(OUTPUT_DIR, 'significance_summary.csv')
    significance.to_csv(sig_path, index=False)
    print(f"Wrote {sig_path} ({len(significance)} rows)")

    print("\nSignificance summary (Wilcoxon, UQ-only vs UQ+XAI GREEN accuracy):")
    print(significance.to_string(index=False))

# AUROC aggregation per dataset + method - the strongest results table.
    # AUROC is computed per experiment on 100-200 samples, so it does not
    # depend on the number of experiments the way Wilcoxon does.
    auroc_cols = ['auroc_uq', 'auroc_xai', 'auroc_uq_plus_xai']
    auroc_agg = (
        per_experiment
        .groupby(['dataset', 'uq_method'])[auroc_cols]
        .mean()
        .round(4)
        .reset_index()
    )
    # count how many experiments actually had a valid (non-null) AUROC
    auroc_agg['n_valid'] = (
        per_experiment
        .groupby(['dataset', 'uq_method'])['auroc_uq_plus_xai']
        .apply(lambda s: s.notna().sum())
        .values
    )
    auroc_agg = auroc_agg.rename(columns={
        'auroc_uq': 'mean_auroc_uq_only',
        'auroc_xai': 'mean_auroc_xai_only',
        'auroc_uq_plus_xai': 'mean_auroc_combined',
    })
    # does combining beat UQ alone, on average?
    auroc_agg['combined_minus_uq'] = (
        auroc_agg['mean_auroc_combined'] - auroc_agg['mean_auroc_uq_only']
    ).round(4)

    auroc_path = os.path.join(OUTPUT_DIR, 'auroc_by_dataset.csv')
    auroc_agg.to_csv(auroc_path, index=False)
    print(f"Wrote {auroc_path} ({len(auroc_agg)} rows)")

    print("\nMean AUROC per dataset (accuracy-only reference = 0.5):")
    print(auroc_agg.to_string(index=False))

if __name__ == '__main__':
    main()