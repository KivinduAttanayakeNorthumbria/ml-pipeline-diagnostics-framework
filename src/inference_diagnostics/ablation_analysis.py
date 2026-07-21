
# Four-way ablation analysis.
#
# Reads every per-sample CSV in results/per_sample/ and, for each experiment,
# recomputes the flagging outcome under four ablation conditions:
#
# accuracy_only : no flagging. Reference = raw accuracy of all reviewed samples.
# uq_only       : uncertainty real, consistency forced to 1.0 (never fails XAI)
# xai_only      : uncertainty forced to 0.0 (never fails UQ), consistency real
# uq_plus_xai   : both real
#
# For each condition it reports the GREEN-group accuracy - the accuracy of the
# predictions the system labelled trustworthy. The research question is answered
# by comparing GREEN accuracy across the four conditions.


import os
import glob
import numpy as np
import pandas as pd


PER_SAMPLE_DIR = os.path.join('results', 'per_sample')
OUTPUT_DIR = os.path.join('results', 'analysis')
XAI_THRESHOLD = 0.5

# Known models, longest first so 'simple_cnn' matches before a naive split
KNOWN_MODELS = ['simple_cnn', 'resnet18', 'fcnn', 'rf', 'xgb']


def parse_experiment_id(filename):
    """
    Split '{dataset}_{model}_{TCxx}[_{condition}].csv' into parts.
    Model names contain underscores, so match against KNOWN_MODELS rather
    than splitting blindly.
    """
    stem = filename[:-4] if filename.endswith('.csv') else filename

    model_found = None
    for model in KNOWN_MODELS:
        token = f"_{model}_"
        if token in stem:
            model_found = model
            dataset = stem.split(token)[0]
            remainder = stem.split(token)[1]
            break

    if model_found is None:
        return None

    # remainder is 'TCxx' or 'TCxx_condition'
    parts = remainder.split('_', 1)
    test_case = parts[0]
    condition = parts[1] if len(parts) > 1 else 'baseline'

    return {
        'dataset': dataset,
        'model': model_found,
        'test_case': test_case,
        'condition': condition,
    }


def assign_flags_vec(uncertainty, consistency, uq_threshold, xai_threshold):
    """Vectorised RED/YELLOW/GREEN assignment. Same logic as the framework."""
    high_uncertainty = uncertainty > uq_threshold
    low_consistency = consistency < xai_threshold

    flags = np.full(len(uncertainty), 'GREEN', dtype=object)
    flags[high_uncertainty | low_consistency] = 'YELLOW'
    flags[high_uncertainty & low_consistency] = 'RED'
    return flags


def green_accuracy(flags, correct):
    """Accuracy within the GREEN group. None if the group is empty."""
    green = (flags == 'GREEN')
    n = int(np.sum(green))
    if n == 0:
        return None, 0
    acc = float(np.mean(correct[green]))
    return round(acc, 4), n


def run_ablation_for_method(df, uncertainty_col, uq_threshold):
    """
    Run the four conditions for one UQ method on one experiment.
    Returns a dict of results, or None if this method is unusable
    (all-NaN uncertainty, e.g. MC on tree models).
    """
    uncertainty = df[uncertainty_col].to_numpy()
    consistency = df['consistency'].to_numpy()
    correct = df['correct'].to_numpy()

    # Skip if the method produced no real uncertainty (RF/XGB under MC)
    if np.all(np.isnan(uncertainty)):
        return None

    # Guard: if any NaN slipped through, treat as zero uncertainty
    uncertainty = np.nan_to_num(uncertainty, nan=0.0)

    ones = np.ones(len(consistency))
    zeros = np.zeros(len(uncertainty))

    # accuracy_only: no flagging, everything is 'trusted' -> raw accuracy
    acc_only = round(float(np.mean(correct)), 4)

    # uq_only: real uncertainty, consistency always passes
    uq_flags = assign_flags_vec(uncertainty, ones, uq_threshold, XAI_THRESHOLD)
    uq_green, uq_n = green_accuracy(uq_flags, correct)

    # xai_only: uncertainty always passes, real consistency
    xai_flags = assign_flags_vec(zeros, consistency, uq_threshold, XAI_THRESHOLD)
    xai_green, xai_n = green_accuracy(xai_flags, correct)

    # uq_plus_xai: both real
    both_flags = assign_flags_vec(uncertainty, consistency, uq_threshold, XAI_THRESHOLD)
    both_green, both_n = green_accuracy(both_flags, correct)

    return {
        'accuracy_only': acc_only,
        'uq_only_green_acc': uq_green,
        'uq_only_green_n': uq_n,
        'xai_only_green_acc': xai_green,
        'xai_only_green_n': xai_n,
        'uq_plus_xai_green_acc': both_green,
        'uq_plus_xai_green_n': both_n,
        # the headline: does combining beat UQ alone?
        'xai_uplift_vs_uq': (round(both_green - uq_green, 4)
                             if (both_green is not None and uq_green is not None)
                             else None),
    }


def load_thresholds():
    """Load per-model-per-method thresholds written by the baseline notebooks."""
    import json
    path = os.path.join('results', 'reports', 'thresholds.json')
    with open(path, 'r') as f:
        return json.load(f)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    thresholds = load_thresholds()

    files = sorted(glob.glob(os.path.join(PER_SAMPLE_DIR, '*.csv')))
    print(f"Found {len(files)} per-sample files")

    rows = []

    for filepath in files:
        filename = os.path.basename(filepath)
        meta = parse_experiment_id(filename)
        if meta is None:
            print(f"  SKIP (unparseable): {filename}")
            continue

        df = pd.read_csv(filepath)

        for method, unc_col in [('mc', 'mc_uncertainty'), ('de', 'de_uncertainty')]:
            # threshold key: {dataset}_{model}_{method}
            key = f"{meta['dataset']}_{meta['model']}_{method}"
            if key not in thresholds:
                # e.g. mc threshold never written for tree models
                continue
            uq_threshold = thresholds[key]

            result = run_ablation_for_method(df, unc_col, uq_threshold)
            if result is None:
                continue

            row = {
                'dataset': meta['dataset'],
                'model': meta['model'],
                'test_case': meta['test_case'],
                'condition': meta['condition'],
                'uq_method': method,
                'n_samples': len(df),
                **result,
            }
            rows.append(row)

    per_experiment = pd.DataFrame(rows)
    per_exp_path = os.path.join(OUTPUT_DIR, 'ablation_per_experiment.csv')
    per_experiment.to_csv(per_exp_path, index=False)
    print(f"Wrote {per_exp_path} ({len(per_experiment)} rows)")

    # Aggregate per dataset + method: mean GREEN accuracy per condition
    agg = per_experiment.groupby(['dataset', 'uq_method']).agg(
        mean_accuracy_only=('accuracy_only', 'mean'),
        mean_uq_only=('uq_only_green_acc', 'mean'),
        mean_xai_only=('xai_only_green_acc', 'mean'),
        mean_uq_plus_xai=('uq_plus_xai_green_acc', 'mean'),
        mean_xai_uplift=('xai_uplift_vs_uq', 'mean'),
        n_experiments=('test_case', 'count'),
    ).round(4).reset_index()

    by_dataset_path = os.path.join(OUTPUT_DIR, 'ablation_by_dataset.csv')
    agg.to_csv(by_dataset_path, index=False)
    print(f"Wrote {by_dataset_path} ({len(agg)} rows)")

    print("\nPer-dataset summary:")
    print(agg.to_string(index=False))


if __name__ == '__main__':
    main()