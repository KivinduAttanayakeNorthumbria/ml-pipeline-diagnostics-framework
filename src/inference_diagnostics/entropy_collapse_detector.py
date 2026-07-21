
# Population-level prediction-entropy collapse detector.
#
# The per-prediction diagnostics (UQ and XAI) share a blind spot: when a model collapses to majority-class-only predictions under severe corruption, uncertainty DROPS (the model is confidently wrong) and consistency RISES. Both instruments go quiet exactly when the model has failed catastrophically.
#
# This detector operates at the POPULATION level instead. It measures the Shannon entropy of the distribution of predicted classes:
#
# H = - sum_c p_c * log(p_c)     where p_c = fraction of predictions in class c
#
#
# This is a structural complement to UQ/XAI: it catches the failure mode that is invisible to per-prediction signals, because the failure is a property of the prediction POPULATION, not of any single prediction.


import os
import glob
import numpy as np
import pandas as pd


PER_SAMPLE_DIR = os.path.join('results', 'per_sample')
OUTPUT_DIR = os.path.join('results', 'analysis')
COLLAPSE_RATIO = 0.5   # alarm if H < 0.5 * baseline H
KNOWN_MODELS = ['simple_cnn', 'resnet18', 'fcnn', 'rf', 'xgb']

# The clean-baseline test case per model type. Entropy of other experiments
# for the same dataset+model is compared against this.
BASELINE_TEST_CASE = {
    'rf': 'TC01',
    'xgb': 'TC02',
    'fcnn': 'TC03',
    'simple_cnn': 'TC10',
    'resnet18': 'TC11',
}


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


def prediction_entropy(y_pred):
    """Shannon entropy (natural log) of the predicted-class distribution."""
    values, counts = np.unique(y_pred, return_counts=True)
    p = counts / counts.sum()
    # p > 0 always here, so no zero-log guard needed
    return float(-np.sum(p * np.log(p)))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(PER_SAMPLE_DIR, '*.csv')))
    print(f"Found {len(files)} per-sample files")

    # First pass: compute entropy for every experiment
    records = []
    for filepath in files:
        filename = os.path.basename(filepath)
        meta = parse_experiment_id(filename)
        if meta is None:
            continue
        df = pd.read_csv(filepath)
        y_pred = df['y_pred'].to_numpy()

        H = prediction_entropy(y_pred)
        n_classes_predicted = int(len(np.unique(y_pred)))

        records.append({
            'dataset': meta['dataset'],
            'model': meta['model'],
            'test_case': meta['test_case'],
            'condition': meta['condition'],
            'entropy': round(H, 4),
            'n_classes_predicted': n_classes_predicted,
        })

    results = pd.DataFrame(records)

    # Second pass: find the baseline entropy per dataset+model, then flag collapses
    baseline_entropy = {}
    for _, row in results.iterrows():
        expected_baseline = BASELINE_TEST_CASE.get(row['model'])
        if row['test_case'] == expected_baseline:
            baseline_entropy[(row['dataset'], row['model'])] = row['entropy']

    def evaluate(row):
        key = (row['dataset'], row['model'])
        base_h = baseline_entropy.get(key)
        if base_h is None or base_h == 0:
            return pd.Series({'baseline_entropy': base_h,
                              'entropy_ratio': None,
                              'collapse_alarm': False})
        ratio = row['entropy'] / base_h
        alarm = ratio < COLLAPSE_RATIO
        return pd.Series({'baseline_entropy': round(base_h, 4),
                          'entropy_ratio': round(ratio, 4),
                          'collapse_alarm': bool(alarm)})

    results = pd.concat([results, results.apply(evaluate, axis=1)], axis=1)

    out_path = os.path.join(OUTPUT_DIR, 'entropy_collapse.csv')
    results.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(results)} rows)")

    # Report the experiments that triggered the collapse alarm
    alarms = results[results['collapse_alarm']]
    print(f"\nCollapse alarms fired: {len(alarms)}")
    if len(alarms) > 0:
        cols = ['dataset', 'model', 'test_case', 'condition',
                'entropy', 'baseline_entropy', 'entropy_ratio']
        print(alarms[cols].to_string(index=False))
    else:
        print("No collapses detected.")


if __name__ == '__main__':
    main()