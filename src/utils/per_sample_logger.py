# Per sample result logging
#
# The reports only store aggregates (means, counts, group accuracy).
# The four way ablation, the fault detection metrics and the collapse detector

import os
import numpy as np
import pandas as pd


# Save per sample diagnostic results for one experiment
def save_per_sample(config, experiment_id, y_true, y_pred, mc_uncertainty=None,
                    de_uncertainty=None, consistency=None):
    save_folder = config['paths']['results_per_sample']
    os.makedirs(save_folder, exist_ok=True)

    lengths = [len(y_true), len(y_pred)]
    if mc_uncertainty is not None:
        lengths.append(len(mc_uncertainty))
    if de_uncertainty is not None:
        lengths.append(len(de_uncertainty))
    if consistency is not None:
        lengths.append(len(consistency))

    n = min(lengths)

    y_true_cut = np.array(y_true)[:n]
    y_pred_cut = np.array(y_pred)[:n]
    correct = (y_pred_cut == y_true_cut).astype(int)

    data = {
        'sample_id': np.arange(n),
        'y_true': y_true_cut,
        'y_pred': y_pred_cut,
        'correct': correct,
    }

    if mc_uncertainty is not None:
        data['mc_uncertainty'] = np.array(mc_uncertainty)[:n]
    else:
        data['mc_uncertainty'] = np.full(n, np.nan)

    if de_uncertainty is not None:
        data['de_uncertainty'] = np.array(de_uncertainty)[:n]
    else:
        data['de_uncertainty'] = np.full(n, np.nan)

    if consistency is not None:
        data['consistency'] = np.array(consistency)[:n]
    else:
        data['consistency'] = np.full(n, np.nan)

    df = pd.DataFrame(data)

    save_path = os.path.join(save_folder, f"{experiment_id}.csv")
    df.to_csv(save_path, index=False)

    print(f"Per sample results saved: {save_path} ({n} rows)")
    return df


# Build a consistent experiment id used for both the CSV and the report file
def build_experiment_id(dataset_short_name, model_name, test_case, condition=None):
    parts = [dataset_short_name, model_name, test_case]
    if condition is not None:
        parts.append(str(condition))
    return "_".join(parts)