# Report generation

import os
import json
import numpy as np
from datetime import datetime

# Generate full diagnostic report
def generate_report(config, dataset_name, stage1_result = None, stage2_result = None, stage3_result = None, performance_result = None):
    print("Generating report.")

    report = {'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
              'dataset': dataset_name,
              'device': config['device'],
              }
    # Stage 1 - Data quality Diagnostics
    if stage1_result is not None:
        report['stage1'] = {
            'description': 'Data quality diagnostic',
            'result': stage1_result,
        }

    # Stage 2 - Training Stability Diagnostic
    if stage2_result is not None:
        report['stage2'] = {
            'description': 'Training Stability diagnostic',
            'result': stage2_result,
        }

    # Stage 3 - Inference Diagnostic
    if stage3_result is not None:
        report['stage3'] = {
            'description': 'Inference diagnostic',
            'result': stage3_result,
        }

    # Performance metrics
    if performance_result is not None:
        report['performance'] = performance_result

    # Recommendations
    report['recommendations'] = generate_recommendation(report)
    print("Diagnostic report generated.")
    return report

# Generate recommendation based on result
def generate_recommendation(report):
    recommendations = []

    result_blocks = []
    for stage in ['stage1', 'stage2', 'stage3']:
        if stage in report and 'result' in report[stage]:
            result_blocks.append(report[stage]['result'])

    for result in result_blocks:
        report_cr = result.get('classification_report')
        accuracy = result.get('accuracy')

        # Rule 1: model collapse via macro F1
        if report_cr is not None:
            macro_f1 = report_cr.get('macro avg', {}).get('f1-score')
            if macro_f1 is not None and macro_f1 < 0.5:
                recommendations.append(
                    "Possible model collapse: macro F1 below 0.5 suggests the model "
                    "may be predicting a single class. Check class balance and training."
                )

        # Rule 2: accuracy vs macro F1 gap (imbalance masking accuracy)
        if report_cr is not None and accuracy is not None:
            macro_f1 = report_cr.get('macro avg', {}).get('f1-score')
            if macro_f1 is not None and (accuracy - macro_f1) > 0.2:
                recommendations.append(
                    "Large gap between accuracy and macro F1: accuracy is likely "
                    "inflated by class imbalance. Report F1 alongside accuracy."
                )

        # Rule 3: high rate of flagged-unreliable predictions
        for flag_key in ['flagging_mc', 'flagging_de']:
            flagging = result.get(flag_key)
            if flagging is not None:
                red = flagging.get('RED', {}).get('count', 0)
                yellow = flagging.get('YELLOW', {}).get('count', 0)
                green = flagging.get('GREEN', {}).get('count', 0)
                total = red + yellow + green
                if total > 0 and (red + yellow) / total > 0.5:
                    method = 'MC Dropout' if flag_key == 'flagging_mc' else 'Deep Ensembles'
                    recommendations.append(
                        f"Over half of predictions flagged unreliable under {method}. "
                        "Review data quality and calibration before relying on outputs."
                    )
                break

        # Rule 4: degenerate (near-zero) uncertainty
        for unc_key, method in [('mc_uncertainty', 'MC Dropout'),
                                ('de_uncertainty', 'Deep Ensembles')]:
            unc = result.get(unc_key)
            if unc is not None:
                mean_unc = unc.get('mean')
                if mean_unc is not None and mean_unc < 0.0001:
                    recommendations.append(
                        f"{method} produced near-zero uncertainty and is uninformative "
                        "for this model (no dropout layers or internal ensemble). "
                        "Rely on the alternative UQ method."
                    )

    seen = set()
    unique = []
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            unique.append(rec)

    if len(unique) == 0:
        unique.append("No major issues detected.")
    return unique

# Save report as JSON file
def save_report(report, filename, config):
    save_path = os.path.join(config['paths']['results_reports'], filename)
    final_report = convert_numpy(report)
    with open(save_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    print("Saving report.")

def convert_numpy(report):
    if isinstance(report, dict):
        new_dict = {}
        for key, value in report.items():
            new_dict[key] = convert_numpy(value)
        return new_dict
    elif isinstance(report, list):
        new_list = []
        for value in report:
            new_list.append(convert_numpy(value))
        return new_list
    elif isinstance(report, np.integer):
        return int(report)
    elif isinstance(report, np.floating):
        return float(report)
    elif isinstance(report, np.ndarray):
        return report.tolist()
    else:
        return report

