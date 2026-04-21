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

    if 'stage1' in report:
        results = report['stage1']['result']
        if 'accuracy_drop' in results:
            if results['accuracy_drop'] > 0.05:
                recommendations.append(
                    "Data quality is significantly impacting on model performance and implement data validation checks before training.",
                )

    if 'stage2' in report:
        results = report['stage2']['result']
        if 'seed_variation' in results:
            if results['seed_variation'].get('std_accuracy', 0) > 0.02:
                recommendations.append(
                    "There is a high variance across seeds and consider using ensemble methods or increasing training data."
                )

    if 'stage3' in report:
        results = report['stage3']['result']
        if 'flag_evaluation' in results:
            flag_evaluation = results['flag_evaluation']
            red_count = flag_evaluation['RED']['count']
            total = flag_evaluation['RED']['count'] + flag_evaluation['YELLOW']['count'] + flag_evaluation['GREEN']['count']
            if total > 0 and red_count / total > 0.1:
                recommendations.append(
                    "There are more than 10% of RED flagged predictions and review the model architecture and training data quality."
                )

    if len(recommendations) == 0:
        recommendations.append(
            "No major issues detected."
        )
    return recommendations

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

