# Flagging system RED, YELLOW, GREEN

import numpy as np

# Assign flags to each prediction based on uncertainty
# RED - Uncertainty very high (do not trust)
# YELLOW - Either one flags (review needed)
# GREEN - Both pass (likely trustworthy)
def assign_flags(uncertainty, explanation_consistency, config):
    thresholds = config['stage3_inference']['flagging']
    uq_threshold = thresholds['uncertainty_threshold']
    xai_threshold = thresholds['explanation_consistency_threshold']

    flags = []

    for uq in range(len(uncertainty)):
        high_uncertainty = uncertainty[uq] > uq_threshold
        low_consistency = explanation_consistency[uq] < xai_threshold

        if high_uncertainty and low_consistency:
            flags.append('RED')
        elif high_uncertainty or low_consistency:
            flags.append('YELLOW')
        else:
            flags.append('GREEN')

    flags = np.array(flags)
    return flags

# Evaluate flagging system
def evaluate_flags(flags, y_pred, y_true):
    correct = (y_pred == y_true)
    results = {}

    for flag_type in  ['RED', 'YELLOW', 'GREEN']:
        flg = (flags == flag_type)
        count = np.sum(flg)

        if count > 0:
            accuracy_in_group = np.mean(correct[flg])
        else:
            accuracy_in_group = 0

        results[flag_type] = {
            'count': int(count),
            'accuracy': round(float(accuracy_in_group), 4)
        }

        print(f"{flag_type}: Count: {count} Accuracy:{accuracy_in_group * 100:.1f}%")

    # GREEN should have high accuracy and RED should have low accuracy
    if results['GREEN']['accuracy'] > results['RED']['accuracy']:
        print("Flagging system is working")
    else:
        print("Flagging system is not working, needs threshold adjustment")

    return results
