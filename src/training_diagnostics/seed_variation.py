# Train the same model for 20 different seeds

import torch
import numpy as np
from src.utils.performance_tracker import PerformanceTracker

def run_seed_variation(train_function, evaluate_function, X_train, y_train, X_test, y_test, config, is_pytorch = False):
    print("Stage 2: Seed Variation experiment")

    seeds = config['random_seeds']['seed_variation_list']
    num_seeds = config['stage2_training_stability']['seed_variation']['num_seeds']
    tracker = PerformanceTracker()

    all_accuracies = []
    all_predictions = []

    for seed in seeds[:num_seeds]:
        np.random.seed(seed)

        if is_pytorch:
            torch.manual_seed(seed)

            tracker.start_performance_track()
            model = train_function(X_train, y_train, config)
            tracker.stop_performance_track(f"Training completed for seed: {seed}")

            tracker.start_performance_track()
            accuracy, y_pred = evaluate_function(model, X_test, y_test, config)
            tracker.stop_performance_track(f"Evaluating completed for seed: {seed}")

        else:
            tracker.start_performance_track()
            model = train_function(X_train, y_train, config)
            tracker.stop_performance_track(f"Training completed for seed: {seed}")

            tracker.start_performance_track()
            accuracy, y_pred = evaluate_function(model, X_test, y_test)
            tracker.stop_performance_track(f"Evaluating completed for seed: {seed}")

        all_accuracies.append(accuracy)
        all_predictions.append(y_pred)

    # Calculate stability metrics
    accuracies = np.array(all_accuracies)
    results = {
        'accuracy': accuracies,
        'mean_accuracy': round(float(accuracies.mean()), 4),
        'std_accuracy': round(float(accuracies.std()), 4),
        'min_accuracy': round(float(accuracies.min()), 4),
        'max_accuracy': round(float(accuracies.max()), 4),
        'predictions': all_predictions,
        'performance': tracker.get_results()
    }

    if results['std_accuracy'] > 0.02:
        print("High variance, model unstable")
    else:
        print("Low variance, model stable")

    print("Stage 2: Seed Variation experiment completed.")
    return results