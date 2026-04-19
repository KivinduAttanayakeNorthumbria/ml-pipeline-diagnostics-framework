# Train the same model for 20 different seeds

import torch
import numpy as np
from src.utils.performance_tracker import PerformanceTracker

def run_seed_variation_tab(train_function, evaluate_function, X_train, y_train, X_test, y_test, config, is_pytorch = False):
    print("Stage 2: Seed variation experiment for tabular")

    seeds = config['random_seeds']['seed_variation_list']
    num_seeds = config['stage2_training_stability']['seed_variation']['num_seeds']
    tracker = PerformanceTracker()

    all_accuracies = []
    all_predictions = []

    for seed in seeds[:num_seeds]:
        np.random.seed(seed)

        if is_pytorch:
            torch.manual_seed(seed)

            # Train model
            tracker.start_performance_track()
            model = train_function(X_train, y_train, config)
            tracker.stop_performance_track(f"Training completed for seed: {seed}")

            # Evaluation
            tracker.start_performance_track()
            accuracy, y_prediction = evaluate_function(model, X_test, y_test, config)
            tracker.stop_performance_track(f"Evaluating completed for seed: {seed}")

        else:
            # Train model
            tracker.start_performance_track()
            model = train_function(X_train, y_train, config)
            tracker.stop_performance_track(f"Training completed for seed: {seed}")

            # Evaluation
            tracker.start_performance_track()
            accuracy, y_prediction = evaluate_function(model, X_test, y_test)
            tracker.stop_performance_track(f"Evaluating completed for seed: {seed}")

        all_accuracies.append(accuracy)
        all_predictions.append(y_prediction)

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

    print("Stage 2: Seed variation experiment for tabular completed.")
    return results

def run_seed_variation_img(train_function, evaluate_function, train_set, test_set, config):
    print("Stage 2: Seed variation experiment for images")

    seeds = config['random_seeds']['image_seed_list']
    num_seeds = len(seeds)
    tracker = PerformanceTracker()

    all_accuracies = []
    all_predictions = []

    for seed in seeds[:num_seeds]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train model
        tracker.start_performance_track()
        model = train_function(train_set, config)
        tracker.stop_performance_track(f"Training completed for seed: {seed}")

        # Evaluation
        tracker.start_performance_track()
        accuracy, y_prediction = evaluate_function(model, test_set, config)
        tracker.stop_performance_track(f"Evaluating completed for seed: {seed}")

        all_accuracies.append(accuracy)
        all_predictions.append(y_prediction)

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

    print("Stage 2: Seed variation experiment for images completed.")
    return results
