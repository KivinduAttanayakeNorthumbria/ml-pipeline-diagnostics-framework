# Train the same model on different dataset sizes

import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.performance_tracker import PerformanceTracker

def run_data_size_variation(train_function, evaluate_function, X_train, y_train, X_test, y_test, config, is_pytorch = False):
    print("Stage 2: Dataset size Variation experiment")

    fractions = config['stage2_training_stability']['data_size_variation']['fractions']
    seed = config['random_seeds']['primary_seed']
    tracker = PerformanceTracker()

    all_results = []

    for fraction in fractions:
        if fraction < 1.0:
            X_subset, X_unused, y_subset, y_unused = train_test_split(X_train, y_train, train_size=fraction, random_state=seed, stratify=y_train)
        else:
            X_subset = X_train
            y_subset = y_train

        print(f"Samples: {len(X_subset)} from: {len(X_train)}")

        # Train model
        tracker.start_performance_track()
        model = train_function(X_subset, y_subset, config)
        tracker.stop_performance_track(f"Training completed for fraction: {fraction}")

        # Evaluate
        tracker.start_performance_track()
        if is_pytorch:
            accuracy, y_pred = evaluate_function(model, X_test, y_test, config)
        else:
            accuracy, y_pred = evaluate_function(model, X_test, y_test)

        tracker.stop_performance_track(f"Evaluating completed for fraction: {fraction}")
        all_results.append({
            'fraction': fraction,
            'num_samples': len(X_subset),
            'accuracy': round(float(accuracy), 4),
            'predictions': y_pred
        })

    # Summary
    results = {
        'fractions': all_results,
        'performance': tracker.get_results()
    }

    for rs in all_results:
        print(f"Fraction: {rs['fraction']}, Accuracy: {rs['accuracy']}, Samples: {rs['num_samples']}")

    print("Stage 2: Dataset size Variation experiment completed.")
    return results

