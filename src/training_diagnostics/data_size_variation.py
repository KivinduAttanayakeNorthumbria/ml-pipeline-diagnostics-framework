# Train the same model on different dataset sizes

import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.performance_tracker import PerformanceTracker
from torch.utils.data import Subset

def run_data_size_variation_tab(train_function, evaluate_function, X_train, y_train, X_test, y_test, config, is_pytorch = False):
    print("Stage 2: Dataset size variation experiment for tabular")

    fractions = config['stage2_training_stability']['data_size_variation']['fractions']
    seed = config['random_seeds']['primary_seed']
    tracker = PerformanceTracker()

    all_results = []

    for fraction in fractions:
        if fraction < 1.0:
            # Use stratify to balance both classes
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
            # For FCNN
            accuracy, y_prediction = evaluate_function(model, X_test, y_test, config)
        else:
            # For RF and XGBoost
            accuracy, y_prediction = evaluate_function(model, X_test, y_test)

        tracker.stop_performance_track(f"Evaluating completed for fraction: {fraction}")
        all_results.append({
            'fraction': fraction,
            'num_samples': len(X_subset),
            'accuracy': round(float(accuracy), 4),
            'predictions': y_prediction
        })

    # Summary
    results = {
        'fractions': all_results,
        'performance': tracker.get_results()
    }

    for rs in all_results:
        print(f"Fraction: {rs['fraction']}, Accuracy: {rs['accuracy']}, Samples: {rs['num_samples']}")

    print("Stage 2: Dataset size Variation experiment for tabular completed.")
    return results

def run_data_size_variation_img(train_function, evaluate_function, train_set, test_set, config):
    print("Stage 2: Dataset size variation experiment for image")

    fractions = config['stage2_training_stability']['data_size_variation']['fractions']
    seed = config['random_seeds']['primary_seed']
    tracker = PerformanceTracker()

    all_results = []
    total_train_samples = len(train_set)

    for fraction in fractions:
        # Calculate subset size from total train size
        subset_size = int(total_train_samples * fraction)
        # Create subset
        gen = np.random.RandomState(seed)
        indices = gen.choice(total_train_samples, subset_size, replace=False)
        train_subset = Subset(train_set, indices)
        print(f"Samples: {len(train_subset)} from: {total_train_samples}")

        # Train model
        tracker.start_performance_track()
        model = train_function(train_subset, config)
        tracker.stop_performance_track(f"Training completed for fraction: {fraction}")

        # Evaluate
        tracker.start_performance_track()
        accuracy, y_prediction = evaluate_function(model, test_set, config)
        tracker.stop_performance_track(f"Evaluation completed for fraction: {fraction}")

        all_results.append({
            'fraction': fraction,
            'num_samples': len(train_subset),
            'accuracy': round(float(accuracy), 4),
            'predictions': y_prediction
        }
        )
    results = {
        'fractions': all_results,
        'performance': tracker.get_results()
    }
    for rs in all_results:
        print(f"Fraction: {rs['fraction']}, Accuracy: {rs['accuracy']}, Samples: {rs['num_samples']}")

    print("Stage 2: Dataset size Variation experiment for image completed.")
    return results




