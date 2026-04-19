# Train the same model with different hyperparameters

import torch
import numpy as np
from src.utils.performance_tracker import PerformanceTracker

def run_hyperparam_variation(train_function, evaluate_function, X_train, y_train, X_test, y_test, config, model_name):
    print("Stage 2: Hyperparameters Variation experiment for tabular ")
    hp_config = config['stage2_training_stability']['hyperparameter_variation']
    learning_rates = hp_config['learning_rates']
    batch_sizes = hp_config['batch_sizes']
    tracker = PerformanceTracker()

    all_results = []

    for lr in learning_rates:
        for bs in batch_sizes:
            # Temporarily change config file
            original_lr = config['models'][model_name]['params']['learning_rate']
            original_bs = config['models'][model_name]['params']['batch_size']
            config['models'][model_name]['params']['learning_rate'] = lr
            config['models'][model_name]['params']['batch_size'] = bs

            # Train
            tracker.start_performance_track()
            model = train_function(X_train, y_train, config)
            tracker.stop_performance_track(f"Training stopped LR: {lr}, BS: {bs}")

            # Evaluate
            tracker.start_performance_track()
            accuracy, y_prediction = evaluate_function(model, X_test, y_test, config)
            tracker.stop_performance_track(f"Evaluating stopped LR: {lr}, BS: {bs}")

            all_results.append({
                'learning_rate': lr,
                'batch_size': bs,
                'accuracy': round(float(accuracy), 4),
                'predictions': y_prediction
            })

            # Restore config file
            config['models'][model_name]['params']['learning_rate'] = original_lr
            config['models'][model_name]['params']['batch_size'] = original_bs

    # Summary
    results = {
        'hp_results': all_results,
        'performance': tracker.get_results()
    }

    for rs in all_results:
        print(f"LR: {rs['learning_rate']}, BS: {rs['batch_size']}, Accuracy: {rs['accuracy']}")

    print("Stage 2: Hyperparameter Variation experiment for tabular completed.")
    return results

def run_hyperparam_variation_img(train_function, evaluate_function,train_set, test_set, config, model_name):
    print("Stage 2: Hyperparameters Variation experiment for images ")
    hp_config = config['stage2_training_stability']['hyperparameter_variation']
    learning_rates = hp_config['learning_rates_images']
    batch_sizes = hp_config['batch_sizes_images']
    tracker = PerformanceTracker()

    all_results = []

    for lr in learning_rates:
        for bs in batch_sizes:
            # Temporarily change config file
            original_lr = config['models'][model_name]['params']['learning_rate']
            original_bs = config['models'][model_name]['params']['batch_size']
            config['models'][model_name]['params']['learning_rate'] = lr
            config['models'][model_name]['params']['batch_size'] = bs

            # Train
            tracker.start_performance_track()
            model = train_function(train_set, config)
            tracker.stop_performance_track(f"Training stopped LR: {lr}, BS: {bs}")

            # Evaluate
            tracker.start_performance_track()
            accuracy, y_prediction = evaluate_function(model,test_set, config)
            tracker.stop_performance_track(f"Evaluating stopped LR: {lr}, BS: {bs}")

            all_results.append({
                'learning_rate': lr,
                'batch_size': bs,
                'accuracy': round(float(accuracy), 4),
                'predictions': y_prediction
            })

            # Restore config file
            config['models'][model_name]['params']['learning_rate'] = original_lr
            config['models'][model_name]['params']['batch_size'] = original_bs

    # Summary
    results = {
        'hp_results': all_results,
        'performance': tracker.get_results()
    }

    for rs in all_results:
        print(f"LR: {rs['learning_rate']}, BS: {rs['batch_size']}, Accuracy: {rs['accuracy']}")

    print("Stage 2: Hyperparameter Variation experiment for images completed.")
    return results