import numpy as np
import torch
from torch.utils.data import DataLoader

# MC Dropout (run the model multiple time with dropout switch on)

def mc_dropout_prediction_tab(model, X_test, config):
    print("MC Dropout started.")

    params = config["stage3_inference"]["uncertainty"]["mc_dropout"]
    num_passes = params["num_forward_passes"]
    device = torch.device(config['device'])

    # Model train with dropout layers
    model.train()

    # Collect predictions for 50 passes
    all_predictions = []

    # Diable backpropagation ( no training)
    with torch.no_grad():
        for pass_num in range(num_passes):
            # Convert to pytorch tensor
            X_tensor = torch.FloatTensor(X_test.values).to(device)
            # Feed the dataset
            outputs = model(X_tensor)
            # Convert score to probabilities
            pass_predictions = torch.softmax(outputs, dim = 1).cpu().numpy()

            all_predictions.append(pass_predictions)
            if (pass_num + 1) % 10 == 0:
                print(f"Pass {pass_num + 1}/{num_passes} done.")

    # Stack all passes
    all_predictions = np.array(all_predictions)

    # Get mean of the sample
    mean_probs = all_predictions.mean(axis = 0)
    # Calculate standard deviation of prediction probabilities and the calculate mean of std to ge one uncertainty value
    uncertainty = all_predictions.std(axis = 0).mean(axis = 1)

    print("MC Dropout finished for tabular.")
    return mean_probs, uncertainty

def mc_dropout_prediction_img(model, X_test, config):
    print("MC Dropout started.")

    params = config["stage3_inference"]["uncertainty"]["mc_dropout"]
    num_passes = params["num_forward_passes"]
    device = torch.device(config['device'])

    # Model train
    model.train()

    # Collect predictions
    all_predictions = []

    # Same as tabular process
    with torch.no_grad():
        for pass_num in range(num_passes):
            # Dataloader used for images
            loader = DataLoader(X_test, batch_size=64, shuffle=False)
            pass_predictions = []
            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(device)
                outputs = model(batch_images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                pass_predictions.append(probs)

            pass_predictions = np.concatenate(pass_predictions, axis=0)
            all_predictions.append(pass_predictions)
            if (pass_num + 1) % 10 == 0:
                print(f"Pass {pass_num + 1}/{num_passes} done.")

    # Stack all passes
    all_predictions = np.array(all_predictions)

    # Get mean and std across passes
    mean_probs = all_predictions.mean(axis = 0)
    uncertainty = all_predictions.std(axis = 0).mean(axis = 1)

    print("MC Dropout finished for images.")
    return mean_probs, uncertainty

# Deep Ensembles (train multiple models with different seeds)

# Deep Ensemble for sklearn models (RF and XGBoost)
def deep_ensemble_prediction_sklern(train_function, X_train, y_train, X_test, config):
    print("Deep Ensemble started for tabular and sklern.")
    params = config["stage3_inference"]["uncertainty"]["deep_ensembles"]
    num_models = params["num_models"]
    ensemble_seeds = params["ensemble_seeds"]

    # Collect prediction for 5 models
    all_predictions = []

    for seed in ensemble_seeds[:num_models]:
        print(f"Training model with seed {seed}")
        # Create random weights for every seed
        np.random.seed(seed)
        # Train a new model from scratch
        model = train_function(X_train, y_train, config)
        predictions = model.predict_proba(X_test)
        all_predictions.append(predictions)

    all_predictions = np.array(all_predictions)

    # Get mean and std across passes
    mean_probs = all_predictions.mean(axis=0)
    # Calculate standard deviation of prediction probabilities and the calculate mean of std to ge one uncertainty value
    uncertainty = all_predictions.std(axis=0).mean(axis=1)

    print("Deep Ensemble finished for tabular sklern.")
    return mean_probs, uncertainty

def deep_ensemble_prediction_tab(train_function, X_train, y_train, X_test, config):
    print("Deep Ensemble started for tabular.")
    params = config["stage3_inference"]["uncertainty"]["deep_ensembles"]
    num_models = params["num_models"]
    ensemble_seeds = params["ensemble_seeds"]
    device = torch.device(config['device'])

    # Collect prediction for 5 models
    all_predictions = []

    for seed in ensemble_seeds[:num_models]:
        print(f"Training model with seed {seed}")

        # Set seed
        torch.manual_seed(seed)
        # Create random weights for every seed
        np.random.seed(seed)

        # Train a new model from scratch
        model = train_function(X_train, y_train, config)

        # Make predictions
        model.eval()
        with torch.no_grad():
            # Convert to pytorch tensor
            X_tensor = torch.FloatTensor(X_test.values).to(device)
            # Feed the dataset
            outputs = model(X_tensor)
            # Convert score to probabilities
            predictions = torch.softmax(outputs, dim=1).cpu().numpy()

        all_predictions.append(predictions)

    # Stack all predictions
    all_predictions = np.array(all_predictions)

    # Get mean and std across passes
    mean_probs = all_predictions.mean(axis=0)
    # Calculate standard deviation of prediction probabilities and the calculate mean of std to ge one uncertainty value
    uncertainty = all_predictions.std(axis=0).mean(axis=1)

    print("Deep Ensemble finished for tabular.")
    return mean_probs, uncertainty

# Same as tabular
def deep_ensemble_prediction_img(train_function, config, test_set, train_set):
    print("Deep Ensemble started for images.")
    params = config["stage3_inference"]["uncertainty"]["deep_ensembles"]
    num_models = params["num_models"]
    ensemble_seeds = params["ensemble_seeds"]
    device = torch.device(config['device'])

    all_predictions = []

    for seed in ensemble_seeds[:num_models]:
        print(f"Training model with seed {seed}")

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train models
        model = train_function(train_set=train_set, config=config)

        # Make predictions
        model.eval()
        with torch.no_grad():
            loader = DataLoader(test_set, batch_size=64, shuffle=False)
            predictions = []
            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(device)
                outputs = model(batch_images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                predictions.append(probs)
            predictions = np.concatenate(predictions, axis=0)

        all_predictions.append(predictions)

    # Stack all predictions
    all_predictions = np.array(all_predictions)

    # Get mean and std across passes
    mean_probs = all_predictions.mean(axis=0)
    uncertainty = all_predictions.std(axis=0).mean(axis=1)

    print("Deep Ensemble finished for images.")
    return mean_probs, uncertainty






