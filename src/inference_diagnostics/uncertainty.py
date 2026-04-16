import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# MC Dropout (run the model multiple time with dropout switch on)
def mc_dropout_prediction(model, X_test, config, is_image = False):
    print("MC Dropout started.")

    params = config["stage3_inference"]["uncertainty"]["mc_dropout"]
    num_passes = params["num_forward_passes"]
    device = torch.device(config['device'])

    # Modal train
    model.train()

    # Collect predictions
    all_predictions = []

    with torch.no_grad():
        for pass_num in range(num_passes):
            # Dataloader used for images
            if is_image:
                loader = DataLoader(X_test, batch_size = 64, shuffle = False)
                pass_preds = []
                for batch_images, batch_labels in loader:
                    batch_images = batch_images.to(device)
                    outputs = model(batch_images)
                    probs = torch.softmax(outputs, dim = 1).cpu().numpy()
                    pass_preds.append(probs)
                pass_preds = np.concatenate(pass_preds, axis = 0)
            # DataFrame used for tabular
            else:
                X_tensor = torch.FloatTensor(X_test.values).to(device)
                outputs = model(X_tensor)
                pass_preds = torch.softmax(outputs, dim = 1).cpu().numpy()

            all_predictions.append(pass_preds)

            if (pass_num + 1) % 10 == 0:
                print(f"Pass {pass_num + 1}/{num_passes} done.")

    # Stack all passes
    all_predictions = np.array(all_predictions)

    # Get mean and std across passes
    mean_probs = all_predictions.mean(axis = 0)
    uncertainty = all_predictions.std(axis = 0).mean(axis = 1)

    print("MC Dropout finished.")
    return mean_probs, uncertainty

# Deep Ensembles (train multiple models with different seeds)
def deep_ensemble_prediction(train_function, X_train, y_train, X_test, config, is_image = False, test_set = None):
    print("Deep Ensemble started.")
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
        if is_image:
            model = train_function(train_set = X_train, config= config)
        else:
            model = train_function(X_train, y_train, config)

        # Make predictions
        model.eval()
        with torch.no_grad():
            if is_image:
                loader = DataLoader(test_set, batch_size = 64, shuffle = False)
                preds = []
                for batch_images, batch_labels in loader:
                    batch_images = batch_images.to(device)
                    outputs = model(batch_images)
                    probs = torch.softmax(outputs, dim = 1).cpu().numpy()
                    preds.append(probs)
                preds = np.concatenate(preds, axis = 0)
            else:
                X_tensor = torch.FloatTensor(X_test.values).to(device)
                outputs = model(X_tensor)
                preds = torch.softmax(outputs, dim = 1).cpu().numpy()

        all_predictions.append(preds)

    # Stack all predictions
    all_predictions = np.array(all_predictions)

    # Get mean and std across passes
    mean_probs = all_predictions.mean(axis=0)
    uncertainty = all_predictions.std(axis=0).mean(axis=1)

    print("Deep Ensemble finished.")
    return mean_probs, uncertainty








