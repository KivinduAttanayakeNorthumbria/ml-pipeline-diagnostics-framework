import numpy as np
import torch
import torch.nn as nn
import shap
import lime
import lime.lime_tabular
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# SHAP for tabular data
def shap_tab(model, X_train, X_test, config, is_pytorch = False):
    print("SHAP started.")

    params = config['stage3_inference']['explainability']['shap']
    num_background = params['num_background_samples']
    max_samples = params['max_samples_to_explain']

    # Sample background data to understand normal feature value (baseline to compare)
    background = shap.sample(X_train, num_background)

    # Sample test data to explain
    samples_explained = X_test.iloc[:max_samples]

    # Select the explainer
    if is_pytorch:
        device = torch.device(config['device'])
        # Evaluation the model without dropout
        model.eval()

        def predict_function(sample):
            # Transform to tensor
            x_tensor = torch.FloatTensor(sample).to(device)
            # Diable backpropagation ( no training )
            with torch.no_grad():
                outputs = model(x_tensor)
                # Convert score to probabilities
                probabilities = torch.softmax(outputs, dim = 1).cpu().numpy()
            return probabilities
        # KernelExplainer works with any model
        explainer = shap.KernelExplainer(predict_function, background)
    else:
        try:
            # Try TreeExplainer for tree models to save time
            explainer = shap.TreeExplainer(model)
        except Exception:
            # If the tree explainer fail, wrap predict_proba to avoid compatibility issues
            def predict_function(sample):
                return model.predict_proba(sample)
            explainer = shap.KernelExplainer(predict_function, background)

    # Calculate SHAP values
    shap_values = explainer.shap_values(samples_explained)

    print(f"SHAP finished. Explained: {len(samples_explained)} samples.")
    return shap_values, samples_explained

# LIME for tabular data
def lime_tab(model, X_train, X_test, config, is_pytorch = False):
    print("LIME started.")

    params = config['stage3_inference']['explainability']['lime']
    num_features = params['num_features']
    num_samples = params['num_samples']
    max_samples = params['max_samples_to_explain']

    # Create LIME explainer
    feature_names = list(X_train.columns)
    # Use range in feature
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data = X_train.values,
        feature_names = feature_names,
        mode = 'classification',
    )

    # Sample test data
    samples_explained = X_test.iloc[:max_samples]

    # Create prediction function
    if is_pytorch:
        device = torch.device(config['device'])
        # Evaluation model without dropout
        model.eval()

        def predict_function(sample):
            x_tensor = torch.FloatTensor(sample).to(device)
            # Diable backpropagation ( no training )
            with torch.no_grad():
                outputs = model(x_tensor)
                # Convert score to probabilities
                probabilities = torch.softmax(outputs, dim = 1).cpu().numpy()
            return probabilities
    else:
        # Different function for tree models
        predict_function = model.predict_proba

    # Explain each sample
    all_explanations = []
    # Explain one sample at a time
    for ex in range(len(samples_explained)):
        # Create 500 slightly different copies and use simple leaner regression
        explanation = explainer.explain_instance(
            data_row = samples_explained.iloc[ex].values,
            predict_fn = predict_function,
            num_features  = num_features,
            num_samples = num_samples,
        )
        all_explanations.append(explanation)

        if (ex + 1) % 50 == 0:
            print(f"Explained {ex + 1} / {len(samples_explained)} samples.")

    print(f"LIME finished. Explained: {len(all_explanations)} samples.")
    return all_explanations, samples_explained

# Grad-CAM for image model
def gradcam_img(model, test_set, config):
    print("GradCam started.")

    params = config['stage3_inference']['explainability']['gradcam']
    max_samples = params['max_samples_to_explain']
    device = torch.device(config['device'])

    # Evaluation the model without dropout
    model.eval()

    # Find the lat convolutional layer
    target_layer = None
    # Check the layers reverse to find the last Conv2d layer
    # Last layer contains meaningful features
    # for module in reversed(list(model.modules())):
    #     if isinstance(module, nn.Conv2d):
    #         target_layer = module
    #         break
    if hasattr(model, 'layer3'):
        print("*****************************************************************************************")
        target_layer = model.layer3[-1]
    else:
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                break

    # Verify there is a Conv2d layer
    if target_layer is None:
        raise ValueError("Target layer not found.")

    # Create Grad-CAM object to watch at last target layer
    cam = GradCAM(model, target_layers = [target_layer] )

    # Get samples to explain
    loader = DataLoader(test_set, batch_size = 1, shuffle = False)

    # Empty array to collect heatmaps
    all_heatmaps = []
    count = 0

    for batch_images, batch_labels in loader:
        if count >= max_samples:
            break

        batch_images = batch_images.to(device)

        # Get model prediction
        with torch.no_grad():
            output = model(batch_images)
            # Select the highest class
            predicted_class = output.argmax(dim = 1).item()

        # Generate heatmap for predicted class
        # Going backwards to find the features which contribute most to the prediction
        targets = [ClassifierOutputTarget(predicted_class)]
        heatmap = cam(input_tensor = batch_images, targets = targets)
        all_heatmaps.append(heatmap[0])

        count = count + 1
        if count % 50 == 0:
            print(f"GradCam finished. Explained: {count} / {max_samples} samples.")

    print(f"GradCam finished. Explained: {len(all_heatmaps)} samples.")
    return all_heatmaps

# Calculate explanation consistency for tabular (SHAP and LIME)
# The agreement of both SHAP and LIME top 5 features
def calculate_consistency_tabular(shap_values, lime_explanation, feature_names, top = 5):
    print("Calculate consistency tabular.")
    # Empty array for collect 200 samples scores
    consistency_scores = []

    # Loop each LIME sample and compare with SHAP
    for i in range(len(lime_explanation)):
        # Check whether the shap values return with two arrays ( TreeExplainer )
        if isinstance(shap_values, list):
            sample_shap = np.abs(shap_values[1][i])
        elif len(shap_values.shape) == 3:
            sample_shap = np.abs(shap_values[i, :, 1])
        else:
            sample_shap = np.abs(shap_values[i])

        # Sort values and return its indices (last 5 biggest values )
        shap_top_indices = set(np.argsort(sample_shap)[-top:])

        # Get LIME top features
        lime_exp = lime_explanation[i]
        lime_top_indices = set()
        # Select top 5 features
        for feature_text, weight in lime_exp.as_list()[:top]:
            # Convert LIME text back to features
            for idx, name in enumerate(feature_names):
                if name in feature_text:
                    # Filtered LIME top indices
                    lime_top_indices.add(idx)
                    break

        # Calculate overlap between SHAP and LIME
        if len(lime_top_indices) > 0:
            overlap = len(shap_top_indices.intersection(lime_top_indices))
            consistency = overlap / top
        else:
            consistency = 0
        # Add one sample consistency score to the list
        consistency_scores.append(consistency)

    consistency_scores = np.array(consistency_scores)
    return consistency_scores

# Collect user feedback on Grad-CAM heatmaps
def collect_gradcam_feedback(num_samples):
    print(f"Review the {num_samples} heatmap above")
    print("Enter verdicts as comma separated numbers. Ex:1,2,3,2,2,1")
    print("1 = Correct (model is looking at the correct areas)")
    print("2 = Partial (model is looking at some correct areas)")
    print("3 = Incorrect (model is looking at wrong areas)")

    # Collect user input
    user_input = input("Verdicts: ")
    verdicts = user_input.split(',')

    # Call again the collect function on wrong inputs
    if len(verdicts) != num_samples:
        print("Invalid number of verdicts.")
        return collect_gradcam_feedback(num_samples)

    # Convert to consistency scores
    score_map = {'1': 1.0, '2': 0.5, '3': 0.0}
    consistency_scores = []

    for verdict in verdicts:
        verdict = verdict.strip()
        if verdict in score_map:
            consistency_scores.append(score_map[verdict])
        else:
            # Assign partial score for unknown user inputs
            consistency_scores.append(0.5)

    consistency_scores = np.array(consistency_scores)

    correct = np.sum(consistency_scores == 1.0)
    partial = np.sum(consistency_scores == 0.5)
    incorrect = np.sum(consistency_scores == 0.0)

    print(f"Correct: {correct}, Incorrect: {incorrect}, Partial: {partial}")
    return consistency_scores, correct, incorrect, partial

