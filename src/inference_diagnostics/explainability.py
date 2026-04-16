from xml.sax.handler import feature_namespaces

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

    # Sample background data
    background = shap.sample(X_train, num_background)

    # Sample test data to explain
    samples_explained = X_test.iloc[:max_samples]

    # Chose the explainer
    if is_pytorch:
        device = torch.device(config['device'])
        model.eval()

        def predict_function(sample):
            x_tensor = torch.FloatTensor(sample).to(device)
            with torch.no_grad():
                outputs = model(x_tensor)
                probs = torch.softmax(outputs, dim = 1).cpu().numpy()
            return probs

        explainer = shap.KernelExplainer(predict_function, background)
    else:
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.KernelExplainer(model.predict_proba, background)

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
        model.eval()

        def predict_function(sample):
            x_tensor = torch.FloatTensor(sample).to(device)
            with torch.no_grad():
                outputs = model(x_tensor)
                probs = torch.softmax(outputs, dim = 1).cpu().numpy()
            return probs
    else:
        predict_function = model.predict_proba

    # Explain each sample
    all_explanations = []
    for ex in range(len(samples_explained)):
        explanation = explainer.explain_instance(
            data_row = samples_explained.iloc[ex].values,
            predict_fn = predict_function,
            num_features  = num_features,#
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

    model.eval()

    # Find the lat convolutional layer
    target_layer = None
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            target_layer = module#
            break

    if target_layer is None:
        raise ValueError("Target layer not found.")

    # Create Grad-CAM
    cam = GradCAM(model, target_layers = [target_layer] )

    # Get samples to explain
    loader = DataLoader(test_set, batch_size = 1, shuffle = False)

    all_heatmaps = []
    count = 0

    for batch_images, batch_labels in loader:
        if count >= max_samples:
            break

        batch_images = batch_images.to(device)

        # Get model prediction
        with torch.no_grad():
            output = model(batch_images)
            predicted_class = output.argmax(dim = 1).item()

        # Generate heatmap for predicted class
        targets = [ClassifierOutputTarget(predicted_class)]
        heatmap = cam(input_tensor = batch_images, targets = targets)
        all_heatmaps.append(heatmap[0])

        count = count + 1
        if count % 50 == 0:
            print(f"GradCam finished. Explained: {count} / {max_samples} samples.")

    print(f"GradCam finished. Explained: {len(all_heatmaps)} samples.")
    return all_heatmaps



