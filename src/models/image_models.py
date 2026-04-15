# Create and train baseline model using default configurations.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from torchvision import models

# Create CNN class
class SimpleCNN(nn.Module):
    def __init__(self, num_channels, num_classes, dropout_rate):
        super(SimpleCNN, self).__init__()

        # Add convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Add dense layers
        self.classifier = None
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

    def build_classifier(self, flat_size):
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        # Create classifier on first pass when we know the flat size
        if self.classifier is None:
            self.build_classifier(x.size(1))
            self.classifier = self.classifier.to(x.device)

        x = self.classifier(x)
        return x

# Train the simple CNN
def train_simple_cnn(train_set, config):
    print(f"Simple CNN training started.")

    params = config['models']['simple_cnn']['params']
    dataset_config = config['datasets']['image_data']
    device = torch.device(config['device'])

    # Create the model
    model = SimpleCNN(
        num_classes = dataset_config['num_classes'],
        num_channels = dataset_config['num_channels'],
        dropout_rate = params['dropout_rate']
    ).to(device)

    # Create data loader for batching
    loader = DataLoader(train_set, batch_size = params['batch_size'], shuffle = True)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = params['learning_rate'])
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(params['epochs']):
        model.train()
        epoch_loss = 0.0

        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.item()

        avg_loss = epoch_loss / len(loader)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter = patience_counter + 1
            if patience_counter >= params['early_stopping_patience']:
                print(f"Early stopping at {epoch + 1}")
                break

        # Print progress every 5 epoches
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{params['epochs']}, Loss: {avg_loss:.4f}")

    print("Simple CNN training completed.")
    return model

# Evaluate image model
def evaluate_image_model(model, test_set, config):
    print(f" Image model evaluation started.")

    device = torch.device(config['device'])
    model.eval()

    loader = DataLoader(test_set, batch_size = 64, shuffle = False)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            preds = outputs.argmax(dim = 1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report =  classification_report(all_labels, all_preds)

    print(f"{report}")
    return accuracy, np.array(all_preds)

# ResNet_18 with pretrained weights
def build_resnet(config):

    dataset_config = config['datasets']['image_data']
    params = config['models']['resnet18']['params']

    # Load pretrained ResNet18
    model = models.resnet18(weights = 'DEFAULT')

    # Modify first layer if input is not 3 channels
    if dataset_config['num_channels'] != 3:
        model.conv1 = nn.Conv2d(
            dataset_config['num_channels'], 64, kernel_size = 7, stride =2, padding = 3, bias = False
        )

    # Replace final layer to match number of classes
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(params['dropout_rate']),
        nn.Linear(num_features, dataset_config['num_classes'])
    )

    return model

# Train ResNet
def train_resnet(train_set, config):
    print("Training resnet model started.")

    params = config['models']['resnet18']['params']
    device = torch.device(config['device'])

    model = build_resnet(config).to(device)
    loader = DataLoader(train_set, batch_size = params['batch_size'], shuffle = True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = params['learning_rate'])
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(params['epochs']):
        model.train()
        epoch_loss = 0.0

        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.item()

        avg_loss = epoch_loss / len(loader)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter = patience_counter + 1
            if patience_counter >= params['early_stopping_patience']:
                print(f"Early stopping at {epoch + 1}")
                break

        # Print progress every 5 epoches
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{params['epochs']}, Loss: {avg_loss:.4f}")

    print("ResNet18 training completed.")
    return model

