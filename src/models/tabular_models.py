# Create and train baseline model using default configurations.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Train the Random Forest.
def train_random_forest(X_train, y_train, config):
    print("Random Forest training started.")

    params =  config['models']['random_forest']['params']

    model = RandomForestClassifier(
        n_estimators = params['n_estimators'],
        max_depth = params['max_depth'],
        random_state = params['random_state'],
        n_jobs = params['n_jobs']
    )

    model.fit(X_train, y_train)
    print("Random Forest training completed.")

    return model

# Train the XGBoost.
def train_xgboost(X_train, y_train, config):
    print("XGBoost training started.")

    params =  config['models']['xgboost']['params']

    model = xgb.XGBClassifier(
        n_estimators = params['n_estimators'],
        max_depth = params['max_depth'],
        learning_rate = params['learning_rate'],
        random_state = params['random_state'],
        eval_metric = 'logloss'
    )

    model.fit(X_train, y_train)
    print("XGBoost training completed.")

    return model

# Evaluate model for random forest and XGBoost and generate report.
def evaluate_model_rf_xgb(model, X_test, y_test):
    print("Model evaluation started for RF,XGB")

    y_prediction =  model.predict(X_test)
    accuracy = accuracy_score(y_test, y_prediction)
    report = classification_report(y_test, y_prediction)

    print(f"{report}")

    return accuracy, y_prediction

# Evaluate model for FCNN and generate report.
def evaluate_model_fcnn(model, X_test, y_test, config):
    print("Model evaluation started for FCNN.")

    device = torch.device(config['device'])
    model.eval()

    with torch.no_grad():
        X_tensor =  torch.FloatTensor(X_test.values).to(device)
        output = model(X_tensor)
        y_prediction = output.argmax(dim = 1).cpu().numpy()

    accuracy = accuracy_score(y_test, y_prediction)
    report = classification_report(y_test, y_prediction)

    print(f"{report}")

    return accuracy, y_prediction

# Create class for fully connected neural network.
class FCNNModel(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate):
        # Call parent class
        super(FCNNModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        # Map last 32 neurons to number of classes.
        self.fc4 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


# Train the FCNN by feeding data batches.
def train_fcnn(X_train, y_train, config):
    print("FCNN training started.")

    params =  config['models']['fcnn']['params']
    device = torch.device(config['device'])

    # Extract number of features and classes from the data.
    input_size = X_train.shape[1]
    numb_classes = len(np.unique(y_train))

    model = FCNNModel(
        input_size = input_size,
        num_classes = numb_classes,
        dropout_rate = params['dropout_rate']
    ).to(device)

    # Convert data to PyTorch tensors.
    X_tensor = torch.FloatTensor(X_train.values).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)

    # Create data loaders using PyTorch tensors.
    dataset =  TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size = params['batch_size'], shuffle = True)


    # Training process by epoch.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = params['learning_rate'])
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(params['epochs']):
        model.train()
        epoch_loss = 0.0

        # Feed the model batch wise.
        for batch_X, batch_y in loader:
            optimizer.zero_grad() # Clear old memory
            outputs = model(batch_X) # Forward pass
            loss = criterion(outputs, batch_y) # Calculate penalty score
            loss.backward() # Backpropagation
            optimizer.step() # Update weights
            epoch_loss = epoch_loss + loss.item()

        avg_loss = epoch_loss / len(loader)

        # Early stopping check.
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter = patience_counter + 1
            if patience_counter >= params['early_stopping_patience']:
                print(f" Early stopping at epoch {epoch + 1}")
                break

        # Print progress every 10 epochs.
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{params['epochs']}")

    print("FCNN training completed.")
    return model
