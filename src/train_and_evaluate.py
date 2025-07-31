from get_data import read_params
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


def train_and_evaluate(config_path):
    config = read_params(config_path)
    training_set = config["split_data"]["train_data"]
    testing_set = config["split_data"]["test_data"]


    scaler = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = scaler.transform(training_set[['Close']])
    X_train, y_train = [], []
    for i in range(60, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.from_numpy(X_train).float())
        loss = criterion(outputs, torch.from_numpy(y_train).float())
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    model.eval()
    with torch.no_grad():
        test_preds = scaler.transform(X)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train and evaluate the model')
    parser.add_argument('--config', default='params.yaml', help='Path to the config file')
    args = parser.parse_args()
    train_and_evaluate(args.config)
