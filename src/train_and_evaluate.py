from get_data import read_params
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor
import joblib
import json

def evaluate(y_pred, y_true):
    rmse = np.sqrt(mean_squared_error(y_pred.reshape(-1, 1), y_true))
    mse = mean_squared_error(y_pred.reshape(-1, 1), y_true)
    r2 = r2_score(y_pred.reshape(-1, 1), y_true)
    return rmse, mse, r2



def train_and_evaluate(config_path):
    config = read_params(config_path)
    training_path = config["split_data"]["train_path"]
    testing_path = config["split_data"]["test_path"]
    model_path = config["model_dir"]  
    target = config['base']['target_col']
    training_set = pd.read_csv(training_path)
    testing_set = pd.read_csv(testing_path)
    training_set.set_index('Date', inplace= True)
    testing_set.set_index('Date', inplace= True)
    X_train, Y_train = training_set.drop(columns= [target]), training_set[target]
    X_test, Y_test = testing_set.drop(columns= [target]), testing_set[target]
    alpha = config['estimators']['params']['alpha']
    l1_ratio = config['estimators']['params']['l1_ratio']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train.values)
    model = LinearRegression()
    model.fit(X_scaled, Y_train)
    joblib.dump(model, model_path)

    X_test_scaled = scaler.transform(X_test.values)
    Y_pred = model.predict(X_test_scaled)
    rmse, mse, r2 = evaluate(Y_pred, Y_test)
    score_file = config['report']['scores']
    params_file = config['report']['params']
    with open(score_file, 'w') as f:
        scores= {
            'RMSE': rmse,
            'MSE': mse,
            'R2': r2
            }
        json.dump(scores, f, indent= 4)
    with open(params_file, 'w') as f:
        params = {
            'alpha': alpha,
            'l1_ratio': l1_ratio
        }
        json.dump(params, f, indent= 4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train and evaluate the model')
    parser.add_argument('--config', default='params.yaml', help='Path to the config file')
    args = parser.parse_args()
    train_and_evaluate(args.config)
