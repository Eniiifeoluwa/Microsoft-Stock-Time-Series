import pandas as pd
import numpy as np
import os
from get_data import read_params
from sklearn.model_selection import train_test_split
import argparse

def split_data(config_path):
    config = read_params(config_path)
    train_path = config['split_data']['train_path']
    test_path = config['split_data']['test_path']
    raw_data_path = config['load_data']['raw_dataset_csv']
    random_state = config['base']['random_state']
    split_ratio = config['split_data']['test_size']
    raw_data = pd.read_csv(raw_data_path)
    real_size = int(len(raw_data) * split_ratio)
    train_data, test_data = raw_data[:real_size], raw_data[real_size:]
    train_data.to_csv(train_path, index = False)
    test_data.to_csv(test_path, index = False)

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config", default= 'params.yaml')
    parsed_argument = argument_parser.parse_args()
    split_data(parsed_argument.config)
