from get_data import get_data, read_params
import os
import argparse

def load_data_and_save(config_path):
    config = read_params(config_path)
    dataset = get_data(config_path)
    new_col = [col for col in dataset.columns]
    raw_data = config['load_data']['raw_dataset_csv']
    dataset.to_csv(raw_data, index = False)
if __name__ == "__main__":
    argpaser = argparse.ArgumentParser()
    argpaser.add_argument("--config", default= 'params.yaml')
    parsed_argument = argpaser.parse_args()
    load_data_and_save(parsed_argument.config)