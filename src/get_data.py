import pandas as pd
import numpy as np
import yaml
import os
import argparse

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    data_frame = config["data_source"]["s3_source"]
    data_frame = pd.read_csv(data_frame)
    print(data_frame.head(5))


if __name__== "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--config', default= 'params.yaml')
    parsed_argument = args.parse_args()
    get_data(config_path = parsed_argument.config)