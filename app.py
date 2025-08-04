import streamlit as st
import joblib
import numpy as np
from src.get_data import read_params
import argparse


def run_prediction(config_path):
    st.title('THIS IS A LINEAR REGRESSION PROBLEM')
    config = read_params(config_path= config_path)
    
    model_path = config['model_dir']['model']
    scaler_path = config['model_dir']['scaler']
    with open(model_path, 'rb') as r:
        model = joblib.load(r)
    with open(scaler_path, 'rb') as s:
        scaler = joblib.load(s)
    st.write("Enter the values to forecast.")
    close = st.slider("Close", min_value = 20.0, max_value = 50.0, step = 1.0)
    opened = st.slider("Open", min_value = 20.0, max_value = 50.0, step = 1.0)
    high = st.slider("High", min_value = 20.0, max_value = 50.0, step = 1.0)
    low = st.slider('Low', min_value = 20.0, max_value = 50.0, step = 1.0)

    if st.button('Predict'):
        features = np.array([[opened, high, low, close]])
        features= scaler.transform(features)
        prediction = model.predict(features)
        st.write(f"The predicted volume is: {prediction[0]}")
if __name__ == '__main__':
    argpaser = argparse.ArgumentParser()
    argpaser.add_argument("--config", default= 'params.yaml')
    parsed_argument = argpaser.parse_args()
    run_prediction(parsed_argument.config)