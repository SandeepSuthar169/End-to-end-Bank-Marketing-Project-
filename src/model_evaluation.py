import numpy as np
import pandas as pd
import json
import os
import pickle

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def load_test_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def load_model(model_path: str):
    with open(model_path, "rb") as file:
        return pickle.load(file)


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def save_metrics(metrics: dict, output_path: str):
    with open(output_path, "w") as file:
        json.dump(metrics, file, indent = 4)

def main():
    test_data_path = "./data/processed/test_processed.csv" 
    mode_path = "model.pkl"
    metrics_path = "metrics.json"

    test_data = load_test_data(test_data_path)
    X_test = test_data.drop(columns=['deposit'])
    y_test = test_data['deposit']

    model = load_model(mode_path)
    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics, metrics_path)



if __name__ == "__main__":
    main()
