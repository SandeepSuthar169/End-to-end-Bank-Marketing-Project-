import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import yaml
import os
import logging


def load_params(filepath: str) -> float:
    with open(filepath, "r") as file:
        params = yaml.safe_load(file)
    return params['data_collection']['test_size']


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def split_data(df: pd.DataFrame, test_size: float):
    return train_test_split(df, test_size=test_size, random_state=42)


def save_data(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False)


def main():
    data_filepath = r"https://raw.githubusercontent.com/SandeepSuthar169/Datasets/main/bank.csv"
    params_filepath = "params.yaml"
    raw_data_path = os.path.join('data', 'raw')

 
    os.makedirs(raw_data_path, exist_ok=True)

    data = load_data(data_filepath)
    test_size = load_params(params_filepath)
    train_data, test_data = split_data(data, test_size)

    save_data(train_data, os.path.join(raw_data_path, 'train.csv'))
    save_data(test_data, os.path.join(raw_data_path, 'test.csv'))


if __name__ == "__main__":
    main()

