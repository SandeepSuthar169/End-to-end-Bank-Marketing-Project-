import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import yaml


def load_params(params_path: str) -> int:
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)
    return params['model_building']['n_estimators']


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=['deposit'])
    y = df['deposit']
    return X, y


def build_pipeline(n_estimators: int) -> Pipeline:
    prepro = ColumnTransformer(transformers=[
        ('one', OneHotEncoder(), ['default', 'housing', 'loan', 'month'])
    ], remainder='passthrough')

    pipe = Pipeline(steps=[
        ('prepro', prepro),
        ("classi", RandomForestClassifier(
            n_estimators=n_estimators
            ))
    ])
    return pipe


def train(pipe: Pipeline, X, y) -> Pipeline:
    pipe.fit(X, y)
    return pipe


def save_model(pipeline: Pipeline, output_path: str):
    with open(output_path, "wb") as file:
        pickle.dump(pipeline, file)


def main():
    input_path = "./data/processed/train_processed.csv"
    model_path = "model.pkl"
    params_path = "params.yaml"

    n_estimators = load_params(params_path)
    df = load_data(input_path)
    X_train, y_train = prepare_features(df)

    pipeline = build_pipeline(n_estimators)
    trained_pipeline = train(pipeline, X_train, y_train)
    save_model(trained_pipeline, model_path)
    print(f"Model trained and saved to {model_path}")


if __name__ == "__main__":
    main()
