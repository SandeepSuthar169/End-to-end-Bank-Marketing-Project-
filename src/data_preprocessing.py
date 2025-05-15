import pandas as pd
import numpy as np
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

train_data = pd.read_csv("./data/raw/train.csv")
test_data = pd.read_csv("./data/raw/test.csv")


def columns_label_encoding(df):
    df['marital'] = df['marital'].replace({
    'divorced': 1,
    'single': 2,
    'married': 3
    })
    df['marital'] = df['marital'].astype(int)

    df['job'] = df['job'].replace({
    'unknown': 1,
    'unemployed': 1,
    'housemaid': 2,
    'student': 2,
    'self-employed': 3,
    'retired': 4,
    'services': 4,
    'admin.': 5,
    'technician': 5,
    'blue-collar': 5,
    'management': 6,
    'entrepreneur': 6   
    })
    df['job'] = df['job'].astype(int)

    df['poutcome'] = df['poutcome'].replace({
    'unknown': 1,
    'other': 1,
    'failure': 2,
    'success': 2    
    })
    df['poutcome'] = df['poutcome'].astype(int)

    df['education'] = df['education'].replace({
    'unknown': 1,
    'primary': 2,
    'tertiary': 3,
    'secondary': 4
    })
    df['education'] = df['education'].astype(int)

    df['contact'] = df['contact'].replace({
    'unknown': 1,
    'cellular': 2,
    'telephone': 3   
    })
    df['contact'] = df['contact'].astype(int)
    return df

train_pro_data = columns_label_encoding(train_data)
test_pro_data = columns_label_encoding(test_data)

data_path = os.path.join("data", "processed")
os.makedirs(data_path)

train_pro_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
test_pro_data.to_csv(os.path.join(data_path, "test_processed.csv"), index= False)
