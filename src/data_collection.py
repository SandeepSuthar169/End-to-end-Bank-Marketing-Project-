import numpy as np
import pandas as pd
import sklearn
from  sklearn.model_selection import train_test_split
import yaml
import os
import logging




url = "https://raw.githubusercontent.com/SandeepSuthar169/Datasets/main/bank.csv"
data = pd.read_csv(url)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

data_path = os.path.join('data', 'raw')

os.makedirs(data_path)

train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
test_data.to_csv(os.path.join(data_path, 'test.csv'), index =False)


