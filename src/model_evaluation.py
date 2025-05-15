import numpy as np
import pandas as pd
import json
import os
import pickle

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

test_data = pd.read_csv("./data/processed/test_processed.csv")

X_test = test_data.drop(columns=['deposit'])
y_test = test_data['deposit']

pipe = pickle.load(open("model.pkl", "rb"))

y_pred = pipe.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
#precision = precision_score(y_test, y_pred)
#recall = recall_score(y_test,y_pred)
#f1 = f1_score(y_test, y_pred)

metrics = {
    'accuracy':accuracy,
    #'precision':precision,
    #'recall':recall,
    #'f1':f1
}

with open('metrics.json', 'w') as file:
    json.dump(metrics, file, indent=4)
