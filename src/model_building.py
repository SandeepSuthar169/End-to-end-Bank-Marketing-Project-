import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

train_data = pd.read_csv("./data/processed/train_processed.csv")


X_train = train_data.drop(columns=['deposit'])
y_train = train_data['deposit']

#deposit, month, loan, housing, default
prepro= ColumnTransformer(transformers=[
        ('one', OneHotEncoder(), ['default', 'housing', 'loan', 'month'])
    ], 
        remainder='passthrough'
    )


pipe = Pipeline(steps =[
        ('prepro', prepro),
        ("classi", RandomForestClassifier())
    ])
    
 
pipe.fit(X_train, y_train)

pickle.dump(pipe, open('model.pkl', 'wb'))
