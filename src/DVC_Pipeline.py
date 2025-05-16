from dvclive import Live
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import mlflow
import dvc

from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    MinMaxScaler,
    StandardScaler
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import (
    make_pipeline,
    Pipeline
)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.metrics import ConfusionMatrixDisplay
import dvc 
from dvclive import Live
from feature_engine.outliers.winsorizer import Winsorizer
import yaml

#n_estimators = yaml.safe_load(open("C:/Users/Sande/Desktop/New folder (2)/End-to-end-Bank-Marketing-Project-/params.yaml"))['n_estimators']
n_estimators = 500
uri = "https://raw.githubusercontent.com/SandeepSuthar169/Datasets/main/bank.csv"
df = pd.read_csv(uri)

print(df.head())


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



#deposit, month, loan, housing, default
prepro= ColumnTransformer(transformers=[
    ('one', OneHotEncoder(), ['default', 'housing', 'loan', 'month'])
], 
     remainder='passthrough'
)


pipe = Pipeline(steps =[
    ('prepro', prepro),
    ("classi", RandomForestClassifier(n_estimators= n_estimators))
])



train_data, test_data=train_test_split(df, random_state=42, test_size=0.2)




X_train = train_data.drop(columns=['deposit'])
y_train = train_data['deposit']

X_test = test_data.drop(columns=['deposit'])
y_test = test_data['deposit']



pipe.fit(X_train, y_train)



y_pred = pipe.predict(X_test)


acc = accuracy_score(y_test,y_pred)
#pre = precision_score(y_test, y_pred)
#recall = recall_score(y_test, y_pred)
#f1 = f1_score(y_test, y_pred)



print("acc:->",acc)
#print('pre:->', pre)
#print('recall:->', recall)
#print('f1 :->', f1)


with Live(save_dvc_exp=True) as live:
    live.log_metric("accuracy", acc * 100)
    live.log_param('n_estimators', n_estimators)