from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import csv
import xgboost as xgb

data = pd.read_csv('train.csv')
data.head()
data_full = data.copy()

X_train = data_full.drop('class' , axis=1)
y_train = data_full.loc[:,'class']

X_test = pd.read_csv("test.csv")

rfc = RandomForestClassifier()
rfc_model = rfc.fit(X_train, y_train)
pred = rfc_model.predict(X_test)
importanceList = list(zip(X_train, rfc.feature_importances_))

for i in importanceList: #feature selection based on importances
    if i[1] == 0.0: 
        X_train = X_train.drop(i[0], axis = 1)

rfc2 = RandomForestClassifier()
rfc_model2 = rfc2.fit(X_train, y_train)
X_test = np.array(X_test)
pred2 = rfc_model2.predict(X_test[:,0:X_train.shape[1]])

final = []
for i in range (80):
    final += [[i+1, pred2[i]]]

with open('rainforest.csv', mode = 'w', newline = '\n') as file:
    f = csv.writer(file, delimiter=',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
    f.writerow(['ID', 'Predicted'])
    for i in range (80):
        f.writerow(final[i])
