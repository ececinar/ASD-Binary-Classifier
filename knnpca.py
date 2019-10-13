import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import datasets
from sklearn.decomposition import PCA

data_full = pd.read_csv('train.csv')

X_train = data_full.drop('class' , axis=1)
y_train = data_full.loc[:,'class']

X_test = pd.read_csv('test.csv')

def pca_func(X_train, X_test):
    
    X_concat = pd.concat([X_train,X_test])

    pca = PCA(0.8, svd_solver='full')
    pca.fit(X_concat)
    X_pca = pca.transform(X_concat)

    return X_pca[:120,:], X_pca[120:,:]

X_train, X_test = pca_func(X_train, X_test)

knn = KNeighborsClassifier() #KNN
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

final = []
for i in range (80):
    final += [[i+1, pred[i]]]

with open('knn.csv', mode = 'w', newline = '\n') as file:
    f = csv.writer(file, delimiter=',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
    f.writerow(['ID', 'Predicted'])
    for i in range (80):
        f.writerow(final[i])
