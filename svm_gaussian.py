################### SVM WITH GAUSSIAN KERNEL#################################
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

data = pd.read_csv("train.csv")

X_train = data.drop('class',axis = 1)
y_train = data.loc[:,'class']

X_test = pd.read_csv("test.csv")

def pca_func(X_train, X_test):
    
    X_concat = pd.concat([X_train,X_test])

    pca = PCA(0.8, svd_solver='full')
    pca.fit(X_concat)
    X_pca = pca.transform(X_concat)

    return X_pca[:120,:], X_pca[120:,:]

X_train, X_test = pca_func(X_train, X_test)

svclassifier = SVC(kernel = 'rbf') #train model with radial basis kernel SVM
svc_model = svclassifier.fit(X_train,y_train)
pred = svc_model.predict(X_test)

final = []
for i in range (80):
    final += [[i+1, pred[i]]]
        
with open('svm_gaussian.csv', mode = 'w', newline = '\n') as file:
    f = csv.writer(file, delimiter=',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
    f.writerow(['ID', 'Predicted'])
    for i in range (80):
        f.writerow(final[i])
