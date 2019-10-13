from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
import csv

def loadData(): #loads data from csv
    data = pd.read_csv('train.csv')
    data.head()
    data_full = data.copy()

    X_train = data_full.drop('class' , axis=1)
    y_train = data_full.loc[:,'class']

    X_test = pd.read_csv("test.csv")

    return X_train, y_train, X_test


def preprocessing(X_train, X_test): #PCA for dimensionality reduction
    X_concat = pd.concat([X_train,X_test])

    pca = PCA(0.85, svd_solver='full')
    pca.fit(X_concat)
    X_pca = pca.transform(X_concat)

    return X_pca[:120,:], X_pca[120:,:]


def trainModel(X_train, X_test, y_train):#train model with AdaBoosted Gaussian Naive Bayes

    adaboost = AdaBoostClassifier(GaussianNB(),
                             algorithm="SAMME",
                             n_estimators=200)
    adaboost = adaboost.fit(X_train, y_train)
    return adaboost

def predict(adaboost, X_test): #make predictions with X_test
    return adaboost.predict(X_test)


def writeOutput(pred): #write to submission file
    final = []
    for i in range (80):
        final += [[i+1, pred[i]]]

    with open('adaboostgaussianFINAL.csv', mode = 'w', newline = '\n') as filename:
        finalfile = csv.writer(filename, delimiter=',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        finalfile.writerow(['ID', 'Predicted'])
        for i in range (80):
            finalfile.writerow(final[i])

X_train, y_train, X_test = loadData()
X_train, X_test = preprocessing(X_train, X_test) #does PCA
writeOutput(predict(trainModel(X_train, X_test,y_train), X_test)) #trains, predicts and writes to submission file sequentially
