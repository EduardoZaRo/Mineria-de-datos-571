import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import numbers
import math
from skfeature.function.similarity_based import fisher_score
pd.options.mode.chained_assignment = None  # default='warn'

def fisher_score_selection(X, target, X_cols, dataset, porcentaje):
    score = fisher_score.fisher_score(X,Y)
    feat_importances = pd.Series(score, X_cols)
    
    feat_importances.plot(kind = 'barh', color = 'teal')
    plt.show()
    
    top = X.shape[1] * porcentaje//100
    feat_importances = feat_importances.sort_values(ascending=False)
    feat_importances.drop(feat_importances.tail(top).index,inplace=True)
    feat_importances = feat_importances.index.tolist()
    print("Selected: ", feat_importances)
    dataset = dataset[dataset.columns.intersection(feat_importances)]
    return dataset
dataset = pd.read_csv('iris.csv')
array = dataset.values
X = array[:,0:4]
X_cols = dataset.columns[0:4]
Y = array[:,4]
dataset = fisher_score_selection(X, Y, X_cols, dataset, 50)
print(dataset)


dataset = pd.read_csv('India_Menu.csv')
dataset1 = dataset.replace('[^0-9]+','', regex=True)
array = dataset.values
array1 = dataset1.values
X = array1[:,2:len(dataset.columns)].astype(float)
X_cols = dataset.columns[2:len(dataset.columns)]
Y = array[:,0]
dataset = fisher_score_selection(X, Y, X_cols, dataset, 50)
print(dataset)




