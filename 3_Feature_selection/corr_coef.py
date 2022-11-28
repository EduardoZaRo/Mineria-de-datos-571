import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import numbers
import math
from skfeature.function.similarity_based import fisher_score
from scipy.stats import pointbiserialr
from math import sqrt
pd.options.mode.chained_assignment = None  # default='warn'

def corr_coef_metodo_1(dataset, threshold):
    to_drop = set()
    corr_matrix = dataset.corr()
    ax = plt.axes()
    sn.heatmap(corr_matrix,annot = True)
    ax.set_title('Original dataset correlation METODO 1')
    plt.show()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]  
                to_drop.add(colname)
    return to_drop

def corr_coef_metodo_2(dataset, threshold):
    corr_matrix = dataset.corr()
    ax = plt.axes()
    sn.heatmap(corr_matrix,annot = True)
    ax.set_title('Original dataset correlation METODO 2')
    plt.show()
    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop
print("_____________DATASET STARBUCKS_____________")
dataset = pd.read_csv('India_Menu.csv')
dataset['Per Serve Size'] = dataset['Per Serve Size'].replace('[^0-9]+','', regex=True)
dataset['Per Serve Size'] = dataset['Per Serve Size'].astype(int)
dataset.fillna(0, inplace=True) #Se reemplaza con 0
label = 'Menu Category'
features = dataset.drop(label, axis = 1)
target = dataset[[label]].apply(lambda x: pd.factorize(x)[0] + 1)

to_drop = corr_coef_metodo_1(features, 0.85)
print("Metodo 1 decide eliminar:",to_drop)

to_drop = corr_coef_metodo_2(features, 0.85)
print("Metodo 2 decide eliminar:",to_drop)

features = features.drop(to_drop, axis = 1)
print("Final dataset: \n", features)

print("_____________DATASET IRIS_____________")
dataset = pd.read_csv('iris.csv')
label = 'species'
features = dataset.drop(label, axis = 1)
target = dataset[[label]].apply(lambda x: pd.factorize(x)[0])

to_drop = corr_coef_metodo_1(features, 0.85)
print("Metodo 1 decide eliminar:",to_drop)

to_drop = corr_coef_metodo_2(features, 0.85)
print("Metodo 2 decide eliminar:",to_drop)

features = features.drop(to_drop, axis = 1)
print("Final dataset: \n", features)



