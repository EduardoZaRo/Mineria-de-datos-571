import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import numbers
import math
pd.options.mode.chained_assignment = None  # default='warn'

#DATASET AIRBNB
dataset = pd.read_csv("Watch_accelerometer.csv",index_col=False)
'''
print("\tINFORMACION DEL DATASET\n______________________________________")
print(dataset.info())
print(dataset.isna().sum())

#Metodo 1: Eliminar registros con valores faltantes
dataset_ignored_NAN = dataset.copy()
dataset_ignored_NAN.dropna(inplace= True)
print(dataset_ignored_NAN.isna().sum())
print(dataset_ignored_NAN.shape)

#Metodo 2: Reemplazar NANs con una constante 
dataset_replace_constant = dataset.copy()
dataset_replace_constant.fillna('No identificado', inplace=True)
print(dataset_replace_constant.isna().sum())
print(dataset_replace_constant.shape)
'''
#Metodo 3: Reemplazar NANs con una medida de tendencia central
dataset_replace_mode = dataset.copy()
for i in dataset_replace_mode.columns:
    if(dataset_replace_mode[i].isnull().values.any()):
        dataset_replace_mode[i].fillna(dataset_replace_mode.groupby("User")[i].transform(lambda x: x.value_counts().idxmax()), inplace=True) #Se reemplaza con la moda
print(dataset_replace_mode.isna().sum())
print(dataset_replace_mode.shape)