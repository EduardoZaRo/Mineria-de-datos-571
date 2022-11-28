import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import numbers
import math
pd.options.mode.chained_assignment = None  # default='warn'

def histogramas(dataset, title):
    atributos = dataset.columns
    subplot_size = math.ceil(( math.sqrt(atributos.size)))
    bin_range = round( math.sqrt(atributos.size))
    fig = plt.figure()
    for i in range(len(atributos)):
        plt.subplot(subplot_size, subplot_size, i+1)
        plt.hist(dataset[atributos[i]], bins=bin_range, edgecolor='black')
        plt.xlabel(atributos[i])
        plt.ylabel('count')
    fig.suptitle(title, fontsize=20)
    fig.tight_layout(pad = 1.2)
    plt.show()
#DATASET AIRBNB
dataset = pd.read_csv("listings.csv")
print("\tINFORMACION DEL DATASET\n______________________________________")
#dataset.info()
'''
print("\tCANTIDAD DE DATOS FALTANTES EN DATASET (TOTAL ROWS X COLS)\n________________________________________________________________________")
print("\t% of missing data: ", dataset.isna().sum().sum() / (dataset.size) *100)

#Metodo 1: Eliminar registros con valores faltantes
dataset_ignored_NAN = dataset.copy()
dataset_ignored_NAN.dropna(axis = 'index',inplace= True)
#dataset_ignored_NAN.info()
print("\tCANTIDAD DE DATOS FALTANTES EN DATASET (TOTAL ROWS X COLS)\n________________________________________________________________________")
print("\t% of missing data: ", dataset_ignored_NAN.isna().sum().sum() / (dataset_ignored_NAN.size) *100)
#En este dataset funciona bastante bien

#Metodo 2: Reemplazar NANs con una constante (No recomendado porque no hay supervicion de un experto)
dataset_replace_constant = dataset.copy()
dataset_replace_constant.fillna(0, inplace=True) #Se reemplaza con 0
#dataset_replace_constant.info() #Se puede ver que ahora no faltan datos, pero a que costo :'(
'''


dataset = pd.read_csv("listings.csv")
for i in dataset.columns:
    try:
        if(dataset.isna()[i].sum().sum() == 0):
            continue
        dataset[i].fillna(dataset.groupby("neighbourhood_group")[i].transform("median"), inplace=True)
    except:
        continue
    print(i, dataset[i].median())
histogramas(dataset.select_dtypes(['number']), "mean by class")
print("_________________________________________")
dataset = pd.read_csv("listings.csv")
for i in dataset.columns:
    try:
        if(dataset.isna()[i].sum().sum() == 0):
            continue
        dataset[i].fillna(dataset[i].median(), inplace=True)
    except:
        continue
    print(i, dataset[i].median())
histogramas(dataset.select_dtypes(['number']), "mean by atributte")

'''
#Metodo 3: Reemplazar NANs con una medida de tendencia central
dataset_replace_mode = dataset.copy()
#Para trabajar media y mediana se ocupan valores numericos, entonces descarto strings
dataset_replace_mean = dataset.select_dtypes(['number'])
dataset_replace_median = dataset_replace_mean.copy()
for i in dataset_replace_mode.columns:
    dataset_replace_mode[i].fillna(dataset_replace_mode[i].mode()[0], inplace=True) #Se reemplaza con la moda
print("__________________________________________________")
for i in dataset_replace_mean.columns:
    if(dataset_replace_mean[i].isnull().values.any()):
        print(i, dataset_replace_mean[i].mean())
        dataset_replace_mean[i].fillna(dataset_replace_mean[i].mean(), inplace=True) #Se reemplaza con la media
print("__________________________________________________")
for i in dataset_replace_median.columns:
    if(dataset_replace_median[i].isnull().values.any()):
        print(i, dataset_replace_median[i].median())
        dataset_replace_mean[i].fillna(dataset_replace_median[i].median(), inplace=True) #Se reemplaza con la mediana
        
print("__________________________________________________")
histogramas(dataset_replace_mean)
print("__________________________________________________")
histogramas(dataset_replace_median)
print("__________________________________________________")
histogramas(dataset_replace_mode.select_dtypes(['number']))'''