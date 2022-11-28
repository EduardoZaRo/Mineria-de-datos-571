import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import numbers
import math
pd.options.mode.chained_assignment = None  # default='warn'

def histogramas(df, title):
    atributos = df.columns
    subplot_size = math.ceil(( math.sqrt(atributos.size)))
    bin_range = round( math.sqrt(atributos.size))
    fig = plt.figure()
    for i in range(len(atributos)):
        plt.subplot(subplot_size, subplot_size, i+1)
        plt.hist(df[atributos[i]], bins=bin_range, edgecolor='black')
        plt.xlabel(atributos[i])
        plt.ylabel('count')
    fig.suptitle(title, fontsize=20)
    fig.tight_layout(pad = 1.2)
    plt.show()
    



dataset = pd.read_csv("Building_Permits.csv", on_bad_lines='skip')
print("\tINFORMACION DEL DATASET\n______________________________________")
#dataset.info()

print("\tCANTIDAD DE DATOS FALTANTES EN DATASET (TOTAL ROWS X COLS)\n________________________________________________________________________")
print("\t% of missing data: ", dataset.isna().sum().sum() / (dataset.size) *100)

#Metodo 1: Eliminar registros con valores faltantes
dataset_ignored_NAN = dataset.copy()
#Si se hace dropna() se elimina todo el dataset ya que en filas y columnas faltan datos
dataset_ignored_NAN.dropna(axis = 'columns', inplace= True) #Primero dropeo las columnas con NAN's
#dataset_ignored_NAN.info()
#Metodo menos recomendado porque como faltan muchos datos al querer ignorarlos se ignora todo el dataset :o

#Metodo 2: Reemplazar NANs con una constante (No recomendado porque no hay supervicion de un experto)
dataset_replace_constant = dataset.copy()
dataset_replace_constant.fillna(0, inplace=True) #Se reemplaza con 0
#dataset_replace_constant.info() #Se puede ver que ahora no faltan datos, pero a que costo :'(

#Metodo 3: Reemplazar NANs con una medida de tendencia central
dataset_replace_media = dataset.copy().select_dtypes(['number'])
dataset_replace_mediana = dataset.copy().select_dtypes(['number'])

def reemplazarPorMTC(modo, df):
    df = df.select_dtypes(['number'])
    for i in df.columns:
        try:
            if(df[i].isnull().values.any()):
                if(modo == 0): #Media
                    df[i].fillna(df[i].mean(), inplace=True)
                    print(i, df[i].mean())
                if(modo == 1): #Mediana
                    df[i].fillna(df[i].median(), inplace=True)
                    print(i, df[i].median())
        except:
            continue
    return df

dataset_replace_media = reemplazarPorMTC(0, dataset_replace_media)
print(dataset_replace_media.isna().sum())
histogramas(dataset_replace_media, "Media")
print("____________________________________________________")
dataset_replace_mediana = reemplazarPorMTC(1, dataset_replace_mediana)
print(dataset_replace_mediana.isna().sum())
histogramas(dataset_replace_mediana, "Mediana")