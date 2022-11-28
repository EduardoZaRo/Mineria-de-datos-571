import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import math
import time
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from dateutil.parser import parse
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from tabulate import tabulate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
import statistics
class regressor():
    """
    Clase para hacer regresiones
    Soporta los splitters: [holdout,random_subsampling,kfold]
    Soporta los regresores: [linear, decisiontree, kneighbors]
    
    regressor(self, X, y, split_method: str, regressor: str, k=None, train_size=None)
    
    Se tienen los metodos:
        regression: Hace el fit y predict
        getMetrics: Retorna r,r2,sse,mae,mse,rmse
    """
    def __init__(self, X, y, split_method: str, regressor: str, k=None, train_size=None):
        self.X = X
        self.y = y
        self.split_method = split_method
        self.regressor = regressor
        self.k = k if k else None
        self.train_size = train_size if train_size else None
        self.y_train = []
        self.y_test = []
        self.X_train = []
        self.X_test = []
        self.y_pred = []
        self.fit = []
        self.metrics = [0,0,0,0,0,0]
        self.multioutput = False
        try:
            if(y.shape[1] >= 2):
                self.multioutput = True
        except:
            self.multioutput = False
                
        print("Parametros recibidos:\n\t", split_method, regressor, k, train_size)
    def regression(self):
        if(self.split_method == 'holdout'):
            if(self.train_size == 1):
                X_train,X_test,y_train,y_test = self.X, self.X, self.y, self.y
            else:
                X_train,X_test,y_train,y_test = train_test_split(self.X, self.y, train_size=self.train_size)
        elif (self.split_method == 'random_subsampling'):
            cv = ShuffleSplit(n_splits=30)
        elif (self.split_method == 'kfold'):
            cv = KFold(n_splits=self.k, shuffle = True)
        else:
            print("Splitter no valido")
            return
            
        if(self.regressor == 'linear'):
            model = LinearRegression()
        elif(self.regressor == 'decisiontree'):
            model = DecisionTreeRegressor()
        elif(self.regressor == 'kneighbors'):
            model = KNeighborsRegressor()
        else:
            print("Regressor no valido")
            return 
        
        
        if(self.split_method == 'holdout'):
            self.fit = model.fit(X_train, y_train)
            if(self.multioutput and self.regressor != 'decisiontree'):
                y_pred = MultiOutputRegressor(model).fit(X_train, y_train).predict(X_test)
            else:
                y_pred = model.fit(X_train, y_train).predict(X_test)
            self.y_pred = y_pred
            self.y_test = y_test
            self.X_test = X_test
            r, r2, sse, mae, mse, rmse = self.getInternalMetrics()
           
            self.metrics[0] = np.array(r).mean()
            self.metrics[1] = r2 
            self.metrics[2] = sse 
            self.metrics[3] = mae 
            self.metrics[4] = mse 
            self.metrics[5] = rmse 
        else:
            metrics = {
                'r':  np.empty(0),
                'r2': np.empty(0),
                'sse': np.empty(0),
                'mae': np.empty(0),
                'mse': np.empty(0),
                'rmse': np.empty(0)
            }
            for train_index , test_index in cv.split(self.X):
                X_train , X_test = X[train_index,:],X[test_index,:]
                y_train , y_test = self.y[train_index] , self.y[test_index]
                if(self.multioutput):
                    y_pred = MultiOutputRegressor(model).fit(X_train, y_train).predict(X_test)
                else:
                    y_pred = model.fit(X_train, y_train).predict(X_test)
                self.fit = model.fit(X_train, y_train)
                self.y_test = y_test
                self.y_pred = y_pred
                self.X_test = X_test
                r, r2, sse, mae, mse, rmse = self.getInternalMetrics()
                metrics['r'] = np.append(metrics['r'], r)
                metrics['r2'] = np.append(metrics['r2'], r2)
                metrics['sse'] = np.append(metrics['sse'], sse)
                metrics['mae'] = np.append(metrics['mae'], mae)
                metrics['mse'] = np.append(metrics['mse'], mse)
                metrics['rmse'] = np.append(metrics['rmse'], rmse)
            self.metrics[0] = metrics['r'].mean()
            self.metrics[1] = metrics['r2'].mean()
            self.metrics[2] = metrics['sse'].mean()
            self.metrics[3] = metrics['mae'].mean()
            self.metrics[4] = metrics['mse'].mean()
            self.metrics[5] = metrics['rmse'].mean()
            
        self.y_pred = y_pred
        self.y_test = y_test
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
    def getInternalMetrics(self):
        sse = np.sum((self.y_test - self.y_pred)**2)
        mse = sse/len(self.y_test)
        rmse = np.sqrt(mse)
        mae = np.sum(np.abs((self.y_test - self.y_pred)))/len(self.y_test)
        r2 = r2_score(self.y_test, self.y_pred) 
        rs = []
        try:
            for i in self.X_test.T:
                for j in self.y_pred.T:
                    rs.append(np.corrcoef([i, j])[1,0])
        except:
            self.y_pred = np.array([self.y_pred]).T
            for i in self.X_test.T:
                for j in self.y_pred.T:
                    rs.append(np.corrcoef([i, j])[1,0])
        r = rs
        
        return  r, r2, sse, mae, mse, rmse
    def getMetrics(self):
        """
        Devuelve metricas de regresion en el siguiente orden:
                r, r2, sse, mae, mse, rmse
        """
        y_pred = self.fit.predict(self.X)
        x_ax = range(len(self.X))
        plt.plot(x_ax, self.y, 'o', label="original", markersize=2)
        plt.plot(x_ax, y_pred, label="predicted")
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend(loc='best',fancybox=True, shadow=True)
        plt.grid(True)
        plt.show() 
        return  self.metrics[0],self.metrics[1],\
                self.metrics[2],self.metrics[3],\
                self.metrics[4],self.metrics[5]
        
  
split_methods = ['holdout', 'random_subsampling', 'kfold', 'kfold']    
regression_methods = ['linear', 'decisiontree', 'kneighbors']    

'''
dataset = pd.read_csv("china.csv", on_bad_lines='skip') #1 output dataset
labels = dataset.columns
X = dataset.loc[:, labels[:-1]].values
y = dataset.loc[:, labels[-1]].values
'''
'''
dataset = pd.read_csv("ENB2012_data.csv", on_bad_lines='skip') #2 output dataset
labels = dataset.columns
X = dataset.loc[:, labels[:-2]].values
y = dataset.loc[:, labels[-2:]].values
'''

'''
dataset = pd.read_csv("yeast.csv", on_bad_lines='skip') #2 output dataset
labels = dataset.columns
X = dataset.loc[:, labels[1:-14]].values
y = dataset.loc[:, labels[-14:]].values
'''

'''
dataset = pd.read_csv("linear.csv", on_bad_lines='skip') #2 output dataset
labels = dataset.columns
X = dataset.loc[:, labels[:-1]].values
y = dataset.loc[:, labels[-1:]].values
'''

'''
dataset = pd.read_csv("wq.csv", on_bad_lines='skip') #2 output dataset
labels = dataset.columns
X = dataset.loc[:, labels[1:-14]].values
y = dataset.loc[:, labels[-14:]].values
'''

'''
dataset = pd.read_csv("oes97.csv", on_bad_lines='skip') #2 output dataset
labels = dataset.columns
X = dataset.loc[:, labels[1:-16]].values
y = dataset.loc[:, labels[-16:]].values
'''


dataset = pd.read_csv("scm20d.csv", on_bad_lines='skip') #2 output dataset
labels = dataset.columns
X = dataset.loc[:, labels[1:-16]].values
y = dataset.loc[:, labels[-16:]].values

best_row = [0,0,0,0,0,0]
best_combination = ['split', 'regressor']
for regression_method in regression_methods:
    i = 0
    table = []
    head_table = ["SPLIT METHOD","r","r2", "sse", "mae","mse","rmse"]
    table.append(head_table)
    for split_method in split_methods:
        if(i == 0):
            rObj = regressor(X,y,split_method,regression_method, 5, 0.6)
        if(i == 1):
            rObj = regressor(X,y,split_method,regression_method, 30, 0)
        if(i == 2):
            rObj = regressor(X,y,split_method,regression_method, 5, 0)
        if(i == 3):
            rObj = regressor(X,y,split_method,regression_method, 10, 0)
        i+=1
        rObj.regression()
        r, r2, sse, mae, mse, rmse = rObj.getMetrics()
        row = [split_method,
               float("{:.3f}".format(r)), 
               float("{:.3f}".format(r2)), 
               float("{:.3f}".format(sse)), 
               float("{:.3f}".format(mae)), 
               float("{:.3f}".format(mse)), 
               float("{:.3f}".format(rmse)) ]
        from statistics import mean
        import warnings
        warnings.filterwarnings('ignore')
        if(split_method == 'holdout'):
            best_row = row
        elif(mean([row[1],row[2]]) > mean([best_row[1],best_row[2]])and
             mean([row[3],row[4],row[5],row[6]]) < mean([best_row[3],best_row[4],best_row[5],best_row[6]])):
            best_row = row
            best_combination = [split_method, regression_method]
        table.append(row)
    print("________________{}________________".format(regression_method))
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
print(best_combination)




'''
dataset = pd.read_csv("china.csv", on_bad_lines='skip')
labels = dataset.columns
X = dataset.loc[:, labels[:-1]].values
y = dataset.loc[:, labels[-1]].values
    
    
model = DecisionTreeRegressor()

X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.6) 
fitted_model = model.fit(X_train,y_train)

y_pred = fitted_model.predict(X)

x_ax = range(len(X))
plt.plot(x_ax, y, 'o', label="original", markersize=2)
plt.plot(x_ax, y_pred, label="predicted")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show() 
'''


