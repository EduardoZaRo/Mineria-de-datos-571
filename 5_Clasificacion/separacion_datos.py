import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import math
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus
import time
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.linear_model import SGDClassifier
'''
Formato matriz de confusion

              y_test
             __________
            |    |    |
            | TN | FP |  N
y_pred      |_________|
            | FN | TP |  P
            |____|____|

https://scikit-learn.org/stable/modules/preprocessing.html
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html?highlight=classifier
https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
'''
class classifier():
    """
        Parameters
        ----------
        X: array of data
        y : array of classes
        split_method: str ['holdout', 'random_subsampling', 'kfold', 'leaveoneout','stratifiedkfold']
        classifier: str ['logistic', 'kneighbors', 'decisiontree', 'gaussianNB', 'sgd', 'adaboost',  'mlp']
        
        k {OPTIONAL}: Obligatory when use [kfold, random_subsampling, stratifiedkfold]
        train_size {OPTIONAL}: Obligatory when use holdout
        
        Methods
        -------
        
        classify(self): Classifies and get metrics
        
        getMetrics(self): Return metrics outside class
    """
    def __init__(self, X, y, split_method: str, classifier: str, k=None, train_size=None,):
        self.X = X
        self.y = y
        self.split_method = split_method
        self.classifier = classifier
        self.k = k if k else None
        self.train_size = train_size if train_size else None
        self.y_train = []
        self.y_test = []
        self.X_train = []
        self.X_test = []
        self.y_pred = []
        self.metrics = [0,0,0,0,0]
        print("Parametros recibidos:\n\t", split_method, classifier, k, train_size)
    def classify(self):
        sc = StandardScaler()
        if(self.split_method == 'holdout'):
            if(self.train_size == 1):
                X_train,X_test,y_train,y_test = self.X, self.X, self.y, self.y
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
            else:
                X_train,X_test,y_train,y_test = train_test_split(self.X, self.y, train_size=self.train_size)
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
        elif (self.split_method == 'random_subsampling'):
            cv = ShuffleSplit(train_size=self.train_size, n_splits=30)
        elif (self.split_method == 'kfold'):
            cv = KFold(n_splits=self.k)
        elif (self.split_method == 'leaveoneout'):
            cv = LeaveOneOut()
        elif (self.split_method == 'stratifiedkfold'):
            cv = StratifiedKFold(n_splits=self.k)
        else:
            print("Opcion no valida")
        sc = StandardScaler()
        X_scaled = sc.fit_transform(self.X)
        if(self.classifier == 'logistic'):
            model = LogisticRegression()
        if(self.classifier == 'kneighbors'):
            model = KNeighborsClassifier()
        if(self.classifier == 'decisiontree'):
            model = DecisionTreeClassifier()
        if(self.classifier == 'gaussianNB'):
            model = GaussianNB() 
        if(self.classifier == 'adaboost'):
            model = AdaBoostClassifier()
        if(self.classifier == 'mlp'):
            model = MLPClassifier(alpha=1, max_iter=500)
        if(self.classifier == 'sgd'):
            model = SGDClassifier()
        if(self.split_method == 'holdout'):
            y_pred = model.fit(X_train, y_train).predict(X_test)
            self.y_test = y_test
            self.y_pred = y_pred
            cm,accuracy,error_rate,sensitivity,specifity,precision = self.getInternalMetrics()
            self.metrics[0] = accuracy 
            self.metrics[1] = error_rate 
            self.metrics[2] = sensitivity 
            self.metrics[3] = specifity 
            self.metrics[4] = precision 
            
        else:

            result = cross_val_score(model , X_scaled, self.y, cv = cv)
            counter = 0
            y_preds = []
            y = self.y if self.split_method == 'stratifiedkfold' else None
            c = 0
            metrics = {
                'accuracy':  np.empty(0),
                'error_rate': np.empty(0),
                'sensitivity': np.empty(0),
                'specifity': np.empty(0),
                'precision': np.empty(0)
            }
            for train_index , test_index in cv.split(self.X, y):
                X_train , X_test = X_scaled[train_index,:],X_scaled[test_index,:]
                y_train , y_test = self.y[train_index] , self.y[test_index]
                y_pred = model.fit(X_train,y_train).predict(X_test)
                self.y_test = y_test
                self.y_pred = y_pred
                
                
                if(self.split_method == 'leaveoneout'):
                    y_preds.append(y_pred)
                else:
                    cm,accuracy,error_rate,sensitivity,specifity,precision = self.getInternalMetrics()
                    metrics['accuracy'] = np.append(metrics['accuracy'], accuracy)
                    metrics['error_rate'] = np.append(metrics['error_rate'], error_rate)
                    metrics['sensitivity'] = np.append(metrics['sensitivity'], sensitivity)
                    metrics['specifity'] = np.append(metrics['specifity'], specifity)
                    metrics['precision'] = np.append(metrics['precision'], precision)
                counter += 1
            if(self.split_method != 'leaveoneout'):
                self.metrics[0] = metrics['accuracy'].mean()
                self.metrics[1] = metrics['error_rate'].mean()
                self.metrics[2] = metrics['sensitivity'].mean()
                self.metrics[3] = metrics['specifity'].mean()
                self.metrics[4] = metrics['precision'].mean()

            #print("En k = ", counter, " esta la mayor accuracy")
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        self.y_pred = y_pred
        if(self.split_method == 'leaveoneout'):
            y_preds = np.asarray(y_preds)
            self.y_pred = y_preds
            self.y_test = self.y
            
            cm,accuracy,error_rate,sensitivity,specifity,precision = self.getInternalMetrics()
            
            
            metrics['accuracy'] = np.append(metrics['accuracy'], accuracy)
            metrics['error_rate'] = np.append(metrics['error_rate'], error_rate)
            metrics['sensitivity'] = np.append(metrics['sensitivity'], sensitivity)
            metrics['specifity'] = np.append(metrics['specifity'], specifity)
            metrics['precision'] = np.append(metrics['precision'], precision)
            self.metrics[0] = metrics['accuracy'].mean()
            self.metrics[1] = metrics['error_rate'].mean()
            self.metrics[2] = metrics['sensitivity'].mean()
            self.metrics[3] = metrics['specifity'].mean()
            self.metrics[4] = metrics['precision'].mean()
    def getInternalMetrics(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        #print(cm)
        tn, fp, fn, tp = cm.ravel()
        p = fn + tp
        n = fp + tn
        accuracy = (tp+tn)/(p+n)
        error_rate = (fp+fn)/(p+n)
        sensitivity = tp/p
        specifity = tn/n
        precision = tp/(tp+fp+1e-8)
        return  cm,accuracy,error_rate, \
                sensitivity,specifity,precision
    def getMetrics(self):
        return  self.metrics[0],self.metrics[1],\
                self.metrics[2],self.metrics[3],\
                self.metrics[4]
def tableMethod(method):
    from tabulate import tabulate
    table = []
    head_table = ["SPLIT METHOD","accuracy","error_rate", "sensitivity", "specifity","precision"]
    split_methods = ['holdout', 'holdout','random_subsampling', 'kfold', 'leaveoneout', 'stratifiedkfold',]
    table.append(head_table)
    k = 0
    train_size = 0
    max_prom = 0
    best_row = [0,99,0,0,0]
    best_split =''
    for i in range(len(split_methods)):
        if(i == 0):
            train_size = 0.6
            k = 0
        elif(i == 1):
            train_size = 1
            k = 0
        elif(i == 2):
            train_size = 0.6
            k = 30
        else:
            train_size = 0
            k = 10
        c = classifier(X, y, split_methods[i], method, k, train_size)
        c.classify()
        accuracy,error_rate,sensitivity,specifity,precision = c.getMetrics()
        row = [split_methods[i], accuracy,error_rate,sensitivity,specifity,precision]
        if(row[1] > best_row[0] and row[2] < best_row[1] and
           row[3] > best_row[2] and row[4] > best_row[3] and row[5] > best_row[4]):
            best_row = row[1:]
            best_split = split_methods[i] + '| k = ' + str(k) + '| train_size = ' + str(train_size)
        table.append(row)
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    return best_row, best_split
dataset = pd.read_csv("wine.csv", on_bad_lines='skip')
labels = dataset.columns
X = dataset.loc[:, labels[:-1]].values
y = dataset.loc[:, labels[-1]].values

import timeit
best_row = [0,99,0,0,0]
best_combination= ''
split_methods = ['holdout', 'random_subsampling', 'kfold', 'stratifiedkfold',]
classifiers = ['logistic', 'kneighbors', 'decisiontree', 'gaussianNB', 'sgd', 'adaboost',  'mlp']

for i in range(len(classifiers)-2):
    print("________________{}________________".format(classifiers[i]))
    start = timeit.default_timer()
    aux, split = tableMethod(classifiers[i])
    end = timeit.default_timer()
    if(aux[0] > best_row[0] and aux[1] < best_row[1] and
       aux[2] > best_row[2] and aux[3] > best_row[3] and 
       aux[4] > best_row[4] and split != 'holdout| k = 0| train_size = 1'):
        best_row = aux
        best_combination = classifiers[i] + '|' + split
    print("Tiempo de ejecucion:", end - start, " segundos")
print("Best combination:", best_combination)



