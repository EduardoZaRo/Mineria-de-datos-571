'''
https://towardsdatascience.com/understanding-dbscan-and-implementation-with-python-5de75a786f9f
https://analyticsindiamag.com/comprehensive-guide-to-k-medoids-clustering-algorithm/
https://www.section.io/engineering-education/dbscan-clustering-in-python/
https://www.datanovia.com/en/lessons/dbscan-density-based-clustering-essentials/#:~:text=The%20aim%20is%20to%20determine,along%20the%20k-distance%20curve.&text=It%20can%20be%20seen%20that%20the%20optimal%20eps,around%20a%20distance%20of%200.15.


Para metodos de calidad:
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import math
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from sklearn.datasets import make_blobs
def graficar_clusters(n_iteraciones, dataset):
    unique_clusters = list(set(y))
    data = {}
    j = 0
    for i in range(len(unique_clusters)):
        data[j] = dataset[dataset.cluster==unique_clusters[i]]
        j += 1
    labels = dataset.columns
    colors = cm.turbo(np.linspace(0, 1, len(unique_clusters)))
    for angle in range(0, n_iteraciones):
        fig = plt.figure(figsize=(20,10))
        kplot = fig.add_subplot(111, projection='3d')
        for i in range(len(data)):
            a = data[i]
            kplot.scatter3D(a[labels[0]], a[labels[1]], a[labels[2]], c=colors[i].reshape(1,-1), label = str(i))
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        kplot.set_zlabel(labels[2])
        plt.legend(loc=2, prop={'size': 15})
        plt.title("DBSCAN")
        kplot.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
    plt.show()

dataset = pd.read_csv("customers_agr.csv", on_bad_lines='skip')
labels = dataset.columns
x = dataset.loc[:, ['Annual Income (k$)',
                 'Spending Score (1-100)'
                 ]].values
dataset = pd.read_csv("r.csv", on_bad_lines='skip')
x = dataset.loc[:, ['X',
                   'Y'
                    ]].values
scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)
neighb = NearestNeighbors(n_neighbors=2) 
nbrs=neighb.fit(x_scaled) 
distances,indices=nbrs.kneighbors(x_scaled) 
distances = np.sort(distances, axis = 0) 
distances = distances[:, 1] 

kn = KneeLocator(
    np.arange(len(distances)),
    distances,
    curve='convex',
    direction='increasing',
    interp_method='polynomial',
)
print("eps optimo:",distances[kn.knee])
plt.figure()
plt.plot(np.arange(len(distances)), distances)
plt.hlines(distances[kn.knee], plt.xlim()[0], plt.xlim()[1], linestyles='dashed')
plt.xlabel("distance")
plt.ylabel("eps")
plt.show()

dbscan = DBSCAN(eps =0.5, min_samples = 4).fit(x_scaled)
y = dbscan.fit_predict(x_scaled)
dataset['cluster'] = y  
print(set(y))
#graficar_clusters(1, dataset)


unique_clusters = list(set(y))
data = {}
j = 0
for i in range(len(unique_clusters)):
    data[j] = dataset[dataset.cluster==unique_clusters[i]]
    j += 1
labels = dataset.columns
colors = cm.turbo(np.linspace(0, 1, len(unique_clusters)))

for i in range(len(data)):
    a = data[i]
    plt.scatter(a[labels[0]], a[labels[1]], c=colors[i].reshape(1,-1), label = str(i))

plt.xlabel(labels[0])
plt.ylabel(labels[1])
plt.show()
#--------------EVALUACION INSTRINSECA--------------
print("________________INSTRINSECO__________________")
calinski = calinski_harabasz_score(x, y)
print("calinski_harabasz_score:",calinski) #Mayor es mejor

silhouette  = silhouette_score(x, y) #De -1 a 1, donde 1 es mejor
print("silhouette_score:",silhouette)

davies  = davies_bouldin_score(x, y)  #Menor es mejor
print("davies_bouldin_score:",davies)

'''
#--------------EVALUACION EXTRINSECA--------------
print("________________EXTRINSECO__________________")
homogeneo = metrics.homogeneity_score(true_labels, y) #0 a 1, mas alto mejor
print("homogeneity_score:",homogeneo)

completeness = metrics.completeness_score(true_labels, y) #0 a 1, mas alto mejor
print("completeness_score:",completeness)

vmeasure = metrics.v_measure_score(true_labels, y) #0 a 1, mas alto mejor
print("v_measure_score:",vmeasure)

'''



