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
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.cm as cm
from kneed import KneeLocator
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
dataset = pd.read_csv("wine_new.csv", on_bad_lines='skip')
labels = dataset.columns[0:11]
x = dataset.loc[:,labels].values
true_labels = dataset[['quality']].apply(lambda x: pd.factorize(x)[0])


neighb = NearestNeighbors(n_neighbors=x.shape[1]) 
nbrs=neighb.fit(x) 
distances,indices=nbrs.kneighbors(x) 
distances = np.sort(distances, axis = 0) 
distances = distances[:, 10] 

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


dbscan = DBSCAN(eps=6.19, min_samples=x.shape[1]).fit(x)
y = dbscan.fit_predict(x)
dataset['cluster'] = dbscan.labels_  
print("Ruido:", list(dbscan.labels_).count(-1))

#--------------EVALUACION EXTRINSECA--------------
homogeneo = metrics.homogeneity_score(true_labels.values.reshape(1,-1)[0], y) #0 a 1, mas alto mejor
print("homogeneity_score:",homogeneo)

completeness = metrics.completeness_score(true_labels.values.reshape(1,-1)[0], y) #0 a 1, mas alto mejor
print("completeness_score:",completeness)

vmeasure = metrics.v_measure_score(true_labels.values.reshape(1,-1)[0], y) #0 a 1, mas alto mejor
print("v_measure_score:",vmeasure)

'''
scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)
#elbow method
elbow = []
for i in range(1,11):
    kmedoids = KMedoids(n_clusters=i).fit(x_scaled)
    elbow.append(kmedoids.inertia_)
kn = KneeLocator(
    list(np.arange(1,11)),
    elbow,
    curve='convex',
    direction='decreasing',
    interp_method='polynomial',
)
plt.figure()
plt.plot(list(np.arange(1,11)), elbow)
plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
plt.plot(np.arange(1,11),elbow)
plt.xlabel('Clusters')
plt.ylabel('inertia_')
plt.show()

kmedoids_opt = KMedoids(n_clusters=5, random_state=0).fit(x_scaled)
y = kmedoids_opt.fit_predict(x_scaled)
dataset['cluster'] = y 


#--------------EVALUACION EXTRINSECA--------------
true = true_labels.values.reshape(1,-1)[0]
homogeneo = metrics.homogeneity_score(true, y) #0 a 1, mas alto mejor
print("homogeneity_score:",homogeneo)

completeness = metrics.completeness_score(true, y) #0 a 1, mas alto mejor
print("completeness_score:",completeness)

vmeasure = metrics.v_measure_score(true, y) #0 a 1, mas alto mejor
print("v_measure_score:",vmeasure)

'''









