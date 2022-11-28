'''
https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html#examples-using-sklearn-extra-cluster-kmedoids
https://scikit-learn-extra.readthedocs.io/en/stable/auto_examples/plot_kmedoids.html#sphx-glr-auto-examples-plot-kmedoids-py
https://medium.com/@sk.shravan00/k-means-for-3-variables-260d20849730
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
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import matplotlib.cm as cm
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
        plt.title("K-Medoids")
        kplot.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
    plt.show()


dataset = pd.read_csv("customers_agr.csv", on_bad_lines='skip')
x = dataset.loc[:, ['Age',
                 'Annual Income (k$)',
                 'Spending Score (1-100)'
                 ]].values
scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)

#elbow method
elbow = []
for i in range(1,11):
    kmedoids = KMedoids(n_clusters=i, init="k-medoids++").fit(x_scaled)
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

plt.xlabel('Clusters')
plt.ylabel('inertia_')
plt.show()

kmedoids_opt = KMedoids(n_clusters=5).fit(x_scaled)
y = kmedoids_opt.fit_predict(x_scaled)
dataset['cluster'] = y 
graficar_clusters(1, dataset)

#--------------EVALUACION INSTRINSECA--------------
calinski = calinski_harabasz_score(x, y)
print("calinski_harabasz_score:",calinski)

silhouette  = silhouette_score(x, y)
print("silhouette_score:",silhouette)

davies  = davies_bouldin_score(x, y)
print("davies_bouldin_score:",davies)

'''
#--------------EVALUACION EXTRINSECA--------------
homogeneo = metrics.homogeneity_score(true_labels.values.transpose()[0], y)
print(homogeneo)

completeness = metrics.completeness_score(true_labels.values.transpose()[0], y)
print(completeness)

vmeasure = metrics.v_measure_score(true_labels.values.transpose()[0], y)
print(vmeasure)
'''