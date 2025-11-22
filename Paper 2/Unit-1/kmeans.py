import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

iris = load_iris()
X = iris.data

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

print("K-means:")
print("Inertia (within-cluster SSE):", kmeans.inertia_)
print("Silhouette Score:", silhouette_score(X, kmeans_labels))


agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg.fit_predict(X)

print("\nHierarchical (Agglomerative):")
print("Silhouette Score:", silhouette_score(X, agg_labels))
