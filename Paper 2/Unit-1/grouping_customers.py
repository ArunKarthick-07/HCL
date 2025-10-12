import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(42)
data = pd.DataFrame({
    'Age': np.random.randint(18, 70, 200),
    'Annual Income': np.random.randint(15000, 150000, 200),
    'Spending Score': np.random.randint(1, 100, 200)
})

X = data.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)
labels = kmeans.labels_

data['Cluster'] = labels

plt.scatter(data['Annual Income'], data['Spending Score'], c=labels, cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segments')
plt.show()

print(data.head())