import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Data: Annual Income (x) and Spending Score (y)
data = np.array([
    [15, 10], [18, 12], [20, 15],  # Cluster 1 (Low Income, Low Spending)
    [60, 80], [62, 85], [65, 90],  # Cluster 2 (High Income, High Spending)
    [30, 30], [32, 28], [70, 10], [75, 12]  # Cluster 3 (Moderate Income)
])

# Applying K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(data)
labels = kmeans.labels_  # Cluster labels for each point
centroids = kmeans.cluster_centers_  # Cluster center points

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis',
            edgecolors='black', s=150, label="Customers")
plt.scatter(centroids[:, 0], centroids[:, 1], c='red',
            marker='X', s=300, label="Centroids")  # Mark centroids

# Labels and title
plt.xlabel("Annual Income ($1000s)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Clustering of Bank Customers")
plt.legend()
plt.grid(True)
plt.show()
