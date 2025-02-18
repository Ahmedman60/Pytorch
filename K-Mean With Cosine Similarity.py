from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
import numpy as np

# Sample documents
documents = [
    "I love machine learning",
    "Artificial intelligence is fascinating",
    "I enjoy hiking and nature",
    "Deep learning is a subset of machine learning",
    "I love to travel and explore new places"
]

# Convert documents to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Normalize the TF-IDF vectors
normalizer = Normalizer(copy=False)
tfidf_matrix_normalized = normalizer.fit_transform(tfidf_matrix)

# Perform K-Means clustering
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(tfidf_matrix_normalized)

# Print the clusters
for i, label in enumerate(kmeans.labels_):
    print(f"Document {i} is in cluster {label}")