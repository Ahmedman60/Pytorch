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

'''L2 Normalization vs. L1 Normalization
When you use Normalizer(copy=False) in scikit-learn with default parameters, it performs L2 normalization (Euclidean normalization), not L1 normalization. This means:

Each vector is scaled to have a Euclidean norm (L2 norm) of 1, not a sum (L1 norm) of 1.
# Normalize the TF-IDF vectors
normalizer = Normalizer(copy=False)
tfidf_matrix_normalized = normalizer.fit_transform(tfidf_matrix)
np.linalg.norm(tfidf_matrix_normalized.toarray(),ord=2,axis=1) #correct
The sum of the squares of the vector's elements will equal 1, not the sum of the elements themselves.
'''
# Perform K-Means clustering on one feature now. which is the normalization of vectors
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(tfidf_matrix_normalized)

# Print the clusters
for i, label in enumerate(kmeans.labels_):
    print(f"Document {i} is in cluster {label}")
