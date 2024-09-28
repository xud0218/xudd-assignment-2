import numpy as np
import random

class KMeans:
    def __init__(self, k=3, init_method='random', max_iters=100, tol=1e-4):
        self.k = k
        self.init_method = init_method
        self.max_iters = max_iters
        self.tol = tol  # Tolerance value for convergence
        self.centroids = None  # Store the centroids after initialization
        self.converged = False

    def initialize_centroids(self, X):
        if self.init_method == 'random':
            centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        elif self.init_method == 'farthest':
            centroids = self.farthest_first_initialization(X)
        elif self.init_method == 'kmeans++':
            centroids = self.kmeans_plus_plus_initialization(X)
        return centroids

    def fit(self, X):
        # Initialize centroids
        centroids = self.initialize_centroids(X)

        for i in range(self.max_iters):
            # Assign clusters
            clusters = self.assign_clusters(X, centroids)
            
            # Recompute centroids
            new_centroids = self.recalculate_centroids(X, clusters)

            # Check for convergence: if centroids do not move significantly
            centroid_shift = np.linalg.norm(new_centroids - centroids)
            if centroid_shift < self.tol:
                print(f"Converged after {i+1} iterations with shift {centroid_shift}")
                break

            centroids = new_centroids

        self.centroids = centroids
        return clusters, centroids

    def assign_clusters(self, X, centroids):
        # Calculate distance between each point and centroids, assign points to nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def recalculate_centroids(self, X, clusters):
        # Recalculate centroids based on mean of points assigned to each cluster
        centroids = np.array([X[clusters == i].mean(axis=0) for i in range(self.k)])
        return centroids

    def step_through(self, X):
        if self.centroids is None:
            self.centroids = self.initialize_centroids(X)

        if self.converged:
            return {'clusters': None, 'centroids': self.centroids.tolist(), 'converged': True}

        # Assign clusters
        clusters = self.assign_clusters(X, self.centroids)
        # Recompute centroids
        new_centroids = self.recalculate_centroids(X, clusters)

        # Check for convergence
        centroid_shift = np.linalg.norm(new_centroids - self.centroids)
        self.centroids = new_centroids

        if centroid_shift < self.tol:
            self.converged = True

        return {'clusters': clusters.tolist(), 'centroids': new_centroids.tolist(), 'converged': self.converged}

    def farthest_first_initialization(self, X):
        centroids = [random.choice(X)]
        for _ in range(1, self.k):
            distances = np.array([min(np.linalg.norm(x - c) for c in centroids) for x in X])
            next_centroid = X[np.argmax(distances)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def kmeans_plus_plus_initialization(self, X):
        centroids = [random.choice(X)]
        for _ in range(1, self.k):
            distances = np.min([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
            probs = distances / distances.sum()
            next_centroid = X[np.random.choice(X.shape[0], p=probs)]
            centroids.append(next_centroid)
        return np.array(centroids)
