import json
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

X = 50


def kmeans_cluster_and_evaluate(data_file):
    print('starting kmeans clustering and evaluation with', data_file)
    df = pd.read_csv(data_file, sep='\t')
    numpyData = df.to_numpy()

    sentences = numpyData[:, 1]
    trueLabels = numpyData[:, 0]
    numOfClusters = len(set(trueLabels))
    print("Number of clusters: ", numOfClusters)

    # Feature extraction from sentences:

    # List of reviews to feature vectors (builds a dictionary of features)
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(sentences)

    # Occurences => tf-idf (instead of the raw frequencies)
    # tf-idf:  scale down the impact of tokens that occur very frequently in a given corpus
    # and that are hence empirically less informative than features that occur in a small fraction of the training corpus.
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    mean_RI_X_times_score = 0.0
    mean_ARI_X_times_score = 0.0

    for i in range(0, X):
        cluster = KMeansPlusPlus(X_tfidf.toarray(), numOfClusters)  # cluster = {label0: label, label1: label, ...}
        predLabels = list(cluster.values())

        # RI evaluation = (number of agreeing pairs) / (number of pairs)
        curr_mean_RI_score = rand_score(trueLabels, predLabels)
        mean_RI_X_times_score += curr_mean_RI_score
        # ARI evaluation = (RI - Expected_RI) / (max(RI) - Expected_RI)
        curr_mean_ARI_score = adjusted_rand_score(trueLabels, predLabels)
        mean_ARI_X_times_score += curr_mean_ARI_score

    mean_RI_score = mean_RI_X_times_score / X
    mean_ARI_score = mean_ARI_X_times_score / X

    evaluation_results = {'mean_RI_score': mean_RI_score, 'mean_ARI_score': mean_ARI_score}

    return evaluation_results


def euclideanDistance(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)


def KMeansPlusPlusInit(X, K):
    # choose one center uniformly at random among the data points
    centroids = [X[np.random.randint(0, len(X))]]

    # repeat until K centers have been chosen
    while len(centroids) < K:
        # for each data point x compute squared distance between x and the nearest centroid
        for dataPoint in X:
            for centroid in centroids:
                minDistances = np.array([min([np.square(euclideanDistance(dataPoint, centroid, None))])])

        # calc weighted probability
        probability = (minDistances / minDistances.sum()).cumsum()
        random = np.random.random()
        index = np.where(probability >= random)[0][0]
        centroids.append(X[index])

    return np.array(centroids)


def KMeansPlusPlus(X, K, maxIterations=300):
    centroids = KMeansPlusPlusInit(X, K)

    clusters = {}  # constant updating clusters

    for i in range(1, maxIterations):
        # for each data point: find the nearest centroid and assign data point to that cluster
        for i in range(len(X)):
            distances = euclideanDistance(X[i], centroids)
            cluster = np.argmin(distances)
            clusters[i] = cluster

        # changing old centroids value
        oldCentroids = np.copy(centroids)
        # for each cluster, find the new centroid (mean of all points assigned to that cluster
        for i in range(K):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            if points:
                centroids[i] = np.mean(points, axis=0)

        convergenceError = euclideanDistance(centroids, oldCentroids, None)
        if convergenceError == 0:
            break

    return clusters


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = kmeans_cluster_and_evaluate(config['data'])

    for k, v in results.items():
        print(k, v)
