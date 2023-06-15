import metispy as metis
from ClusterView.densityCanopy import MyCanopy
from sklearn.cluster import KMeans

def metisClustering(graph, clusterNumber):
    (edgecuts, parts) = metis.part_graph(graph, nparts=clusterNumber)
    return parts

def densityCanopyClustering(data_):
    canopy = MyCanopy()
    canopy.fit(data_)
    centroids = [canopy.centroids[i] for i in range(len(canopy.centroids))]
    kmeans = KMeans(n_clusters = len(centroids), init=centroids).fit(data_)
    parts = list(kmeans.labels_)
    return parts, len(centroids)