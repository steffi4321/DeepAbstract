from sklearn.cluster import DBSCAN
import numpy as np
import random

"""
    The classes are used for testing new methods.
    it should implement a fit() method returning labels and cluster_centers
"""

class SensitivityAnalysis:
    def __init__(self, name="TestSensitivityAnalysis", distances=None, dist_list = None, eps=None, layer_wise = False, rrs=None):
        self.name = name
        self.labels_ = None             # of shape (n_samples,)
        self.cluster_centers_ = None    # of shape (n_clusters, n_features)
        self.distances = distances
        self.dist_list = dist_list
        self.eps = eps
        self.rrs = rrs

    def fit(self, K, values, layer):
        self.labels_ = (-1)*np.ones((values.shape[0],))
        self.cluster_centers_ = np.array([])
        (self.labels_, self.cluster_centers_) = self.fit_one(K, values, layer)
        return (self.labels_, self.cluster_centers_)
        
    def fit_many(self, K, values, layer):
        # TODO: this method is not finished for fit_many
        self.labels_ = (-1)*np.ones((values.shape[0],))
        self.cluster_centers_ = np.array([])

        if self.dist_list is None: # compute multiple clusterings and then merge afterwords
            (self.labels_, self.cluster_centers_) = self.fit_one(K, values, layer)
            return (self.labels_, self.cluster_centers_)
        
        for dist in dist_list:
            self.distances = dist
            (labels, cluster_centers) = self.fit_one(K, values, layer)  # returns of type np.array()
            clusters = []
            for i in np.unique(labels):
                cluster = np.where(labels == i)[0]
                if i >= 0 and len(cluster) > 1:
                    clusters.append(set(cluster))

        return (self.labels_, self.cluster_centers_)
    
    def fit_one(self, K, values, layer):
        if self.distances is None:
            print("DO NOTHING")
            return (self.labels_, self.cluster_centers_)

        dists = self.distances["d"+str(layer)]
        if self.rrs is None or self.rrs[layer] == 0.0:
            db = DBSCAN(min_samples=2,eps=self.eps[layer], metric='precomputed')
            labels = db.fit_predict(dists)
        else:
            labels = self.find_clustering(dists, self.rrs[layer])

        print("using d{0} with eps={1}".format(layer, self.eps[layer]))
        print("labels \n", [str(i)+":"+str(np.where(labels==i)[0].shape) for i in np.unique(labels)])
        print("num neurons reduced={0}".format(np.where(labels!=-1)[0].shape[0] - np.unique(labels).shape[0] +1))

        cluster_centers = []
        for i in np.unique(labels):
            if i>-1:
                cluster_centers.append(np.mean(values[np.where(labels==i)],axis=0))
        cluster_centers = np.array(cluster_centers)
        self.labels_ = labels
        self.cluster_centers_ = cluster_centers
        return (labels, cluster_centers)

    def find_clustering(self, dists, rr):
        N = dists.shape[0]
        #binary Search
        L = 0.0
        R = 2.0
        print('Binary Search started ...')
        while L <= R:
            m = (L+R)/2 
            db = DBSCAN(min_samples=2,eps=m, metric='precomputed')
            labels = db.fit_predict(dists)
            N_red = np.where(labels!=-1)[0].shape[0] - np.unique(labels).shape[0] +1
            rr_curr = float(N_red) / float(N)
            print('     eps = {0}, rr = {1}'.format(m, rr_curr))
            if rr_curr < (rr - 0.01):
                L = m
            elif rr_curr > (rr + 0.01):
                R = m 
            else:
                return labels
        print('... Unsuccessful')
        return (-1)*np.ones((values.shape[0],))
             

class RandomClustering:
    def __init__(self, name="TestRandomClustering", num_clusters = 3):
        self.name = name
        self.labels_ = None
        self.cluster_centers_ = None 
        self.num_clusters = num_clusters

    def fit(self, K, values):
        print("K=", K)
        num_nodes = values.shape[0]
        self.labels_ = (-1)*np.ones((num_nodes,))
        self.cluster_centers_ = np.array([])

        if K < num_nodes-1:
            print("sample ", num_nodes, K)
            clusters = random.sample(range(num_nodes), K)
            self.labels_[clusters] = np.random.randint(0, self.num_clusters, self.labels_[clusters].shape)

            cluster_centers = []
            for i in np.unique(self.labels_):
                if i>-1:
                    cluster_centers.append(np.mean(values[np.where(self.labels_==i)],axis=0))
            self.cluster_centers_ = np.array(cluster_centers)
        return (self.labels_, self.cluster_centers_)
        