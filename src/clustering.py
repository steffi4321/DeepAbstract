import time
import warnings
import numpy as np
from multiprocessing import Pool
from src.models import *
from sklearn.cluster import KMeans, SpectralClustering, MeanShift, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.exceptions import ConvergenceWarning
from src.cluster_methods import SensitivityAnalysis, RandomClustering

class Cluster_Class:
    """ Class for saving all important parameter for the clustering """
    def __init__(self, model, param_layer=[0],cl_method="kmeans",minP=2, backtoforth=False,
                    test_method =None):
        """ 
        creates an element that performs the clustering

        Parameters
        ----------
        model : src.models.Keras_Model 
            object to do clustering on
        param_layer : list of int, optional
            the parameter of clustering for each layer, for kmeans it is the number of clusters for each layer, for DBSCAN it is the distance value for the layer; if it is only one value, then this value is taken for all layers (default is [0])
        cl_method : str, optional
            either "kmeans", "dbscan", "hierarchical" or "meanshift", other clustering algorithms are so far not considered (default is 'kmeans')
        minP : int, optional
            parameter for DBSCAN, minimumpoints (default is 2)
        backtoforth : Boolean
            defines the order of the layers to be clustered
        test_method : cluster_methods.class
            used for testing new implementations. should implement a .fit()-method
        """
        self.model = model
        self.num_cluster = param_layer
        self.activations = None
        self.method = cl_method
        self.less_cluster = None
        self.model_type='keras'
        self.minPoints=minP
        self.backtoforth = backtoforth
        self.test_method = test_method
        return

    def set_model(self, model):
        self.model = model
        return
    
    def clustering_method(self, K, values, layer):
        """ this method is called with the datapoints and the parameter and performs the clustering """
        self.less_cluster = None
        if self.method == "kmeans":
            # performs the clustering with kmeans
            km = KMeans(n_clusters=K)
            w = None
            # the warning has to be fetched, such that the correct number of nodes can bepropagated
            with warnings.catch_warnings(record=True) as w:
                km = km.fit(values)
            if not (w is None or len(w)==0):
                if type(w[-1].message) == ConvergenceWarning:
                   self.less_cluster = np.shape(values)[0]-len(np.unique(km.labels_))
                   #print("Less Clusters than expected: ", self.less_cluster)
            labels = km.labels_
            cluster_centers = km.cluster_centers_
        elif self.method == "gm":
            # SO FAR NOT WORKING!!!
            # GM cannot work with such big data
            gm = GaussianMixture(K)
            labels = gm.fit_predict(values)
            cluster_centers = gm.means_
        elif self.method == "meanshift":
            ms = MeanShift(bandwidth=K).fit(values)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_
        elif self.method == "dbscan":
            num_nodes = np.shape(values)[0]
            db = DBSCAN(min_samples=self.minPoints,eps=num_nodes-K).fit(values)
            labels = db.labels_
            cluster_centers = []
            #print(len(np.unique(labels)), ' clusters found')
            for i in np.unique(labels):
                #print("Check for i: ",i)
                if i>-1:
                    cluster_centers.append(np.mean(values[np.where(labels==i)],axis=0))
        elif self.method == "hierarchical":
            # this would be imitating KMeans
            #ac = AgglomerativeClustering(n_clusters=K).fit(values)
            # this should be different
            ac = AgglomerativeClustering(compute_full_tree=True,distance_threshold=K,n_clusters=None,memory='cache').fit(values)
            labels = ac.labels_
            cluster_centers = []
            #print(len(np.unique(labels)), ' clusters found')
            for i in np.unique(labels):
                #print("Check for i: ",i)
                if i>-1:
                    cluster_centers.append(np.mean(values[np.where(labels==i)],axis=0))
        elif self.test_method is not None: 
            labels, cluster_centers = self.test_method.fit(K, values, layer)
        return labels, cluster_centers

    def get_activations(self, amount=1000, layers=[], save=False):
        """ 
        gets the activation-values for each node in each layer

        Parameters
        ----------
        amount : int, optional
           how many activation values shall be calculated (default is 1000)
        layers : list of int, optional
            all layers, for which activations shall be calculated, counting starts with 0 (default is [])
        """
        self.activations = self.model.get_activations(amount,layers,save)
        return self.activations
    
    def cluster_nodes_a(self, layer, reduce_size, best_node_selection=True, bias_shift=False):
        """
        performs the clustering for a specific layer based on the activations

        Parameters
        ----------
        layer : int
            defines which layer to cluster (lowest possible value is 1)
        reduce_size : int
            parameter for the clustering algorithm; number nodes to reduce
        best_node_selection : bool, optional
            decides which node to take as representative, random = False, choosing the node closest to the centroid = True (defaut is True)
        bias_shift : bool, optional
            decides whether to shift the bias towards the centroid (True) (default is False)
        """
        acti = self.activations['activations_l'+str(layer)]
        num_nodes = np.shape(acti)[1]
        K = num_nodes-reduce_size
        labels, cluster_centers = self.clustering_method(K, acti.transpose(), layer)
        num_clusters =  int(max(labels))
        clusters = []
        #for i in range(int(K)):
        for i in range(num_clusters+1): # TODO: keep this?
            try:
                cluster = np.where(labels==i)[0]
                if len(cluster)>1:
                    if best_node_selection:
                        myc = cluster_centers[i]
                        myc = myc.reshape(1,np.shape(myc)[0])
                        closest, _ = pairwise_distances_argmin_min(myc, acti[:,cluster].transpose())
                        best_node = cluster[closest[0]]
                        if bias_shift:
                            #print(np.shape(myc))
                            #print(np.shape(acti[:,closest[0]].transpose()))
                            bs = np.mean(acti[:,closest[0]].transpose()-myc)
                            #print(bs)
                        else:
                            bs=0
                        diameter = np.max( pairwise_distances(acti[:,cluster].transpose()) )
                        clusters.append((i,cluster,best_node,bs, diameter))
                    else:
                        clusters.append((i,cluster))
            except Exception as e:
                print(e)
                print("Layer: ",layer)
                print("labels: ",labels)
                print("cluster centers: ",cluster_centers)
                print("K: ", K)
        return clusters
 
    def cluster_a(self, layer, reduce_size, variant, bns=True, verbose=True, bias_shift=False):
        if verbose:
            print("----- [Get Clusters] -----")
        clusters = self.cluster_nodes_a(layer, reduce_size, best_node_selection=bns, bias_shift=bias_shift)
        if verbose:
            print(clusters)
        if len(clusters)==0:
            return
        if verbose:
            print("----- [Apply the Clustering to the Network] -----")
        self.apply_clustering(clusters, layer, variant)
        return clusters

    def get_delete_nodes(self, clusters, layer, variant, comp=False):
        delete_nodes = []
        for cluster in clusters:
            nodes = list(cluster[1])
            try:
                best_node=cluster[2]
            except:
                best_node = cluster[1][0]
            try:
                bias_shift=cluster[3]
            except:
                bias_shift=0
            if comp:
                weights = self.model.params['W'+str(layer)]
                new_weights = np.mean(weights[:,nodes], axis=1)
                weights[:,best_node] = new_weights
                self.model.params['W'+str(layer)] = weights
            bias = self.model.params['b'+str(layer)]
            n_weights = self.model.params['W'+str(layer+1)]
            #n_bias = self.model.params['b'+str(layer+1)]
            if variant == "mean":
                #new_bias = np.mean(n_bias[nodes])
                new_weights = np.mean(n_weights[nodes,:], axis=0)
            else:
                try:
                    #new_bias = np.sum(n_bias[nodes])
                    new_weights = np.sum(n_weights[nodes,:], axis=0)
                except Exception as e:
                    print(e)
                    print("Nodes: ", nodes)

            #n_bias[best_node] = new_bias
            n_weights[best_node,:] = new_weights
            indices = list(range(len(nodes)))
            try:
                indices.remove(nodes.index(best_node))
            except:
                print("Error with best_node ", best_node)
                print("not in ", nodes)
                exit(1)
            delete_nodes.extend([nodes[i] for i in indices])
            bias[best_node] = bias[best_node] - bias_shift
        delete_nodes.sort()
        self.model.params['b'+str(layer)] = bias
        #self.model.params['b'+str(layer+1)] = n_bias
        self.model.params['W'+str(layer+1)] = n_weights
        return delete_nodes

    def apply_clustering(self, clusters, layer, variant, comp=False):
        # the nodes in the 'layer' are clustered in 'clusters'
        delete_nodes = self.get_delete_nodes(clusters, layer, variant, comp)
        self.delete_nodes_layer(layer, delete_nodes)
        return

    def delete_nodes_layer(self, layer, delete_nodes):
        delete_nodes.sort()
        weights = self.model.params['W'+str(layer)]
        #print('layer',layer,'weights',np.shape(weights))
        bias = self.model.params['b'+str(layer)]
        n_weights = self.model.params['W'+str(layer+1)]
        for i,node in enumerate(delete_nodes):
            weights = np.delete(weights, node-i, 1)
            bias = np.delete(bias, node-i, 0)
            n_weights = np.delete(n_weights, node-i,0)
        self.model.params['W'+str(layer)] = weights
        #print("Weights ",np.shape(weights))
        self.model.params['b'+str(layer)] = bias
        self.model.params['W'+str(layer+1)] = n_weights
        self.model.layers[layer-1] = np.shape(weights)[1]
        return
   
    def clustering_on_activations(self, verbose=True, variant="mean", bns=True, bias_shift=False):
        """
        performs the clustering based on activation values

        Parameters
        ----------
        verbose : bool, optional
            switches on and off the verbose mode (defaul is True)
        variant : str, optional
            switches the two merging methods, 'sum' or 'mean' (default is 'sum')
        bns : bool, optional
            decides whether a random node shall be chosen as representative for a cluster (False), or the best node (True) (default is True)
        bias_shift : bool, optional
            shift the bias towards the cluster centroid (True), else no shifting (default is False)
        """
        start = time.time()
        # set the dense layers that can be clustered
        la_to_c = self.model.dense_layers
        # load activations always in case they were already reduced before
        if verbose:
            print("----- [Get Activations] -----")
        self.get_activations(amount=1000,layers = la_to_c, save=False)
        if verbose:
            print("----- [Start Clustering] -----")
        before = 0
        after = 0
        list_to_cluster = []
        all_cs = {}
        if self.backtoforth:
            list_to_cluster = reversed(list(enumerate(la_to_c)))
        else:
            list_to_cluster = enumerate(la_to_c)
        for i,j in list_to_cluster:
            if not len(self.num_cluster) == len(la_to_c):
                num = self.num_cluster[0]
            else:
                num = self.num_cluster[i]
            if num == 0:
                continue
            if verbose:
                print("   - Layer %d" %j)
            before = before + np.shape(self.model.params['W'+str(j)])[1]
            cls_ = self.cluster_a(j, num,variant, bns, verbose, bias_shift)
            all_cs[j] = cls_
            after = after + np.shape(self.model.params['W'+str(j)])[1]
            self.model.update_keras()
            self.get_activations(amount=1000,layers=la_to_c, save=False)
            num = None
            #self.model.set_params()
            #print('Layer sizes ', self.model.layers)
        name = "Network_clustered"
        #print("Model layers", self.model.layers)
        self.model.update_keras()
        ende = time.time()
        if verbose:
            print("Overall Time: ", (ende-start))
        acc = self.model.test_accuracy(verbose)
        if verbose:
            print("----- [End Clustering] -----")
        if before > 0:
            rrel = (before-after)/before
        else:
            rrel = 0
        return acc, rrel, all_cs


    def get_cluster_diameters(self, clusters):
        """
        TODO: description
        method is used in proof_lifting

        Parameters
        ----------
        clusters : dict, int -> list of 5-tuples
            maps layer_num to its clusters
        """
        diameter = []
        for i in range(1, self.model.num_layers + 1):
            if i not in clusters.keys():
                diameter.append(0)
                continue
            lnodes = np.shape(self.model.params['b' + str(i)])[0]
            newd = np.zeros(lnodes)
            deleted = self.get_deleted(clusters[i])
            for cl in clusters[i]:
                ind = cl[2] - (self.lower(deleted, cl[2]))
                newd[ind] = cl[4]
            diameter.append(newd)
        return diameter

    def get_deleted(self, clusters):
        """ used for get_cluster_diameters """
        alldel = []
        for cl in clusters:
            dell = [i for i in cl[1] if i!= cl[2]]
            alldel.extend(dell)
        alldel.sort()
        return alldel

    def lower(self, liste, val):
        """ used for get_cluster_diameters """
        num = len([i for i in liste if i<val])
        return num

    def perform_clustering(self, verbose=False, variant="sum", bns=True,
                           bias_shift=False, with_clusters=False):
        """
        runs the clustering

        Parameters
        ----------
        verbose : bool, optional
            turns on or off the verbose-mode (default is True)
        variant : str, optional
            switches the two merging methods, 'sum' or 'mean' (default is 'sum')
        bns : bool, optional
            decides whether a random node shall be chosen as representative for a cluster (False), or the best node (True) (default is True)
        bias_shift : bool, optional
            shift the bias towards the cluster centroid (True), else no shifting (default is False)
        """
        start = time.time()
        if self.model is None:
            print("You have to define the model first!!!")
            return
        before_layers = np.sum(self.model.layers)
        self.model.test_accuracy(verbose)
        rr_rel = 0
        
        #clustering:
        acc, rr_rel, all_c = self.clustering_on_activations(verbose, variant, bns, bias_shift)
        
        after_layers = np.sum(self.model.layers)
        ende = time.time()

        dic = {}
        dic['rr'] = 1-np.sum(after_layers)/np.sum(before_layers)
        dic['time']=ende-start
        dic['rr_rel'] = rr_rel

        if with_clusters:
            return acc, dic, all_c

        return acc, dic

