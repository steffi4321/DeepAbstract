import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #for no tensorflow warnings"

from src.models import *
from src.clustering import Cluster_Class

def get_params(model, epsilon=0.01, cl_method="kmeans", verbose=False):
    ss = Search_Space(model, epsilon=epsilon)
    params, dic = ss.perform_binarySearch(verbose=verbose, isglobal=True, method=cl_method)
    return params, dic

class Search_Space:
    def __init__(self, model, epsilon=0.01):
        """ 
        creates an element that performs the binarySearch

        Parameters
        ----------
        model : Keras_Model 
            object to do clustering on
        kmc : Cluster_Class 
            object that does the clustering for set parameters
        epsilon: int, optional
            abstracted network has minimum accuracy equal to original_acc-epsilon
        num_layers: int
            number layers that get clustered (total number layers -2)
        best_reduce: list of int
            saves the optimum cluster parameters per layer
        """
        self.model = model
        self.kmc = None   
        self.epsilon = epsilon
        self.num_layers = self.model.num_layers-2
        self.layer_sizes = get_layer_sizes(model)
        self.best_reduce = [0]*self.num_layers

    def do_cluster(self,cluster_num):
        self.kmc.model.reload()
        self.kmc.num_cluster = cluster_num
        print("cluster_num", cluster_num)
        acc = self.kmc.perform_clustering(verbose=False, variant="sum", bns=True)[0]
        return acc

    def for_each_layer(self,layer,verbose=False):
        if verbose:
            print("Layer ",layer)
            print('Layer size ',self.layer_sizes[layer])
        num,acc = self.binary_search(0,self.layer_sizes[layer],layer,verbose=verbose)
        if verbose:
            print("  --> We would cluster layer ",layer," with ", num)
        return num

    def binary_search(self, l, r, layer=None, acc=0, verbose=False):
        if verbose: print(" l: ",l," r:",r)
        if r>=1:
            if r == l:
                return r-1,acc
            elif l>r:
                return r,acc
            mid = int(l + (r-l)/2)

            if layer is None:
                cluster_num = [mid]
            else:
                cluster_num = [i for i in self.best_reduce]
                cluster_num[layer] = mid

            acc = self.do_cluster(cluster_num)

            if acc<self.goal:
                if verbose: print("   ", acc," < ",self.goal," --> smaller")
                if r == mid-1:
                    return mid,acc
                return self.binary_search(l,mid-1,layer,acc)
            elif acc>=self.goal:
                if verbose: print("   ", acc," > ",self.goal," --> bigger")
                if l == mid+1:
                    return mid,acc
                return self.binary_search(mid+1,r,layer,acc)
        else:
            return 0,0

    def perform_binarySearch(self, verbose=False, method="kmeans", isglobal=False, mP=2):
        starttime = time.time()
        
        self.kmc = Cluster_Class(self.model ,cl_method=method, minP=mP)

        before_layers = self.kmc.model.layers
        self.acc_orig = self.kmc.model.test_accuracy(False)
        self.goal = self.acc_orig-self.epsilon

        # global binSearch searches over all layers at once
        if isglobal:
            min_layersize = min(self.layer_sizes[:-1])
            res = self.binary_search(0,min_layersize)
            self.best_reduce = [res]
        else:
            for layer in range(self.num_layers-1,-1,-1):
                res = self.for_each_layer(layer, verbose=verbose)
                self.best_reduce[layer] = res
                    
        # print results:
        self.kmc.model.reload()
        self.kmc.num_cluster = self.best_reduce
        acc = self.kmc.perform_clustering(False, variant="sum", bns=True)[0]
        rrate = (np.sum(before_layers)-np.sum(self.kmc.model.layers))/np.sum(before_layers)
        endtime = time.time()
        if verbose:
            print("Accuracy after: %f" %acc)
            print("Clusters: ",self.best_reduce)
            print("Reduction-Rate: ",rrate)
            print("Time: ", endtime-starttime)
        dic = {}
        dic['acc'] = acc
        dic['rr'] = rrate
        dic['time']=endtime-starttime
        self.kmc.model.reload()
        return self.best_reduce, dic


if __name__ == "__main__":
    from src.main import set_file_model
    model = set_file_model('models/MNIST_6x100.tf')
    params, _ = get_params(model, cl_method="kmeans", verbose=True)
