import os
import sys
import tensorflow as tf

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(current_path, "../../framework/eran"))
sys.path.insert(1, os.path.join(current_path, "../../framework/eran/tf_verify"))
sys.path.insert(1, os.path.join(current_path, "../../framework/eran/ELINA"))
sys.path.insert(1, os.path.join(current_path, "../../framework/eran/ELINA/python_interface"))

from src.clustering import *
from src.models import *
from src.main_eran import mmain
from src.main import set_file_model #TODO: this dependency should be removed 
from tf_verify.config import config
from shutil import copyfile # from eran as well


class ErrorCalculation:

    def __init__(self, weight_matrices, biases, upperbounds, lowerbounds, diameter, inputsize):
        """ 
        creates an element that calculates error bounds 

        Parameters
        ----------
        upperbounds: list
        lowerbounds: list
            u,l are lists of length 2*(num_layers+1) with inner lists length #neurons with min-dist i from input-layer

        """
        self.weight_matrices = weight_matrices
        self.biases = biases
        self.upperbounds = upperbounds
        self.lowerbounds = lowerbounds
        self.diameter = diameter
        self.inputsize = inputsize
        self.lhats = dict()
        self.uhats = dict()
        self.lhats[0] = self.lowerbounds[0]
        self.uhats[0] = self.upperbounds[0]
        return

    def diam(self, l):
        if l == 0:
            val = np.zeros(self.inputsize)
        else:
            val = self.diameter[l - 1]
        return val

    def layersize(self, l):
        bias = self.biases[l + 1]
        return np.shape(bias)[0]

    def uhat(self, l):
        if l in self.uhats.keys():
            return self.uhats[l]
        try:
            wplus = np.maximum(0, self.weight_matrices[l - 1])
            wminus = np.minimum(self.weight_matrices[l - 1], 0)

            uhlm = self.uhat(l - 1)
            val = wplus.T.dot(uhlm + self.diam(l - 1))
            val += wminus.T.dot(self.lhat(l - 1) - self.diam(l - 1))
            val += self.biases[l - 1]
            val = np.maximum(0, val)
            self.uhats[l] = val
        except Exception as e:
            print(e)
            print('uhat l-1',np.shape(uhlm))
            print('wplus', np.shape(wplus.T))
            print('diam', np.shape(self.diam(l-1)))
        return val

    def utilde(self, l):
        return self.upperbounds[2*l-1]

    def lhat(self, l):
        if l in self.lhats.keys():
            return self.lhats[l]
        wplus = np.maximum(0, self.weight_matrices[l - 1])
        wminus = np.minimum(self.weight_matrices[l - 1], 0)

        lhlm = self.lhat(l - 1)
        val = wminus.T.dot(self.uhat(l - 1) + self.diam(l - 1))
        val += wplus.T.dot(lhlm - self.diam(l - 1))
        val += self.biases[l - 1]
        val = np.maximum(0, val)
        self.lhats[l] = val
        return val

    def ltilde(self, l):
        return self.lowerbounds[2*l-1]

    def error_u(self, l):
        if l == 0:
            error_l = np.zeros(self.inputsize)
        else:
            error_l = self.uhat(l) - self.utilde(l)
        return error_l

    def error_l(self, l):
        if l == 0:
            error_l = np.zeros(self.inputsize)
        else:
            error_l = self.ltilde(l) - self.lhat(l)
        return error_l

def verify_original(network_location, epsilon, num_tests=999):
    setattr(config, 'netname', network_location)
    setattr(config, 'domain', 'deeppoly')
    setattr(config, 'dataset', 'mnist')
    setattr(config, 'epsilon', epsilon)
    setattr(config, 'num_tests', num_tests)
    model = set_file_model(network_location)
    ind, dicti = mmain(verbose=False, dir_prefix=get_dirPrefix(model, []))
    return dicti

def get_dirPrefix(model, cluster_params):
    """
        returns the directory tmp/MNIST_nxm_a-b-c with n=num_layers, m=neurons_per_layer, abc=cluster_params
    """
    mname = model.dataset + "_" + str(model.num_layers) + "x" + str(model.layers[len(model.layers)-2]) + "_"

    clname = ""
    for i in cluster_params:
        clname += (str(i) + '-')
    clname = clname[:-1]
    if not cluster_params:
        clname = "orig"

    storename = 'tmp/'+ mname + clname 
    return storename

def get_weight_matrices(km):
    weights = []
    biases = []
    for la in km.model.layers:
        we = la.get_weights()
        if len(we)>0:
            weights.append(we[0])
            biases.append(we[1])
    return weights, biases

def cluster_net(model, cluster_params=[0], cl_method='kmeans'):
    weight_matrices, bias_matrices = get_weight_matrices(model)
    cc = Cluster_Class(model, param_layer=cluster_params, cl_method=cl_method)

    _, dic, clusters = cc.perform_clustering(verbose=False, with_clusters=True)

    # get all diameters for each neuron
    diameter = cc.get_cluster_diameters(clusters)

    # get new weight matrices
    new_weight_matrices, bias_matrices = get_weight_matrices(cc.model)
    return (clusters, diameter, dic['time'])

def get_bounds(lowerBounds, upperBounds, testImages):
    l_bounds = np.load(lowerBounds, allow_pickle=True)
    u_bounds = np.load(upperBounds, allow_pickle=True)
    if os.path.isfile(testImages):
        images = np.load(testImages, allow_pickle=True)
    else:
        images = []
    return (l_bounds, u_bounds, images)


def perform_evaluation_complete(cluster_params, epsilon, verbose=False, numImages=500):
    """
        runs the clustering, then calculates error bounds from eran

        Parameters
        ----------
        network_name : str
            gets name of network or path to network relative to new_structure/
        epsilon: float
            accuracy for ERAN
        numImages: int, optional
            number of images used to verify in ERAN
    """
    model = set_file_model('models/MNIST_6x50.tf')
    dir_prefix = get_dirPrefix(model, cluster_params)
    
    clusters, diameter, cl_time = cluster_net(model, cluster_params)
    new_weight_matrices, new_bias_matrices = get_weight_matrices(model)

    model.save_for_eran(dir_prefix+'.tf')

    setattr(config, 'netname', dir_prefix + '.tf')
    setattr(config, 'domain', 'deeppoly')
    setattr(config, 'dataset', 'mnist')
    setattr(config, 'epsilon', epsilon)
    setattr(config, 'num_tests', numImages)

    # execute ERAN 
    ind, dicti = mmain(verbose=False, dir_prefix=dir_prefix)

    #get bounds calculated by ERAN
    lb = dir_prefix + "_lower-bounds-" + str(epsilon) + "_" + str(ind) + ".npy"
    ub = dir_prefix + "_upper-bounds-" + str(epsilon) + "_" + str(ind) + ".npy"
    im = dir_prefix + "_images-" + str(epsilon) + "_" + str(ind) + ".npy"
    l_bounds, u_bounds, images = get_bounds(lb, ub, im)

    mstart = time.time()
    verified = 0
    veriIm = []

    for i in range(len(images)):
        if verbose:
            print("Image", i)
        x = images[i]
        u = u_bounds[i]
        l = l_bounds[i]
        u = np.hstack(([0], u))
        u[0] = np.clip(x + np.zeros_like(x) * epsilon, 0, 1) 
        l = np.hstack(([0], l))
        l[0] = np.clip(x - np.zeros_like(x) * epsilon, 0, 1)

        EC = ErrorCalculation(new_weight_matrices, new_bias_matrices, u, l, diameter, 784)
        del_u_acc = EC.error_u(7)
        del_l_acc = EC.error_l(7)
        
        index = np.argmax(model.model.predict(x.reshape(1, 28, 28, 1)))
        
        if verbose:
            print('  lower bound true', index, (l[-1] + del_l_acc)[index])
            print('  upper bounds', (u[-1] + del_u_acc))
        if ((l[-1] + del_l_acc)[index] > (u[-1] + del_u_acc)[[i for i in range(len(u[-1])) if i != index]]).all():
            verified += 1
            veriIm.append(0)
            if verbose:
                print(f"  verified {index}")
        else:
            veriIm.append(1)
    mende = time.time()
    veri_time = dicti['time'] + mende-mstart
    os.remove(dir_prefix+'.tf')
    return verified, veriIm, cl_time, veri_time, dicti
