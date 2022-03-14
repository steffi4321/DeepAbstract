import argparse
import numpy as np
from src.clustering import Cluster_Class
from src.models import Keras_Model
from src.clusterparam_search import get_params
#from src.proof_lifting import get_eran_bounds

DESCRIPTION = "DeepAbstract-Tool in development"
    
def set_keras_model(model):
    """ sets the model, on which the clustering is performed, by using a Keras_Model """
    km = Keras_Model()
    km.model = model
    km.set_params()
    return km

def set_file_model(netname, dataset="MNIST", input_dim = 28*28):
    """ sets the model on which the clustering shall be performed, by using a model that is stored in a file """
    if "CIFAR" in netname or dataset =='CIFAR':
        model = Keras_Model(filename=netname, input_shape=(32,32,3), dataset='CIFAR')
        model.set_params()
    elif "GTSRB" in netname or dataset == 'GTSRB':
        ini = int(np.sqrt(input_dim))
        model = Keras_Model(filename=netname, input_shape=(ini,ini,3), dataset='GTSRB')
        model.set_params()
    else:
        model = Keras_Model(filename=netname)
        model.set_params()
    return model

def main(args):
    model = set_file_model('models/MNIST_6x100.tf')
    acc_pre = model.test_accuracy(args.verbose)
    print(f'Model loaded (accuracy: {acc_pre}).')

    # abstraction
    params, _ = get_params(model, cl_method="kmeans", verbose=False)
    print(f'Optimum parameters found (params: {params}).')
    cc = Cluster_Class(model, params, cl_method="kmeans")
    acc_post, dic = cc.perform_clustering(verbose = False)
    print(acc_post, dic)

    # verification on abstracted net
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("model", type=str, help="model filename")
    parser.add_argument("-v", "--verbose", type=bool, help="enable verbosity")
    args = parser.parse_args()
    main(args)