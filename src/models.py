"""
Created on Tue May  7 09:51:17 2019

@author: Stefanie Muehlberger
"""

import numpy as np
import time
from src.data_utils import get_MNIST_data, get_CIFAR10_data, get_GTSRB_data, get_GTSRB_regression_data, get_ACASXU_data
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import model_from_json
import os

def str2list(v):
    if type(v) == str:
        layer_list = []
        v = v.replace('[','')
        v = v.replace(']','')
        v = v.split(';')
        for item in v:
            if item.startswith('Conv') or item.startswith('conv'):
                item = item.replace('(','').replace(')','')[4:]
                item = item.split(',')
                num_filters = int(item[0])
                filtersize = int(item[1])
                stride = int(item[2])
                padding = int(item[3])
                layer_list.append(('Conv2d',[num_filters,filtersize,stride,padding]))
            elif item.startswith('max'):
                item = item.replace('(','').replace(')','')[3:]
                layer_list.append(('MaxPool',item))
            elif item.startswith('drop'):
                item = item.replace('(','').replace(')','')[4:]
                layer_list.append(('Dropout', item))
            else:
                hidden_units = int(item)
                layer_list.append(('FC',[hidden_units]))
    elif type(v) == list:
        layer_list = []
        for item in v:
            layer_list.append(('FC',[item]))
    else:
        layer_list = []
        print("Please give String or List!")
    return layer_list

def show(model):
    for i,layer in enumerate(model.model.layers):
        we = layer.get_weights()
        if len(we)>0:
            print("ReLU")
            print(np.shape(we[0])[1])
    return

def get_layer_sizes(model):
    las = []
    for i,layer in enumerate(model.model.layers):
        we = layer.get_weights()
        if len(we)>0:
            las.append(np.shape(we[0])[1])
    return las

def runRepl(arg, repl):
    for a in repl:
        arg = arg.replace(a+"=", "'"+a+"':")
    return eval("{"+arg+"}")

def parseVec(net):
    return np.array(eval(net.readline()[:-1]))

class Keras_Model:
    def __init__(self, structure=None, filename=None, reg=0, input_shape=(28,28,1), type_reg=None, acti="relu", dataset='MNIST', regression=False, crown=False):
        self.input_shape=input_shape
        if isinstance(structure, list) and isinstance(structure[0], tuple):
            self.structure = structure
        elif isinstance(structure, list):
            self.structure = str2list(structure)
        elif not structure is None:
            raise TypeError("The format of the structure should be a list containing the sizes of the layers.")
        else:
            self.structure = None
        self.dataset = dataset
        self.regression=regression
        self.U = []
        self.W = []
        self.crown = crown
        self.num_layers = None
        self.filename=filename
        if not structure is None:
            self.num_layers = len(structure)
            self.get_model(structure, reg, type_reg, acti=acti)
        elif not filename is None:
            self.read_from_file(filename)
        return

    def set_params(self):
        self.params = {}
        self.layers = []
        self.dense_layers = []
        self.conv_layers = []
        self.flatten_layers = []
        self.num_layers = len(self.model.layers)-1 # -1 because of self.layers.pop below

        count = 1
        for i,layer in enumerate(self.model.layers):
            if layer.__class__.__name__ in ("Dense", "Conv2D"):
                we = layer.get_weights()
                self.params['W'+str(count)] = we[0]
                self.params['b'+str(count)] = we[1]
                count = count +1 
            if layer.__class__.__name__ == "Dense":
                self.layers.append(layer.output_shape[1])
                self.dense_layers.append(count-1)
            elif layer.__class__.__name__ == "Conv2D":
                self.conv_layers.append(count-1)
                self.layers.append(np.prod(np.shape(we[0]))+np.prod(np.shape(we[1])))
            elif layer.__class__.__name__ == "Flatten":
                self.flatten_layers.append(count-1)
                self.num_layers -=1
        self.layers.pop()
        self.dense_layers.pop()
        return

    def reload(self):
        '''reload keras model and Keras_Model.params e.g. from file'''
        if self.filename is not None:
            self.read_from_file(self.filename)
        else:
            print("reload not implemented for this type") #TODO: model derived by structure
        self.set_params()

    def get_activations(self, num_samples = 1000, layers=[], save=False):
        """
        Returns a dictionary {"activations_l<num>": acts} where acts is
        an array of shape (num_samples, size_of_layer_<num>).

        :param num_samples:
        :param layers:
        :param save:
        :return:
        """
        # Load datasets
        if self.input_shape == (28,28,1) and self.dataset == "MNIST":
            data = get_MNIST_data()
        elif self.input_shape == (32,32,3)  and self.dataset == "CIFAR":
            data = get_CIFAR10_data()
        elif self.dataset == "ACASXU":
            num_samples = 2
            data = get_ACASXU_data()
        elif self.regression:
            ini = self.input_shape[0]
            data = get_GTSRB_regression_data(sizes=(ini,ini))
        else:
            ini = self.input_shape[0]
            data = get_GTSRB_data(sizes=(ini,ini))
        X_train = np.squeeze(data['X_train'])
        if self.input_shape == (28,28,1):
            X_train = X_train.reshape(np.shape(X_train)[0],np.shape(X_train)[1],np.shape(X_train)[2],1)
        elif self.input_shape == (32,32,3) and self.dataset == "CIFAR":
            X_train = X_train.reshape(np.shape(X_train)[0],np.shape(X_train)[2],np.shape(X_train)[3],np.shape(X_train)[1])
        acts = {}
        num = 1
        if layers==[]:
            layers = list(range(len(self.model.layers)))
        #print(layers)
        for i,layer in enumerate(self.model.layers):
            #print(i,'th layer ', layer)
            #print(' num is ', num)
            if not layer.__class__.__name__ in ['Dense', 'Conv2D']:
                continue
            if not num in layers:
                num = num+1
                continue
            #print("Now we get the Activations!")
            get_layer_output = K.function([self.model.layers[0].input],
                                  [self.model.layers[i].output])
            if self.dataset == 'MNIST' and os.path.isfile("images_clustering.npy"):
                vals = (np.load("images_clustering.npy")[:,1:]).reshape(num_samples,28,28)/255.0
            else:
                vals = X_train[:num_samples,:]
            acts['activations_l'+str(num)] = get_layer_output([vals])[0]
            num = num +1
        if save:
            storage = np.hstack([data['y_train'][:num_samples].reshape(num_samples,1),X_train[:num_samples,:].reshape(num_samples,-1)*255])
            np.save('images_clustering', storage)
        return acts                            

    def test_accuracy(self, verbose=True):
        if self.regression:
            data = get_GTSRB_regression_data((self.input_shape[0],self.input_shape[1]))
            X_Test = data['X_test']
            y_test = data['y_test']
            self.model.compile(optimizer='adam',
                loss='mean_squared_error')
            acc = self.model.evaluate(X_Test,y_test, verbose=0)
            if verbose:
                print('Test set MSE: ', acc)
        else:    
            if self.input_shape == (28,28,1) and self.dataset == "MNIST":
                data = get_MNIST_data()
                X_Test = np.squeeze(data['X_test'])
                X_Test = X_Test.reshape(np.shape(X_Test)[0],np.shape(X_Test)[1], np.shape(X_Test)[2],1)
            elif self.input_shape == (32,32,3) and self.dataset == "CIFAR":
                data = get_CIFAR10_data()
                X_Test = data['X_test']
                X_Test = X_Test.reshape(np.shape(X_Test)[0],np.shape(X_Test)[2], np.shape(X_Test)[3], np.shape(X_Test)[1])
            elif self.dataset == 'ACASXU':
                data = get_ACASXU_data()
                X_Test = data['X_test']
            else:
                data = get_GTSRB_data((self.input_shape[0],self.input_shape[1]))
                X_Test = data['X_test']
            y_test = data['y_test']
            if self.dataset == 'ACASXU':
                self.model.compile(optimizer='adam',
                                   loss='mean_squared_error',
                                   metrics=['accuracy'])
            else:
                self.model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
            acc = self.model.evaluate(X_Test,y_test, verbose=0)[1]
            if verbose:
                print('Test set accuracy: ', acc)
        return acc

    def test_MNIST_labelacc(self, label):
        data = get_MNIST_data()
        idx = np.where(data['y_test'] == label)
        X_test = np.squeeze(data['X_test'][idx])
        X_test = np.expand_dims(X_test, axis=3)
        y_test = data['y_test'][idx]

        self.model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        acc = self.model.evaluate(X_test,y_test, verbose=0)[1]
        return acc
            

    def read_from_file(self, net_file):
        if net_file.endswith('tf'):
            self.read_from_file_eran(net_file)
        elif net_file.endswith('nnet'):
            self.read_from_file_nnet(net_file)
        else:
            self.read_from_file_json(net_file)
        return

    def read_from_file_json(self, net_name):
        json_file = open(net_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(net_name + ".h5")
        self.model = loaded_model
        self.get_structure()
        if self.crown:
            self.prep_for_crown()
        return

    def read_from_file_nnet(self, net_name):
        '''
        Read a .nnet file and return list of weight matrices and bias vectors
        Adapted from https://github.com/sisl/NNet/blob/master/utils/readNNet.py

        Inputs:
            nnetFile: (string) .nnet file to read
        '''

        # Open NNet file
        f = open(net_name, 'r')

        # Skip header lines
        line = f.readline()
        while line[:2] == "//":
            line = f.readline()

        # Extract information about network architecture
        record = line.split(',')
        numLayers = int(record[0])
        inputSize = int(record[1])

        line = f.readline()
        record = line.split(',')
        layerSizes = np.zeros(numLayers + 1, 'int')
        for i in range(numLayers + 1):
            layerSizes[i] = int(record[i])

        # Skip extra obsolete parameter line
        f.readline()

        # Read the normalization information
        line = f.readline()
        inputMins = [float(x) for x in line.strip().split(",") if x]

        line = f.readline()
        inputMaxes = [float(x) for x in line.strip().split(",") if x]

        line = f.readline()
        means = [float(x) for x in line.strip().split(",") if x]

        line = f.readline()
        ranges = [float(x) for x in line.strip().split(",") if x]

        # Initialize Keras model
        model = Sequential()

        # Read weights and biases and add to model
        for layernum in range(numLayers):
            previousLayerSize = layerSizes[layernum]
            currentLayerSize = layerSizes[layernum + 1]

            weights = np.zeros((currentLayerSize, previousLayerSize))
            for i in range(currentLayerSize):
                line = f.readline()
                aux = [float(x) for x in line.strip().split(",")[:-1]]
                for j in range(previousLayerSize):
                    weights[i, j] = aux[j]
            # biases
            biases = np.zeros(currentLayerSize)
            for i in range(currentLayerSize):
                line = f.readline()
                x = float(line.strip().split(",")[0])
                biases[i] = x

            # Add layer to model
            # TODO crown
            model.add(Dense(currentLayerSize, activation="relu", input_shape=(previousLayerSize,)))
            model.layers[-1].set_weights([weights.T, biases])

        f.close()

        self.model = model
        self.dataset = "ACASXU"
        self.input_shape = (inputSize,ks)
        self.get_structure()
        
    def prep_for_crown(self):
        self.layer_outputs = []
        for layer in self.model.layers:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                self.layer_outputs.append(K.function([self.model.layers[0].input], [layer.output]))
        return

    def read_from_file_eran(self, net_file):
        numl = 0
        model = Sequential()
        first = True
        firstFC = True
        last_layer = None
        net = open(net_file,'r')
        is_conv = False
        if self.structure is None:
            self.structure = []
        while True:
            curr_line = net.readline()[:-1]
            if ((curr_line == "ReLU") or (curr_line == "Sigmoid") or (curr_line == "Tanh") or (curr_line == "Affine")):
                W = None
                W = parseVec(net).transpose()
                b = parseVec(net)
                if first:
                    first = False
                    firstFC = False
                    model.add(Flatten(input_shape=self.input_shape))
                elif firstFC:
                    firstFC = False
                    model.add(Flatten())
                if(curr_line=="Affine"):
                    self.U.append(Dense(np.shape(b)[0]))
                    model.add(self.U[-1])
                    model.layers[-1].set_weights([W,b])
                elif(curr_line=="ReLU"):
                    if self.crown:
                        self.U.append(Dense(np.shape(b)[0]))
                        model.add(self.U[-1])
                        model.layers[-1].set_weights([W,b])
                        model.add(Activation("relu"))
                    else:
                        model.add(Dense(np.shape(b)[0], activation="relu"))
                        model.layers[-1].set_weights([W,b])
                elif(curr_line=="Sigmoid"):
                    model.add(Dense(np.shape(b)[0], activation="sigmoid"))
                    model.layers[-1].set_weights([W,b])
                else:
                    model.add(Dense(np.shape(b)[0], activation="tanh"))
                    model.layers[-1].set_weights([W,b])
                self.structure.append(('FC',[np.shape(b)[0]]))
                numl = numl + 1
            elif curr_line == "MaxPooling2D":
                args = runRepl(net.readline()[:-1], ["input_shape" , "pool_size"])
                ksize = args['pool_size']
                model.add(MaxPooling2D(pool_size=ksize))
                self.structure.append(('MaxPool',ksize))
            elif curr_line == "Conv2D":
                is_conv = True
                line = net.readline()
                args = None
                start = 0
                act = ""
                if("ReLU" in line):
                    start = 5
                    act="relu"
                elif("Sigmoid" in line):
                    start = 8
                    act="sigmoid"
                elif("Tanh" in line):
                    start = 5
                    act="tanh"
                if 'padding' in line:
                    args =  runRepl(line[start:-1], ["filters", "input_shape", "kernel_size", "stride", "padding"])
                else:
                    args = runRepl(line[start:-1], ["filters", "input_shape", "kernel_size"])

                W = parseVec(net)
                b = parseVec(net)
                if("padding" in line):
                    if(args["padding"]==1):
                        padding_arg = "same"
                    else:
                        padding_arg = "valid"
                else:
                    padding_arg = "valid"

                if("stride" in line):
                    stride_arg = (args["stride"][0],args["stride"][1])
                else:
                    stride_arg = (1,1)
                ks = (np.shape(W)[0],np.shape(W)[1])
                if first:
                    first = False
                    model.add(Conv2D(filters=np.shape(W)[3], kernel_size=ks, strides=stride_arg, padding=padding_arg, activation=act, input_shape=self.input_shape))
                else:
                    model.add(Conv2D(filters=np.shape(W)[3], kernel_size=ks, strides=stride_arg, padding=padding_arg, activation=act))
                model.layers[-1].set_weights([W,b])
                self.structure.append(('Conv2d',[np.shape(W)[3],ks[0],stride_arg[0],args["padding"]]))
                numl = numl+1
            elif curr_line == "":
                self.W = model.layers[-1]
                break
            else:
                continue
            
        net.close()
        self.U = self.U[:-1]
        self.model = model
        self.num_layers = numl -1
        if self.crown:
            self.prep_for_crown()
        return

    def predict(self, data):
        return self.model(data)

    def get_model(self, structure, reg, type_reg, acti = "relu"):
        first_fc = True
        first = True
        model = Sequential()
        for i,(layer,var) in enumerate(structure):
            if layer == 'Conv2d':
                num_filters = var[0]
                filtersize = var[1]
                stride = var[2]
                if var[3] == 0:
                    padding = 'valid'
                else:
                    padding = 'same'
                if first:
                    first = False
                    if type_reg == "l1":
                        model.add(Conv2D(num_filters, kernel_size=filtersize, activation=acti, input_shape=self.input_shape, kernel_regularizer = tf.keras.regularizers.l1(reg), strides=stride, padding=padding))
                    elif type_reg == "l2":
                        model.add(Conv2D(num_filters, kernel_size=filtersize, activation=acti, input_shape=self.input_shape, kernel_regularizer = tf.keras.regularizers.l2(reg), strides=stride, padding=padding))
                    else:
                        model.add(Conv2D(num_filters, kernel_size=filtersize, activation=acti, input_shape=self.input_shape, strides=stride, padding=padding))
                else:
                    if type_reg == "l1":
                        model.add(Conv2D(num_filters, kernel_size=filtersize, activation=acti, kernel_regularizer = tf.keras.regularizers.l1(reg), strides=stride, padding=padding))
                    elif type_reg == "l2":
                        model.add(Conv2D(num_filters, kernel_size=filtersize, activation=acti, kernel_regularizer = tf.keras.regularizers.l2(reg), strides=stride, padding=padding))
                    else:
                        model.add(Conv2D(num_filters, kernel_size=filtersize, activation=acti,strides=stride, padding=padding))
            elif layer == 'FC':
                if first_fc:
                    first_fc=False
                    if first:
                        model.add(Flatten(input_shape=self.input_shape))
                    else:
                        model.add(Flatten())
                hidden_units = var[0]
                if type_reg == "l1":
                    model.add(Dense(hidden_units, activation=tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l1(reg)))
                elif type_reg == "l2":
                    model.add(Dense(hidden_units, activation=tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l2(reg)))
                else:
                    model.add(Dense(hidden_units, activation=tf.nn.relu))
            elif layer == 'MaxPool':
                size = int(var[0])
                model.add(MaxPooling2D(size))
            elif layer == 'Dropout':
                size = float(var[0])
                model.add(Dropout(size))
        if self.regression:
            model.add(Dense(4))
        else:
            if (self.input_shape==(28,28,1) or self.input_shape==(32,32,3)) and not self.dataset == "GTSRB":
                model.add(Dense(10, activation=tf.nn.softmax))
            else:
                model.add(Dense(43, activation=tf.nn.softmax))
        self.model = model
        return 
    
    def train(self, epochs=5, verbose=True, optimizer='adam', delete=True):
        if self.regression:
            data = get_GTSRB_regression_data((self.input_shape[0],self.input_shape[1]))
            x_train = data['X_train']
            y_train = data['y_train']
            self.model.compile(optimizer=optimizer,
                loss='mean_squared_error')
            start = time.time()
            checkpoint = ModelCheckpoint('best.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
            hi = self.model.fit(x_train, y_train, epochs=epochs, verbose=verbose, validation_split=0.1, callbacks=[checkpoint])
            self.model.load_weights("best.h5")
        else:
            if self.input_shape == (28,28,1) and self.dataset == "MNIST":
                mnist = tf.keras.datasets.mnist
                (x_train, y_train),(x_test, y_test) = mnist.load_data()
                x_train, x_test = x_train / 255.0, x_test / 255.0
                x_train = x_train.reshape(60000,28,28,1)
                x_test = x_test.reshape(10000,28,28,1)
            elif self.input_shape == (32,32,3) and self.dataset == "CIFAR":
                data = get_CIFAR10_data()
                x_train = data['X_train'].reshape(40000,32,32,3)
                y_train = data['y_train']
                x_val = data['X_val'].reshape(10000,32,32,3)
                y_val = data['y_val']
            else:
                data = get_GTSRB_data((self.input_shape[0],self.input_shape[1]))
                x_train = data['X_train']
                y_train = data['y_train']
            self.model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
            start = time.time()
            if self.input_shape == (28,28,1) and self.dataset == "MNIST":
                checkpoint = ModelCheckpoint('best.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
                hi = self.model.fit(x_train, y_train, epochs=epochs, verbose=verbose, validation_split=0.1, callbacks=[checkpoint])
                self.model.load_weights("best.h5")
            elif self.input_shape == (32,32,3) and self.dataset == "CIFAR":
                checkpoint = ModelCheckpoint('best.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
                hi = self.model.fit(x_train, y_train, epochs=epochs, verbose=verbose, validation_data=(x_val,y_val), callbacks=[checkpoint])
                self.model.load_weights("best.h5")
            else:
                checkpoint = ModelCheckpoint('best.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
                hi = self.model.fit(x_train, y_train, epochs=epochs, verbose=verbose, validation_split=0.1, callbacks=[checkpoint])
                self.model.load_weights("best.h5")
        ende = time.time()
        print("Overall Training Time: ", (ende-start), "s")
        if os.path.isfile("best.h5") and delete:
            os.remove("best.h5")
        return hi
        
    def get_from_keras(self, model):
        self.model = model
        self.udpate_new()
        return self
        
    def save(self, name='Net'):
        model_json = self.model.to_json()
        with open(name+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(name+".h5")
        return

    def save_for_eran(self, name="MNIST_trained_with_keras.tf"):
        f = open(name, "w+")
        np.set_printoptions(threshold=np.infty)
        num = 0
        for i,layer in enumerate(self.model.layers):
            #print('i',i)
            #print('layer',layer)
            #print('num',num)
            we = layer.get_weights()
            if i == len(self.model.layers)-1:
                #print("Layer is Dense")		
                f.write("Affine\n")
                weights = we[0].transpose()
                bias = we[1].transpose()
                f.write(np.array2string(weights, separator=', ').replace("\n",""))
                f.write("\n")
                f.write(np.array2string(bias, separator=', ').replace("\n",""))
                f.write("\n")
            elif layer.__class__.__name__ == "Conv2D":
                #print("Layer is Conv")
                f.write("Conv2D\n")
                num_filter = layer.output_shape[3]
                filtsize = self.structure[num][1][1]
                in_shape = (layer.input_shape[1], layer.input_shape[1], layer.input_shape[3])
                stride = self.structure[num][1][2]
                if self.structure[num][1][3] >= 1:
                    padding = 1
                else:
                    padding = 0
                line = ("ReLU, filters=%d, kernel_size=[%d, %d], input_shape=[%d, %d, %d], stride=[%d, %d], padding=%d\n" %(num_filter, filtsize, filtsize, in_shape[0], in_shape[1], in_shape[2], stride, stride, padding))
                f.write(line)
                weights = we[0]
                bias = we[1]
                f.write(np.array2string(weights, separator=', ', precision=20).replace("\n",""))
                f.write("\n")
                f.write(np.array2string(bias, separator=', ', precision=20).replace("\n",""))
                f.write("\n")
                num = num + 1
            elif layer.__class__.__name__ == "MaxPooling2D":
                #print("Layer is MaxPooling")
                f.write("MaxPooling2D\n")
                (_,is0,is1,is2)=layer.input_shape
                (i0,i1) = layer.strides
                line = "input_shape=[%d, %d, %d], pool_size=[%d, %d]\n" %(is0,is1,is2,i0,i1)
                #print(line)
                f.write(line)
                num = num + 1
            elif layer.__class__.__name__ == "Dropout":
                #print("Layer is Dropout")
                num = num+1
                continue
            else:
                if len(we)==0:
                    continue
                #print("Layer is Dense")
                f.write("ReLU\n")
                weights = we[0].transpose()
                bias = we[1].transpose()
                f.write(np.array2string(weights, separator=', ', precision=20).replace("\n",""))
                f.write("\n")
                f.write(np.array2string(bias, separator=', ', precision=20).replace("\n",""))
                f.write("\n")
                num = num + 1
        f.close()   
 
    def delete_layer(self,del_layers,init='glorot_uniform'):
        if del_layers[0]<0:
            Ndel_layers = []
            for i in del_layers:
                Ndel_layers.append(len(self.model.layers)+i-1)
            del_layers = Ndel_layers
        numl = 0
        model = Sequential()
        #print(model)
        last_layer = None
        if self.structure is None:
            self.structure = []
            writestructure = True
        else:
            writestructure =False
        #model.add(Flatten(input_shape=self.input_shape))
        firstFC = True
        veryfirst = True
        for i,layer in enumerate(self.model.layers[0:-1]):
            if numl in del_layers:
                print("Ueberspringe layer ", i, layer)
                numl = numl + 1
                continue
            print(i, "th Layer ", layer)
            if layer.__class__.__name__ == "Dense":
                print('Dense')
                #if firstFC:
                #    if veryfirst:
                #        model.add(Flatten(input_shape=self.input_shape))
                #        veryfirst = False
                #        fistFC = False
                #    else:
                #        model.add(Flatten())
                #        firstFC = False
                W = self.params['W'+str(numl + 1)]
                b = self.params['b'+str(numl + 1)].transpose()
                model.add(Dense(np.shape(b)[0], activation="relu"))
                print("added dense ",np.shape(b)[0])
                try:
                    model.layers[-1].set_weights([W,b])
                except Exception as e:
                    print(e)
                    print("Layer ",i)    
                    print(layer)
                if writestructure:
                    self.structure.append(('FC',[np.shape(b)[0]]))
                model.layers[-1].trainable = False
                numl = numl + 1
            elif layer.__class__.__name__ == "Conv2D":
                print("Conv")
                W = self.params['W'+str(numl + 1)]
                print(" Shape of weight ", np.shape(W))
                b = self.params['b'+str(numl + 1)].transpose()
                num_filters = layer.output_shape[3]
                filtersize = (np.shape(W)[0], np.shape(W)[1])
                stride = layer.strides
                padding =layer.padding
                if veryfirst:
                    model.add(Conv2D(num_filters, kernel_size=filtersize, activation="relu", input_shape=self.input_shape, strides=stride, padding=padding))
                    veryfirst = False
                else:
                    model.add(Conv2D(num_filters, kernel_size=filtersize, activation="relu",strides=stride, padding=padding))
                try:
                    model.layers[-1].set_weights([W,b])
                except Exception as e:
                    print(e)
                    print("Layer ",i)    
                    print(layer)
                    return
                if writestructure:
                    if layer.padding == 'valid':
                        padding = 0
                    else:
                        padding = 1
                    self.structure.append(('Conv2d',[num_filters,filtersize,stride,padding]))
                model.layers[-1].trainable = False
                numl = numl +1
            elif layer.__class__.__name__ == "MaxPooling2D":
                size = layer.pool_size[0]
                model.add(MaxPooling2D(size))
                if writestructure:
                    self.structure.append(('MaxPool',size))
            else:
                print(layer.__class__.__name__)
                model.add(layer)
            #print(model.layers[-1].output_shape)
            #we = layer.get_weights()
            #W =  we[0].transpose().transpose()
            #b = we[1].transpose()
           
  
            
        layer = self.model.layers[-1]  
        W = self.params['W'+str(numl + 1)]
        b = self.params['b'+str(numl + 1)].transpose()
        model.add(Dense(np.shape(b)[0], kernel_initializer=init))
        try:
            model.layers[-1].set_weights([W,b])
        except Exception as e:
            print(e)
            print("Layer ",i)    
            print(layer)
        if writestructure:
            self.structure.append(('FC',[np.shape(b)[0]]))
        #print(model)
        #print(model.layers)
        self.model = model
        self.num_layers = numl
        return model
    
    def get_structure(self):
        """returns list of tuples (layer_type, num_neurons) for each layer"""
        if len(self.structure) == len(self.model.layers)-1:
            return self.structure

        if self.structure is None:
            self.structure = []
        for i,layer in enumerate(self.model.layers):
            if layer.__class__.__name__ == "Dense":
                we = layer.get_weights()
                self.structure.append(('FC',[np.shape(we[1])[0]]))
            elif layer.__class__.__name__ == "Conv2D":
                W = layer.get_weights()[0]
                b = layer.get_weights()[1]
                num_filters = layer.output_shape[3]
                filtersize = (np.shape(W)[0], np.shape(W)[1])
                stride = layer.strides
                padding =layer.padding
                if layer.padding == 'valid':
                    padding = 0
                else:
                    padding = 1
                self.structure.append(('Conv2d',[num_filters,filtersize,stride,padding]))
            elif layer.__class__.__name__ == "MaxPooling2D":
                size = layer.pool_size[0]
                self.structure.append(('MaxPool',layer.pool_size[0]))
        return self.structure

    def update_keras(self):
        '''update keras model weights to clustered weights'''
        numl = 0
        model = Sequential()
        last_layer = None
        if self.structure is None:
            self.structure = []
            writestructure = True
        else:
            writestructure = False
        if self.dataset != 'ACASXU':
            model.add(Flatten(input_shape=self.input_shape))
        firstFC = True
        veryfirst = True
        # don't adjust last dense layer, last dense layer index depends on whether softmax layer exists
        for i, layer in enumerate(self.model.layers[0:-1]):
            if layer.__class__.__name__ == "Dense":
                W = self.params['W' + str(numl + 1)]
                b = self.params['b' + str(numl + 1)].transpose()
                if firstFC and self.dataset == 'ACASXU':
                    model.add(Dense(np.shape(b)[0], activation="relu", input_shape=self.input_shape))
                    firstFC = False
                else:
                    model.add(Dense(np.shape(b)[0], activation="relu"))
                try:
                    model.layers[-1].set_weights([W, b])
                except Exception as e:
                    print(e)
                    print("Layer ", i)
                    print(layer)
                if writestructure:
                    self.structure.append(('FC', [np.shape(b)[0]]))
                numl = numl + 1
            elif layer.__class__.__name__ == "Conv2D":
                W = self.params['W' + str(numl + 1)]
                b = self.params['b' + str(numl + 1)].transpose()
                num_filters = layer.output_shape[3]
                num_filters = np.shape(W)[3]
                filtersize = (np.shape(W)[0], np.shape(W)[1])
                stride = layer.strides
                padding = layer.padding
                if veryfirst:
                    model.add(
                        Conv2D(num_filters, kernel_size=filtersize, activation="relu", input_shape=self.input_shape,
                               strides=stride, padding=padding))
                    veryfirst = False
                else:
                    model.add(
                        Conv2D(num_filters, kernel_size=filtersize, activation="relu", strides=stride, padding=padding))
                try:
                    model.layers[-1].set_weights([W, b])
                except Exception as e:
                    print(e)
                    print("Layer ", i)
                    print(layer)
                    return
                if writestructure:
                    if layer.padding == 'valid':
                        padding = 0
                    else:
                        padding = 1
                    self.structure.append(('Conv2d', [num_filters, filtersize, stride, padding]))
                numl = numl + 1
            elif layer.__class__.__name__ == "MaxPooling2D":
                size = layer.pool_size[0]
                model.add(MaxPooling2D(size))
                if writestructure:
                    self.structure.append(('MaxPool', layer.pool_size[0]))
            elif layer.__class__.__name__ == "Flatten":
                if i > 0 and self.model.layers[i - 1].__class__.__name__ in ["Conv2D", "MaxPooling2D"]:
                    model.add(layer)
                else:
                    continue
            else:
                model.add(layer)

        layer = self.model.layers[-1]
        W = self.params['W' + str(numl + 1)]
        b = self.params['b' + str(numl + 1)].transpose()
        model.add(Dense(np.shape(b)[0]))
        model.layers[-1].set_weights([W, b])
        if writestructure:
            self.structure.append(('FC', [np.shape(b)[0]]))
        self.model = model
        self.num_layers = numl
        return model
