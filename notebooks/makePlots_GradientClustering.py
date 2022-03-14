import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from tqdm import tqdm
import pickle
from sklearn.metrics.pairwise import cosine_distances
import csv  

import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #for no tensorflow warnings"
sys.path.append(os.path.abspath('..'))

from src import *
from models import *
from src.main import Cluster_Class, set_file_model, set_keras_model
from src.models import Keras_Model
from src.cluster_methods import SensitivityAnalysis


def load_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 28, 28).astype("float32") / 255
    x_test = x_test.reshape(10000, 28, 28).astype("float32") / 255

    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    #y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10)
    #y_test = tf.keras.utils.to_categorical(y_test, num_classes = 10)

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def make_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28,1), name="in"))
    model.add(tf.keras.layers.Dense(100, activation="relu", name="d1"))
    model.add(tf.keras.layers.Dense(100, activation="relu", name="d2"))
    model.add(tf.keras.layers.Dense(100, activation="relu", name="d3"))
    model.add(tf.keras.layers.Dense(100, activation="relu", name="d4"))
    model.add(tf.keras.layers.Dense(100, activation="relu", name="d5"))
    model.add(tf.keras.layers.Dense(10, name="out"))
    #model.summary()
    return

def softmax_loss():        
    def loss(y_true, y_pred):
        y_soft = tf.keras.activations.softmax(y_pred, axis=1)
        loss_soft=keras.losses.SparseCategoricalCrossentropy()(y_true, y_soft)
        return loss_soft
    return loss

def train_model(model, trainset, valset, testset, batch_size=64, epochs=20, lr = 3e-4):
    (x_train, y_train)= trainset
    (x_val, y_val)=valset
    (x_test, y_test)=testset

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=softmax_loss(), 
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=0,    
    )

    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=1)
    return test_acc

def save_model(model, MODEL_PATH):
    model.save(MODEL_PATH)
    return 

# calculate jacobian -- single input 
def get_jac(inp):
    actis = {}
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inp)
        a = inp
        for layer in model.layers:
            curr_a = layer(a)
            actis[layer.name] = curr_a
            a = curr_a
        y = a

    jacobian = {}
    for key in actis:
        a = actis[key]
        dy_da = tape.jacobian(y, a)
        new_shape = [dy_da.shape[i] for i in range(0, len(dy_da.shape)) if dy_da.shape[i] != 1]
        dy_da = tf.reshape(dy_da, shape=tuple(new_shape))
        jacobian[key] = dy_da.numpy()
    return jacobian

# calculate jacobian -- multiple inputs
def get_jacobian(inputs):
    all_jacs = []
    for i in tqdm(range(inputs.shape[0])):
        inp = inputs[i]
        inp = tf.expand_dims(inp, axis=0)
        one_jac = get_jac(inp)
        all_jacs.append(one_jac)
    
    jacobian = {}
    for k in all_jacs[0].keys():
        jacobian[k] = sum(jac[k] for jac in all_jacs)
    return jacobian

# calculate 10 Jacobians - 1 per label class (for MNIST: labels 0,..,9)
def get_allJacobians(x_test, y_test, num_samples=30):
    all_jacobians = {}
    for i in range(10):
        x_test_labeli = tf.squeeze(x_test[tf.where(y_test == float(i))])
        inpt = x_test_labeli[0:num_samples]
        jacobian = get_jacobian(inpt)
        all_jacobians[i] = jacobian
    return all_jacobians

def save_jacobians(all_jacobians, JAC_PATH):
    with open(JAC_PATH, 'wb') as f:
        pickle.dump(all_jacobians, f)
    return 

def load_jacobians(JAC_PATH):
    with open(JAC_PATH, 'rb') as f:
        all_jacobians = pickle.load(f)
    return all_jacobians

def get_denoisedJacobian(all_jacobians):
    num_classes = 10
    denoised_jacobian = {}

    # shape = (#layers, #neurons_p_layer, #out_neurons)
    sample = list(all_jacobians.values())[0]
    for layer in sample.keys():
        denoised_jacobian[layer] = np.zeros(sample[layer].shape)
        for idx in all_jacobians.keys():
            curr_jac = all_jacobians[idx][layer]
            denoised_jacobian[layer][idx] = curr_jac[idx]
    return denoised_jacobian

def pairwise_cosine_dist(A):
    D = cosine_distances(A)
    return D

def get_distances(jacobian):
    distances = {}
    for k in jacobian.keys():
        if k != 'in' and k != 'out':
            j = jacobian[k].T
            distances[k] = pairwise_cosine_dist(j)
    return distances

def load_model(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'loss': softmax_loss()})
    return model

def get_kerasModel(model):
    m = set_keras_model(model)
    m.update_keras()
    print("model accuracy. ", m.test_accuracy(verbose=False))
    return m

def get_classAccuracies(kerasModel):
    pre_acc = []
    for i in range(0, 10):
        pre_acc.append(kerasModel.test_MNIST_labelacc(i))
    return pre_acc

if __name__ == "__main__":
    MODEL_PATH = 'MNIST_5x100.h5'
    JAC_PATH = 'MNIST_5x100_jacobian.pkl'
    RESULT_PATH = 'MNIST_5x100_results.csv'

    use_new_model = False
    if use_new_model:
        testset, valset, testset = load_dataset()
        model = make_model()
        test_acc = train_model(model, trainset, valset, testset)
        print("test acc. ", test_acc)
        save_model(model, MODEL_PATH)

        tf.executing_eagerly()
        all_jacobians = get_allJacobians(x_test, y_test, num_samples=30)
        save_jacobians(all_jacobians, JAC_PATH)
    
    model = load_model(MODEL_PATH)
    all_jacobians = load_jacobians(JAC_PATH)
    #denoised_jacobian = get_denoisedJacobian(all_jacobians)
    #curr_jac = denoised_jacobian
    
    DATA = []
    for IDX in range(0, 10):
        print("idx={0}".format(IDX))
        curr_jac = all_jacobians[IDX]
        for key in curr_jac.keys():
            curr_jac[key] -= np.amin(curr_jac[key])
            curr_jac[key] /= np.amax(curr_jac[key])
        distances = get_distances(curr_jac)

        kerasModel = get_kerasModel(model)
        pre_acc = get_classAccuracies(kerasModel)[IDX]
        print("-> pre acc. ", pre_acc)

        eps = {1: 1e-7, 2:1e-7, 3:0.01, 4:0.017, 5:0.022}
        SA = SensitivityAnalysis(distances = distances, eps=eps)
        cc = Cluster_Class(kerasModel, [10], cl_method="gradients", test_method=SA)
        acc, dic = cc.perform_clustering(verbose=True)
        print("-> model clustered. ", acc, dic)
        
        post_acc = get_classAccuracies(kerasModel)[IDX]
        print("-> post acc. ", post_acc)

        data_entry = ["IDX"+str(IDX)]
        data_entry.append(dic['rr'])
        data_entry.append(pre_acc)
        data_entry.append(post_acc)
        data_entry.append(acc)
        DATA.append(data_entry)


    header = ['ClassIndex', 'rr', 'PreAcc', 'PostAcc', 'TotalAcc']
    with open(RESULT_PATH, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for data_entry in DATA:
            writer.writerow(data_entry)



