# For loading MNIST-, GTSRB- and CIFAR-data
import pickle as pickle
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import PIL
from PIL import Image
from random import shuffle
import configparser

try:
    config = configparser.ConfigParser()
    config.read('utils/docs.ini')
    GTSRB_LOCATION = config.get('data','GTSRB')
    if not os.path.exists(GTSRB_LOCATION):
        print("Warning: GTSRB Location is not existant!!!\n Please update directory "+GTSRB_LOCATION)
        GTSRB_LOCATION = None
except:
    print("Warning: No config for GTSRB data provided")

# -------------------------- ACASXU -------------------------

def get_ACASXU_data():
    # get the centroids ...
    # TODO
    centroids = np.array([[325,0.3,-3.143592, 250, 200],[325,0.35,-3.143592,250,200]])
    data = dict()
    data['X_train'] = centroids
    data['X_test'] = centroids
    data['y_train'] = np.zeros_like(centroids)
    data['y_test'] = np.zeros_like(centroids)
    return data

# -------------------------- GTSRB --------------------------
def get_GTSRB_data(sizes=(20,20)):
    """ Loads Data of GTSRB for classification task
     images are automatically rescaled to sizes """
    im,la = load_and_scale(GTSRB_LOCATION + 'Final_Training/Images/', sizes)
    data = {}
    data['X_train'] = im
    data['y_train'] = la
    bot,im,la = get_all_images(GTSRB_LOCATION + 'Final_Test/Images/',cf_file='GT-final_test.csv', sizes=sizes)
    #idx = [True if x in range(43) else False for x in la]
    im = np.stack(im,axis=0)
    #im = im[idx,:,:,:]
    la = np.array(la)
    data['X_test'] = im
    data['y_test'] = la
    return data

def get_GTSRB_regression_data(sizes=(50,50)):
    """ loads GTSRB images and labels for regression task
     scale images AND (!!!!) labels down to some specific size (eg. 50x50)
     s.t. network can really train on this without caring about the data """
    im,la = load_and_scale(GTSRB_LOCATION + 'Final_Training/Images/', sizes, regression=True)
    data = {}
    data['X_train'] = im
    data['y_train'] = la
    bot,im,la = get_all_images(GTSRB_LOCATION + 'Final_Test/Images/',cf_file='GT-final_test.csv', sizes=sizes, regression=True)
    #idx = [True if x in range(43) else False for x in la]
    im = np.stack(im,axis=0)
    #im = im[idx,:,:,:]
    la = np.array(la)
    data['X_test'] = im
    data['y_test'] = la
    return data

def load_and_scale(dire, sizes, regression=False):
    """ runs through all subdirectories of 'dire' and collects all images with their corresponding labels
       this is for the training data, because it is split into a folder for each category"""
    all_dirs = [x[0] for x in os.walk(dire)]
    all_dirs = all_dirs[1:]
    all_ims = []
    all_las = []
    allinall = []
    for diri in all_dirs:
		#print("directory ",diri)
        bot, ims, las = get_all_images(diri+'/',sizes=sizes, regression=regression)
        all_ims.extend(ims)
        all_las.extend(las)
        allinall.extend(bot)
    shuffle(allinall)
    all_ims = [x[0] for x in allinall]
    all_las = [x[1] for x in allinall]
    all_ims = np.stack(all_ims, axis=0)
    all_las = np.array(all_las)
    return all_ims, all_las
	
def get_all_images(diri, cf_file='',sizes=(20,20), regression=False):
    """ runs through 'diri' and collects all images with their corresponding labels
       this is for the test data, because there are no subdirectories"""
    if cf_file == '':
        cf_file = 'GT-' + diri[-6:-1] + '.csv'
    dat = []
    labels = []
    images = []
    bot = []
    with open(diri + cf_file, 'r') as f:
        for line in f:
            dat.append(line[:-1])
    for i in range(1,len(dat)):
        im_name = dat[i].split(';')[0]
        croppi = (int(dat[i].split(';')[3]),int(dat[i].split(';')[4]),int(dat[i].split(';')[5]),int(dat[i].split(';')[6]))
        img = Image.open(diri + im_name)
        if not regression:
            # crop the relevant traffic sign out of the image
            img = img.crop(croppi)
        img = img.resize(sizes)
        nm = np.array(img)
        images.append(nm)
        if regression:
            # add the bounding box values to the label
            x_size = int(dat[i].split(';')[1])
            y_size = int(dat[i].split(';')[2])
            x1 = int(dat[i].split(';')[3])/x_size*sizes[0]
            y1 = int(dat[i].split(';')[4])/y_size*sizes[1]
            x2 = int(dat[i].split(';')[5])/x_size*sizes[0]
            y2 = int(dat[i].split(';')[6])/y_size*sizes[1]
            labels.append((x1,y1,x2,y2))
            bot.append((nm,(x1,y1,x2,y2)))
        else:	
            labels.append(int(dat[i].split(';')[-1]))
            bot.append((nm,int(dat[i].split(';')[-1])))
    return bot, images, labels

# -------------------------- MNIST --------------------------
def get_MNIST_data(val_split=0.1,one_hot=False):
        """ Loads the tensorflow MNIST dataset """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        if val_split>0:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
        # Reshaping the array to 4-dims so that it can work with the Keras API
        if one_hot:
            x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
            x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
            if val_split>0:
                x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
        else:
            x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
            x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
            if val_split>0:
                x_val = x_val.reshape(x_val.shape[0], 1, 28, 28)
        if val_split==0:
            x_val = []
            y_val = []
        # Making sure that the values are float so that we can get decimal points after division
        x_train = x_train.astype('float32')
        if val_split>0:
            x_val = x_val.astype('float32')
        x_test = x_test.astype('float32')
        # Normalizing the RGB codes by dividing it to the max RGB value.
        x_train /= 255
        if val_split>0:
            x_val /= 255
        x_test /= 255
        if one_hot:
            y_train_new = np.zeros((len(y_train),10))
            y_val_new = np.zeros((len(y_val),10))
            y_test_new = np.zeros((len(y_test),10))
            y_train_new[np.arange(len(y_train)),y_train] = 1
            y_val_new[np.arange(len(y_val)),y_val] = 1
            y_test_new[np.arange(len(y_test)),y_test] = 1
            y_train = y_train_new
            y_val = y_val_new
            y_test = y_test_new
        return {
                'X_train': x_train, 'y_train': y_train,
                'X_val': x_val, 'y_val': y_val,
                'X_test': x_test, 'y_test': y_test,
                }


# -------------------------- CIFAR --------------------------
def get_CIFAR10_data(num_training=48000, num_validation=1000, num_test=1000,one_hot=False):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers.
    """
    # Load the raw CIFAR-10 data
    (x_train, y_train), (x_test, y_test) =  tf.keras.datasets.cifar10.load_data()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    # Reshaping the array to 4-dims so that it can work with the Keras API
    if one_hot:
        x_train = x_train.reshape(x_train.shape[0], 32,32,3)
        x_val = x_val.reshape(x_val.shape[0], 32,32,3)
        x_test = x_test.reshape(x_test.shape[0], 32,32,3)
    else:
        x_train = x_train.reshape(x_train.shape[0], 3, 32,32)
        x_val = x_val.reshape(x_val.shape[0], 3, 32,32)
        x_test = x_test.reshape(x_test.shape[0], 3, 32,32)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_val /= 255
    x_test /= 255
    if one_hot:
        y_train_new = np.zeros((len(y_train),10))
        y_val_new = np.zeros((len(y_val),10))
        y_test_new = np.zeros((len(y_test),10))
        y_train_new[np.arange(len(y_train)),y_train] = 1
        y_val_new[np.arange(len(y_val)),y_val] = 1
        y_test_new[np.arange(len(y_test)),y_test] = 1
        y_train = y_train_new
        y_val = y_val_new
        y_test = y_test_new
    return {
                'X_train': x_train, 'y_train': y_train,
                'X_val': x_val, 'y_val': y_val,
                'X_test': x_test, 'y_test': y_test,
                }


class MNIST:
    def __init__(self):
        data = get_MNIST_data(val_split=0,one_hot=True)
        self.test_data = data['X_test']
        self.test_labels = data['y_test']
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = data['X_train'][:VALIDATION_SIZE, :, :, :]
        self.validation_labels = data['y_train'][:VALIDATION_SIZE]
        self.train_data = data['X_train'][VALIDATION_SIZE:, :, :, :]
        self.train_labels = data['y_train'][VALIDATION_SIZE:]
        return

class CIFAR:
    def __init__(self):
        data = get_CIFAR10_data(num_training=44000, num_validation=5000, num_test=1000,one_hot=True)
        self.test_data = data['X_test']
        self.test_labels = data['y_test']       
        self.validation_data = data['X_val']
        self.validation_labels = data['y_val']
        self.train_data = data['X_train']
        self.train_labels = data['y_train']
        return
