import numpy as np
import pickle
import cv2, os
from glob import glob
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from numpy.linalg import norm
from cv2 import ml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
	img = cv2.imread('gestures/1/100.jpg', 0)
	return img.shape

def get_num_of_classes():
	return len(glob('gestures/*'))

image_x, image_y = get_image_size()

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    #Initialize SVM model
    def __init__(self, C = 1, gamma = 0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setGamma(gamma)
        self.model.setC(C)

    #Train svm with given training images and labels
    def train(self, images, labels):
        self.model.train(images, cv2.ml.ROW_SAMPLE, labels)

    #Predict the labels of given images
    def predict(self, images):
        return self.model.predict(images)[1].ravel()

#Convert input images into histograms of gradient
def preprocess_hog(digits):
    samples = []
    for img in digits:
        samples += hog_single(img)
    return np.float32(samples)

#Convert input image into the histogram of gradient
def hog_single(img):
    samples=[]
    img = np.squeeze(img)
    
    #Approximate the gradient in x and y direction
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
    mag, ang = cv2.cartToPolar(gx, gy)

    #Create the histogram of gradient
    bin_n = 20
    bin = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = bin[:25,:25], bin[25:,:25], bin[:25,25:], bin[25:,25:]
    mag_cells = mag[:25,:25], mag[25:,:25], mag[:25,25:], mag[25:,25:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    #Transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= norm(hist) + eps

    samples.append(hist)
    return samples


def trainSVM():
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    train_images = np.reshape(train_images, (train_images.shape[0], 50, 50, 1))

    samples=preprocess_hog(train_images)    #Convert training images into histograms of gradient.
    print('SVM is building wait some time ...')
    model = SVM(C=2.67, gamma=5.383)    #Build a SVM model
    model.train(samples, train_labels)  #Train SVM with the HOGs and training labels.
    return model

def predict(model,img):
	samples=np.float32(hog_single(img))     #Convert image into the histogram of gradient.
	pred = model.predict(samples)         #Predict the label with the trained model.
	return pred

