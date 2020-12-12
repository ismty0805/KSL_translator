import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import pickle
import os

## 모든 데이터셋을 가져와서 디렉토리 이름으로 label을 지어주고 모든 img행렬 정보와 label정보를 반환
def pickle_images_labels():
	images_labels = []
	## 모든 데이터셋 가져오기
	images = glob("gestures/*/*.jpg")
	images.sort()
	for image in images:
		print(image)
		## image의 디렉토리 이름을 데이터의 label(class)로 지정(os.sep == 디렉토리 구분자 역할)
		label = image[image.find(os.sep)+1: image.rfind(os.sep)]
		img = cv2.imread(image, 0)
		images_labels.append((np.array(img, dtype=np.uint8), int(label)))
	return images_labels

images_labels = pickle_images_labels()
images_labels = shuffle(shuffle(shuffle(shuffle(images_labels)))) ## Shuffle data sets
images, labels = zip(*images_labels)
print("Length of images_labels", len(images_labels))

## 총 image 900장
## trian : 600장
## validation : 150장
## test : 150장

## 600 train images --> "train_imges"
train_images = images[:int(2/3*len(images))]
print("Length of train_images", len(train_images))
with open("train_images", "wb") as f:
	pickle.dump(train_images, f)
del train_images ## 바로 지워주기

## 600 train labels --> "train_labels"
train_labels = labels[:int(2/3*len(labels))]
print("Length of train_labels", len(train_labels))
with open("train_labels", "wb") as f:
	pickle.dump(train_labels, f)
del train_labels

## 150 validation images --> "val_imges"
val_images = images[int(5/6*len(images)):]
print("Length of test_images", len(val_images))
with open("val_images", "wb") as f:
	pickle.dump(val_images, f)
del val_images

## 150 validation labels --> "val_labels"
val_labels = labels[int(5/6*len(labels)):]
print("Length of val_labels", len(val_labels))
with open("val_labels", "wb") as f:
	pickle.dump(val_labels, f)
del val_labels

## 150 test images --> test_imges"
test_images = images[int(4/6*len(images)):int(5/6*len(images))]
print("Length of test_images", len(test_images))
with open("test_images", "wb") as f:
	pickle.dump(test_images, f)
del test_images

## 150 test labels --> "test_labels"
test_labels = labels[int(4/6*len(labels)):int(5/6*len(images))]
print("Length of test_labels", len(test_labels))
with open("test_labels", "wb") as f:
	pickle.dump(test_labels, f)
del test_labels
