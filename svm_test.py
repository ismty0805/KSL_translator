import cv2, pickle
import numpy as np
from numpy.linalg import norm
import svm_model_train as st 

#Train SVM model
model=st.trainSVM()

with open("KSL/test_images2", "rb") as f:
    test_images = np.array(pickle.load(f))
with open("KSL/test_labels2", "rb") as f:
    test_labels = np.array(pickle.load(f), dtype=np.int32)

count=0.0
k=0
for i in test_images:
	test_sample=np.float32(st.hog_single(i))	#Convert test image into histogram of gradient
	pred = model.predict(test_sample)	#Predict the class
	if test_labels[k]==int(pred[0]):
		count+=1.0
	k+=1
print(count, k)
print("accuracy=" , (float(count)/float(k))*100.0 ," %" )
