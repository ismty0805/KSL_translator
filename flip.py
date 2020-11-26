import cv2
import numpy as np
import pickle, os, sqlite3, random


for i in [1, 2, 3, 11]:
    for j in range(401, 801):
        img = cv2.imread("gestures/"+str(i)+"/"+str(j)+".jpg")
        # print(img, "gestures/"+str(i)+"/"+str(j)+".jpg")
        img = cv2.flip(img, 1)
        
        cv2.imwrite("gestures/"+str(i)+"/"+str(j)+".jpg", img)
