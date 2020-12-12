import cv2, pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread
from PIL import ImageFont, ImageDraw, Image
from hangul_utils import split_syllable_char, split_syllables, join_jamos

engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'	
model = load_model('cnn_model_keras2.h5')

## 저장해둔 histogram 파일 로드하고 반환
def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

## 찍어둔 학습용 이미지 임의로 가져와 크기 반환
def get_image_size():
	img = cv2.imread('gestures/0/700.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

## img를 학습시킨 이미지 크기에 맞게 바꾸고 반환
def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

## model이 예측한 image의 class와 해당 클래스의 정확도를 반환함
def keras_predict(model, image):
	processed = keras_process_image(image) ## 먼저 img를 학습데이터 이미지 크기에 맞춰줌
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

## pred_class에 해당하는 글자를 반환
def get_pred_text(pred_class):
	pred_texts = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅡ", "ㅣ"]
	return pred_texts[pred_class]

## thresh에서 contour부분만 boundingRect한것을 model의 input으로 넣어 class를 예측하고 해당하는 text를 반환한다.
def get_pred_from_contour(contour, thresh):
	x1, y1, w1, h1 = cv2.boundingRect(contour) ## contour을 포함하는 사각형 경계의 크기
	save_img = thresh[y1:y1+h1, x1:x1+w1]
	text = ""
	## boundingRect 으로 자른 이미지를 1/4로 줄임.
	if w1 > h1:
		save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
	elif h1 > w1:
		save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
	pred_probab, pred_class = keras_predict(model, save_img) ## 모델의 img class 예측
	## 유사율이 70%가 넘으면 해당 class에 해당하는 text 가져온다.
	if pred_probab*100 > 70:
		text = get_pred_text(pred_class)
	return text

hist = get_hand_hist()
x, y, w, h = 300, 100, 300, 300 #손 인식 구역 지정
is_voice_on = True

## img 기본설정 및 thresh(img의 histogram 기반 흑백 영상)형성 및 findcountours
## 전체 img, 인식구역 thresh,   
def get_img_contour_thresh(img):
	img = cv2.flip(img, 1) #좌우 반전
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
	cv2.filter2D(dst,-1,disc,dst)
	## 블러링 과 경계구분(잡음 제거, 부드러운 영상)
	blur = cv2.GaussianBlur(dst, (11,11), 0)
	blur = cv2.medianBlur(blur, 15)
	thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	thresh = cv2.merge((thresh,thresh,thresh))
	thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
	thresh = thresh[y:y+h, x:x+w] ##크기 조정(img내 손 인식 구역과 동일)
	contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
	return img, contours, thresh


## 실시간 작동 함수, 
## q나 c가 입력시 1이 아닌 숫자 리턴
def text_mode(cam):
	text = "" ## text : 실시간 model이 예측한 결과
	word = "" ## 
	count_same_frame = 0
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (640, 480))
		img, contours, thresh = get_img_contour_thresh(img)
		old_text = text ## old_text : 총 출력 결과
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				text = get_pred_from_contour(contour, thresh)
				if old_text == text:
					count_same_frame += 1
				else:
					count_same_frame = 0
			elif cv2.contourArea(contour) < 1000:
				text = ""

		blackboard = np.zeros((480, 640, 3), dtype=np.uint8) ## 결과 출력 창
		b,g,r,a = 255,255,255,0
		img_pil = Image.fromarray(blackboard)

		#text(한글) 결과 draw
		draw = ImageDraw.Draw(img_pil)
		draw.text((180,50), " ", font =ImageFont.truetype('malgun.ttf', 36), fill=(255, 0,0, 0))
		draw.text((30, 100), "Predicted text-" + text, font = ImageFont.truetype('malgun.ttf', 36), fill = (255, 255, 0, 0))
		
		line_break_idx = word.rfind("\n")
		## 총 결과 text가 13글자가 넘을때마다 줄 바꿈
		if line_break_idx==-1:
			if len(word) > 13:
				word += "\n"
		else:
			if len(word)-line_break_idx > 14:
				word += "\n"
		y0 = 150	
		dy = 50
		## 총 text 한줄씩 draw
		for i, line in enumerate(word.split('\n')):
			y = y0 + i*dy
			line = join_jamos(line)
			draw.text((30, y), line, font=ImageFont.truetype('malgun.ttf', 36), fill=(255, 255, 255, 0))
		
		## 화면 띄우기
		blackboard = np.array(img_pil)
		cv2.putText(blackboard, " ", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2) ## 손 인식 구역 사각형으로 표시
		res = np.hstack((img, blackboard)) ## video창 과 결과 출력창 이어 붙임
		cv2.imshow("Recognizing gesture", res)
		cv2.imshow("thresh", thresh)
		keypress = cv2.waitKey(1)

		##q 또는 c 입력시 종료
		if keypress == ord('q') or keypress == ord('c'):
			break

		#if pressed enter, add current letter to the word
		if keypress == 13:	
			word = word + text
			count_same_frame = 0
		 #if pressed backspace, delete one character
		if keypress == 8: 
			word = word[:-1]

	if keypress == ord('c'):
		return 2
	else:
		return 0

## 작동 시작 함수, video를 켜고 text_mode가 1을 반환하지 않을 시 종료
def recognize():
	cam = cv2.VideoCapture(1)
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	text = ""
	word = ""
	count_same_frame = 0
	keypress = 1
	while True:
		if keypress == 1:
			keypress = text_mode(cam)
		else:
			break

keras_predict(model, np.zeros((50, 50), dtype = np.uint8))		
recognize()