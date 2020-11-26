import cv2, pickle
import numpy as np
import tensorflow as tf
# from cnn_tf import cnn_model_fn
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

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	pred_texts = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅡ", "ㅣ"]
	return pred_texts[pred_class]
	
def get_pred_from_contour(contour, thresh):
	x1, y1, w1, h1 = cv2.boundingRect(contour)
	save_img = thresh[y1:y1+h1, x1:x1+w1]
	text = ""
	if w1 > h1:
		save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
	elif h1 > w1:
		save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
	pred_probab, pred_class = keras_predict(model, save_img)
	if pred_probab*100 > 70:
		text = get_pred_text_from_db(pred_class)
	return text

hist = get_hand_hist()
x, y, w, h = 300, 100, 300, 300
is_voice_on = True

def get_img_contour_thresh(img):
	img = cv2.flip(img, 1)
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
	cv2.filter2D(dst,-1,disc,dst)
	blur = cv2.GaussianBlur(dst, (11,11), 0)
	blur = cv2.medianBlur(blur, 15)
	thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	thresh = cv2.merge((thresh,thresh,thresh))
	thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
	thresh = thresh[y:y+h, x:x+w]
	contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
	return img, contours, thresh

def say_text(text):
	if not is_voice_on:
		return
	while engine._inLoop:
		pass
	engine.say(text)
	engine.runAndWait()

def text_mode(cam):
	global is_voice_on
	text = ""
	word = ""
	count_same_frame = 0
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (640, 480))
		img, contours, thresh = get_img_contour_thresh(img)
		old_text = text
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				text = get_pred_from_contour(contour, thresh)
				if old_text == text:
					count_same_frame += 1
				else:
					count_same_frame = 0

				# if count_same_frame > 20:
				# 	# if len(text) == 1:
				# 		# Thread(target=say_text, args=(text, )).start()	#c
				# 	word = word + text
				# 	if word.startswith('I/Me '):
				# 		word = word.replace('I/Me ', 'I ')
				# 	elif word.endswith('I/Me '):
				# 		word = word.replace('I/Me ', 'me ')
				# 	count_same_frame = 0

			elif cv2.contourArea(contour) < 1000:
				# if word != '':	#c
					#print('yolo')
					#say_text(text)
					# Thread(target=say_text, args=(word, )).start()	#c
				text = ""
				# word = ""
		# else:	#c
			# if word != '':	#c
				#print('yolo1')
				#say_text(text)
				# Thread(target=say_text, args=(word, )).start()	#c
			# text = ""
			# word = ""
		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		b,g,r,a = 255,255,255,0
		img_pil = Image.fromarray(blackboard)
		draw = ImageDraw.Draw(img_pil)
		draw.text((180,50), " ", font =ImageFont.truetype('malgun.ttf', 36), fill=(255, 0,0, 0))
		draw.text((30, 100), "Predicted text-" + text, font = ImageFont.truetype('malgun.ttf', 36), fill = (255, 255, 0, 0))
		# draw.text((30, 240), word, font=ImageFont.truetype('malgun.ttf', 36), fill=(b,g,r,a))
		# blackboard = np.array(img_pil)
		# cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
		# cv2.putText(blackboard, "Predicted text- " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
		# cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		line_break_idx = word.rfind("\n")	#a/ Line breaks on blackboard
		if line_break_idx==-1:
			if len(word) > 13:
				word += "\n"
		else:
			if len(word)-line_break_idx > 14:
				word += "\n"
		y0 = 150	
		dy = 50
		for i, line in enumerate(word.split('\n')):
			# print(f'line: {line}')
			y = y0 + i*dy
			line = join_jamos(line)
			draw.text((30, y), line, font=ImageFont.truetype('malgun.ttf', 36), fill=(255, 255, 255, 0))
			# cv2.putText(blackboard, line, (30, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		blackboard = np.array(img_pil)
		if is_voice_on:
			cv2.putText(blackboard, " ", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
		else:
			cv2.putText(blackboard, " ", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		res = np.hstack((img, blackboard))
		cv2.imshow("Recognizing gesture", res)
		cv2.imshow("thresh", thresh)
		keypress = cv2.waitKey(1)
		if keypress == ord('q') or keypress == ord('c'):
			break
		if keypress == ord('v') and is_voice_on:
			is_voice_on = False
		elif keypress == ord('v') and not is_voice_on:
			is_voice_on = True
		elif keypress == 13:	#a/if pressed enter, add current letter to the word
			word = word + text
			count_same_frame = 0
		elif keypress == 8:		#if pressed backspace, delete one character
			word = word[:-1]

	if keypress == ord('c'):
		return 2
	else:
		return 0

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