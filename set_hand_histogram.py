import cv2
import numpy as np
import pickle
import time

## histogram을 계산할 구역을 사각형(5x10 개)격자로 만들고 시각적으로 표시해줌 해당 구역 범위는 배열로 쌓아 crop으로 반환
def build_squares(img):
   x, y, w, h = 420, 140, 10, 10 ## 구역 시작 좌표, 사각형 격자 크기
   d = 10 ## 격자간 간격
   imgCrop = None
   crop = None
   ## 한 행에 5개씩 총 10개의 사각형 구역을 지정 
   ## imgCrop : 한 행의 총 인식 구역
   ## crop : 총 인식 구역
   for i in range(10):
      for j in range(5):
         if np.any(imgCrop == None):
            imgCrop = img[y:y+h, x:x+w]
         else:
            imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
         cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1) ## 초록색 사각형 띄우기
         x+=w+d
      if np.any(crop == None):
         crop = imgCrop
      else:
         crop = np.vstack((crop, imgCrop))
      imgCrop = None
      x = 420
      y+=h+d
   return crop

## video 속 화면 의 hist를 추출하여 저장해둠.
def get_hand_hist():
   cam = cv2.VideoCapture(1)
   if cam.read()[0]==False:
      cam = cv2.VideoCapture(0)
   x, y, w, h = 300, 100, 300, 300
   flagPressedC, flagPressedS = False, False
   imgCrop = None

   ## 실시간 작동 코드
   while True:
      ## 기본 화면 설정
      img = cam.read()[1] ## 화면에 video 띄우기
      img = cv2.flip(img, 1) ## video 인식 좌우 반전
      img = cv2.resize(img, (640, 480)) ## 화면 크기 설정
      hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) ##Color-space BGR->HSV 변환
      
      keypress = cv2.waitKey(1)
      ## c를 누르면 지정한 구역(imgCrop)의 histogram 계산 및 정규화
      if keypress == ord('c'):   
         hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
         flagPressedC = True
         hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256]) ## 지정구역(imgCrop)의 hsv 정보로 histogram 계산
         cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX) ## histogram 정규화(화질 개선)
      ## s를 누르면 반복문 빠져나가고 video 출력 정지
      elif keypress == ord('s'):
         flagPressedS = True   
         break
      ## c를 눌렀을때, thresh(부드러운 흑백 video 화면) 형성
      if flagPressedC:   
         dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1) ## 전체 video화면을 지정구역의 histogram에 대해 Backproject를 시켜 이진 픽셀 정보(흰, 검)를 얻음
         dst1 = dst.copy()
         disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))## 타원형의 객체 만듬
         cv2.filter2D(dst,-1,disc,dst)  ## 타원형 필터를 씌워 경계를 부드럽게 함
         ## 블러딩(잡음 제거, 영상을 부드럽게)
         blur = cv2.GaussianBlur(dst, (11,11), 0)
         blur = cv2.GaussianBlur(blur, (11,11), 0)
         blur = cv2.GaussianBlur(blur, (11,11), 0)
         blur = cv2.GaussianBlur(blur, (11,11), 0)
         blur = cv2.medianBlur(blur, 15)
         ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) ##흑백 경계를 정확히 나눔
         thresh = cv2.merge((thresh,thresh,thresh))
         cv2.imshow("Thresh", thresh) ##최종 형성된 video창 띄어주기
      # hist를 감지 구역을 지정하고 imgCrop에 넣어둠
      if not flagPressedS:
         imgCrop = build_squares(img)
      cv2.imshow("Set hand histogram", img)
   cam.release()
   cv2.destroyAllWindows() 
   ## hist 정보를 "hist"에 저장
   with open("hist", "wb") as f:
      pickle.dump(hist, f)


get_hand_hist()
