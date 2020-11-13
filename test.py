import cv2
import numpy as np
import time

count = 0
avg_time = 0

#카메라 안에 손이 들어가있어야 작동이 되게 했습니다 
#손과 비슷한색깔이 손보다 크면 검출이 잘 안됩니다
cap = cv2.VideoCapture(0)
if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            laptime = time.time()
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
            hue = np.array([0, 48, 80])
            hue2 = np.array([20, 255, 255])#손 색깔
            hand = cv2.inRange(hsv, hue, hue2)
            
            #손의 컨투어를 잘 잡기위한 작업들     
            hand = cv2.GaussianBlur(hand, (3,3), 0)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            hand = cv2.morphologyEx(hand, cv2.MORPH_CLOSE, k)
            hand = cv2.erode(hand, k)
            #손가락을 없애서 손바닥만남긴후 손바닥의 중심좌표를 구하기위함
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (27,27))
            fingerless = cv2.erode(hand, k)
            #cv2.imshow("fingerless", fingerless) #손가락없는 손 출력
            #cv2.imshow("hand", hand) #손 출력
            
            #그레이스케일 모델을 이진화로 바꿔줌
            ret, thresh_cv = cv2.threshold(hand, 254, 255, cv2.THRESH_BINARY)
            ret, thresh_cv2 = cv2.threshold(fingerless, 254, 255, cv2.THRESH_BINARY)
            hand = thresh_cv
            fingerless = thresh_cv2
            
            #손의 컨투어
            handcontours, hierachy = cv2.findContours(hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            handcontrarray = np.array(handcontours) 
            
            #손가락없는 컨투어
            fingerlesscontours, hierachy = cv2.findContours(fingerless, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            fingerlesscontrarray = np.array(fingerlesscontours)            
            
            #손의 컨투어중에 제일 길이간 긴 컨투어를 찾아줌
            handmax = 0
            for i in range(len(handcontrarray)):
                if len(handcontrarray[i]) >= len(handcontrarray[handmax]):
                    handmax = i
                    
            #손가락없는 컨투어중에 제일 길이간 긴 컨투어를 찾아줌
            fingerlessmax = 0
            for i in range(len(fingerlesscontrarray)):
                if len(fingerlesscontrarray[i]) >= len(fingerlesscontrarray[fingerlessmax]):
                    fingerlessmax = i
                        
            #-----------------------------------------
            if fingerlesscontours != []:   
                fx = fingerlesscontours[fingerlessmax][:,:,0] #손가락없는컨투어의 x좌표
                fy = fingerlesscontours[fingerlessmax][:,:,1] #손가락없는컨투어의 y좌표

                avgx = np.mean(fx) #컨투어의 평균 좌표값
                avgy = np.mean(fy)
                
            #---------------------------------------------------------
                #손 컨투어의 x좌표중 최대값, 최소값을 구함
                x_min, x_max = 0,0
                value = list()
                for j in range(len(handcontours[handmax])):
                    value.append(handcontours[handmax][j][0][0])
                    x_min = np.min(value)-5
                    x_max = np.max(value)+5
                    
                #손 컨투어의 y좌표중 최대값, 최소값을 구함
                y_min, y_max = 0,0
                value = list()
                for j in range(len(handcontours[handmax])):
                    value.append(handcontours[handmax][j][0][1])
                    y_min = np.min(value)-5
                    y_max = np.max(value)+5

                x = x_min
                y = y_min
                w = x_max-x_min
                h = y_max-y_min

                #손 컨투어의 크기만큼 손roi를 만듬
                handroi = thresh_cv[y:y+h, x:x+w]
                
                #기존 캠 크기만큼의 검정색픽셀들로 이루어진 배경생성
                background = np.zeros_like(fingerless)
                
                try:
                    #손roi를 검정배경에 삽입, 삽입시 창 크기를 벗어나면 오류남
                    background[int(avgy-h/2):int(avgy+h/2), int(avgx-w/2):int(avgx+w/2)] = handroi
                except ValueError:
                    print("roi범위 벗어남") #삽입이 안될때
                #cv2.imshow("handroi", handroi) #손roi를 검정배경에 삽입했을 때 출력
                
            #-----------------------------------
                
                #손가락없는 컨투어의 x좌표중 최대값, 최소값을 구함
                x_min, x_max = 0,0
                value = list()
                for j in range(len(fingerlesscontours[fingerlessmax])):
                    value.append(fingerlesscontours[fingerlessmax][j][0][0]) #네번째 괄호가 0일 때 x의 값
                    x_min = np.min(value)
                    x_max = np.max(value)

                #손가락없는 컨투어의 y좌표중 최대값, 최소값을 구함
                y_min, y_max = 0,0
                value = list()
                for j in range(len(fingerlesscontours[fingerlessmax])):
                    value.append(fingerlesscontours[fingerlessmax][j][0][1]) #네번째 괄호가 0일 때 x의 값
                    y_min = np.min(value)
                    y_max = np.max(value)

                x2 = x_min
                y2 = y_min
                w2 = x_max-x_min
                h2 = y_max-y_min

                #손가락없는 컨투어의 크기만큼 손roi를 만듬
                #fingerlessroi = thresh_cv2[y2:y2+h2, x2:x2+w2]
                #cv2.imshow("fingerlessroi", fingerlessroi)#손가락없는roi를 검정배경에 삽입했을 때 출력
                
                
            #---------------------------------------------------------------------               
                #손가락셀 때의 원의 반지름을 구할 때, 주먹을쥐었을 때도 원의 반지름이 주먹크기에 맞게 줄어들어서 
                #손가락으로 인식하게되는 문제가 있음
                #그래서 주먹을 쥐었을때 원의 반지름을 크게해 손가락으로 인식하지 않게함
                
                #손roi의 너비보다 높이가 더 클 때 손가락을 세어줄 원의 반지름 계산
                if h > w:
                    #손roi와 손가락없는roi의 높이를 비교해 차이가 많이나지 않을때
                    if abs(h2 / h) <= 0.7:
                        r = h * 0.37
                    #주먹을 쥐었을때 손roi와 손가락없는roi의 높이값이 비슷함
                    else:
                        r = h
                        
                #손roi의 높이보다 너비가 더 클 때 손가락을 세어줄 원의 반지름 계산
                elif h < w:
                    #손roi와 손가락없는roi의 너비를 비교해 차이가 많이나지 않을때
                    if abs(w2 / w) <= 0.7:
                        r = w * 0.37
                    #주먹을 쥐었을때 손roi와 손가락없는roi의 너비값이 비슷함
                    else:
                        r = w
            #-------------------------------------------------------------------------
                
                #cv2.drawContours(img, fingerlesscontours[fingerlessmax], -1, (0,255,0), 3) #손가락없는 컨투어 그림               
                #cv2.circle(img, (int(avgx), int(avgy)), int(r), (255,0,0), 3) #손가락인식하는 원

                circle = np.zeros_like(fingerless) #손가락검출을위해 흰색원 생성
                cv2.circle(circle,(int(avgx), int(avgy)),int(r),(255,255,255),-1) #손가락없는 컨투어의 좌표에 원을 생성
                #cv2.imshow("circle", circle) 만들어진 흰색 원 그림
                
                #원 둘레의 좌표를 알기위한 컨투어
                circlecontour,_ = cv2.findContours(circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                circlecontourarray = np.array(circlecontour)

                if circlecontourarray is not None: #원 컨투어의 배열이 없을 때는 실행안함
                    cx=circlecontourarray[:,:,:,0] #원 둘레의 x좌표
                    cy=circlecontourarray[:,:,:,1] #원 둘레의 y좌표

                    finger = 0 #손가락수
                    for i in range(len(circlecontourarray[0])): #원의 둘레만큼 반복
                        a = background[cy[0][i-1],cx[0][i-1]] #손 마스크의 전 좌표값
                        b = background[cy[0][i],cx[0][i]] #손 마스크의 현재 좌표값을 비교해서 
                        if(a == 0) and (b == 255): # 값이 달라졌을때
                            #cv2.circle(img, (cx[0][i], cy[0][i]), 5, (0,0,255), -1) #겹친 부분에 조그만 원을 그려줌
                            finger = finger + 1

                    finger = finger - 1 #손목이나 손바닥주변이 손가락으로 인식되는것 때문에 1을 빼줌
                    #cv2.imshow("img", img) #위에 있는 주석들을 출력

                    if finger == 1: #손가락 1개일 때
                        edges = cv2.Canny(img, 100, 200)
                        cv2.imshow("result", edges)

                    elif finger == 2: #손가락 2개일 때
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img_gray = cv2.GaussianBlur(img_gray, (9,9), 0)
                        edges = cv2.Canny(img_gray, 100, 200)
                        ret, sketch = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
                        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
                        sketch = cv2.erode(sketch, kernel)
                        sketch = cv2.medianBlur(sketch, 5)
                        cv2.imshow("result", sketch)

                    elif finger == 4: #손가락 4개일 때
                        rows, cols = img.shape[:2]
                        exp = 0.5
                        scale = 1
                        mapy, mapx = np.indices((rows, cols), dtype=np.float32)
                        mapx=2*mapx/(cols-1)-1
                        mapy=2*mapy/(rows-1)-1
                        r, theta = cv2.cartToPolar(mapx, mapy)
                        r[r<scale] = r[r<scale] **exp
                        mapx, mapy = cv2.polarToCart(r, theta)
                        mapx = ((mapx + 1)*(cols - 1))/2
                        mapy = ((mapy + 1)*(rows - 1))/2
                        distorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
                        cv2.imshow("result", distorted)

                    else: #그 외의 경우일 때
                        cv2.imshow("result", img)
                    laptime = time.time() - laptime
                    count += 1
                    avg_time = (avg_time * (count - 1) + laptime) / count
            if cv2.waitKey(1) != -1:    
                break
        else:
            print('no frame')
            break
else:
    print("can't open camera.")
print("count :", count)
print("average time :", avg_time)
cap.release()
cv2.destroyAllWindows()