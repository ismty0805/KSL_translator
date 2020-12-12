# KSL_translator

CS470 Project

Translate Korean Sign language to Text

The main goal of this project is making a KSL translator which translates KSL to Hangul text.  Our outcome is a python program, which detects KSL hand signs from real-time video, translates it into Hangul text, and displays it on the screen.

Classes:
"ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅡ", "ㅣ"

<br>

## Create data(train, validation, test)
>Extract hand's color and make a histogram of it. Applying it to a real-time video, make a thresh play and capture the thresh frames for dataset. A thresh is a grayscale image made from backprojecting the histogram, which makes pixels with the color of hand white, and the background black.
><div><img width="32%" src="https://user-images.githubusercontent.com/62564712/101983460-8c495100-3cbe-11eb-8e72-600fc0f8a309.PNG">
><img width="32%" src=https://user-images.githubusercontent.com/62564712/101983473-910e0500-3cbe-11eb-81bc-80af3068be6c.PNG>
><img width="32%" src="https://user-images.githubusercontent.com/62564712/101983479-93705f00-3cbe-11eb-8939-944c821cfa6e.PNG">
</div>


### 1. Set hand histogram
#### 'c' : exstract histogram
#### 's' : save and exit

<div>
<img width="40%" src="https://user-images.githubusercontent.com/62564712/101983485-9a976d00-3cbe-11eb-992e-fd77fa88317c.PNG">
<img width="45%" align="right" src="https://user-images.githubusercontent.com/62564712/101983488-9f5c2100-3cbe-11eb-92bd-52be29469490.PNG">

</div>
찍기전 사진, 찍고나서 thresh 사진

### 2. Capture


Made 900 pictures for 20 classes.

<img width="100%" align="right" src="https://user-images.githubusercontent.com/62564712/101984259-8013c280-3cc3-11eb-89a6-584e6e26b1ab.PNG">

### 3. Divide data set
[load_images.py](https://github.com/ismty0805/KSL_translator/blob/main/load_images.py)
train data : 600
validate data : 150
test data : 150

<br>

## Build CNN model and train
[cnn_model_train.py](https://github.com/ismty0805/KSL_translator/blob/main/cnn_model_train.py)
Baseline model

    model = Sequential()
    model.add(Conv2D(16, (2,2), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_of_classes, activation='softmax'))
    sgd = optimizers.SGD(lr=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
   

[cnn_normalization.py](https://github.com/ismty0805/KSL_translator/blob/main/cnn_normalization.py)
with batch normalization layer
[cnn_wm_dropout.py](https://github.com/ismty0805/KSL_translator/blob/main/cnn_wm_dropout.py)
with more dropout rate
## Final Program
[![KSL_translator](http://img.youtube.com/vi/K3tHbIjavdM/0.jpg)](https://youtu.be/K3tHbIjavdM?t=0s) 
## Tech Stack
opencv, python, tensorflow
