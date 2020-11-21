import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
img = np.zeros((200,400,3),np.uint8)
b,g,r,a = 255,255,255,0
fontpath = "fonts/gulim.ttc"
font = ImageFont.truetype(fontpath, 20)
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
draw.text((60, 70),  "김형준ABC123#GISDeveloper", font=font, fill=(b,g,r,a))
img = np.array(img_pil)
cv2.putText(img,  "by Dip2K", (250,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r), 1, cv2.LINE_AA)
cv2.imshow("res", img)
cv2.waitKey()
cv2.destroyAllWindows()