import numpy as np
import cv2

card_cascade=cv2.CascadeClassifier("classifier/one_green.xml")
img = cv2.imread("images/one_green.jpg")
resized = cv2.resize(img, (400, 200))
gray=cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
card=card_cascade.detectMultiScale(gray, 6.5, 17)

for(x,y,w,h) in card:
    resized=cv2.rectangle(resized, (x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('img', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()