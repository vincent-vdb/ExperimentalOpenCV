import cv2 
import numpy as np

img = cv2.imread('coins.jpg')
gc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gc = cv2.medianBlur(gc, 3)
ret, gc = cv2.threshold(gc, 0, 255, cv2.THRESH_OTSU)

kernal = np.ones((5,5), np.uint8)
kernal1 = np.ones((3,3), np.uint8)
gc1 = cv2.erode(gc,kernal1, iterations = 1)
gc1 = cv2.dilate(gc1,kernal, iterations = 1)
gc1 = cv2.morphologyEx(gc1, cv2.MORPH_GRADIENT, kernal1)

circles =cv2.HoughCircles(gc1,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
#print(circles)
circles = np.uint16(np.around(circles))
#print(circles)

gc2, contours, hierarchy = 
cv2.findContours(gc1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0,255,0), 3)
for c in contours:
    if cv2.contourArea(c) > 2000:
        #print(c)
        cv2.drawContours(img, contours, -1, (0,255,0), 3)

cv2.imshow('grey2',gc1)
cv2.imshow('grey1',gc)
cv2.imshow('grey3',gc2)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
