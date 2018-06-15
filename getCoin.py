import cv2 as cv
import numpy as np
import matplotlib as plt

imName = ("test3.jpg")

# Read the image
roi = cv.imread(imName, 1)
gray = cv.imread(imName, 0)
hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

# Blur it
gray_blur = cv.GaussianBlur(gray, (15, 15), 0)

# Threshold it
thresh = cv.adaptiveThreshold(gray_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
cv.THRESH_BINARY_INV, 11, 1)



# Hough circle
circles = cv.HoughCircles(gray_blur,cv.HOUGH_GRADIENT,1,20,
param1=50,param2=30,minRadius=0,maxRadius=0)

print(len(circles))

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
  # draw the outer circle
  cv.circle(roi,(i[0],i[1]),i[2],(0,255,0),2)
  # draw the center of the circle
  cv.circle(roi,(i[0],i[1]),2,(0,0,255),3)


# Make eroding
kernel = np.ones((4, 4), np.uint8)
closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)

# Contours
cont_img = closing.copy()
im2, contours, hierarchy = cv.findContours(cont_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

print(len(contours))

#print(contours)

for cnt in contours:
  area = cv.contourArea(cnt)
  #if area < 2000 or area > 4000:
  #  continue
  if len(cnt) < 5: 
    continue
  ellipse = cv.fitEllipse(cnt) 

  #cv.ellipse(roi, ellipse, (0,255,0), 2) 

cv.imshow("final result", roi)
cv.imshow("eroded", closing)
cv.imshow('image',thresh)
cv.waitKey(0)
cv.destroyAllWindows()

