import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

imName = ("test4.jpg")

# Read the image
img = cv.imread(imName, cv.IMREAD_UNCHANGED)
RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.imread(imName, 0)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# HSV space segmentation
silverMin = np.array([80, 0, 0],np.uint8)
silverMax = np.array([150, 100, 255],np.uint8)

threshImg = cv.inRange(hsv, silverMin, silverMax)

res = cv.bitwise_and(RGB_img, RGB_img, mask=threshImg)

res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

# Blur it
blur = cv.GaussianBlur(res, (11, 11), 0)
#or median blur?

# Make eroding
#kernel = np.ones((4, 4), np.uint8)
#eroded = blur#cv.erode(blur,kernel,iterations = 2)

# Hough circle
circles = cv.HoughCircles(blur,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)

print(len(circles))

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
  # draw the outer circle
  cv.circle(RGB_img,(i[0],i[1]),i[2],(0,255,0),2)
  # draw the center of the circle
  cv.circle(RGB_img,(i[0],i[1]),2,(0,0,255),3)

plt.imshow(RGB_img)
plt.show()


print(i[0],i[1],i[2])
