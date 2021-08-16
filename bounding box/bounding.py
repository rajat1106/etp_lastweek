import cv2
import numpy as np

# Load image, grayscale, Otsu's threshold 
image = cv2.imread('main.png') #loading the image
original = image.copy() #copying the image in another varibale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converting the image into grayscale
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] #image segmentation to get binary image

# Find contours, obtain bounding box, extract and save ROI
ROI_number = 0
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1] #cnts consist of our height and width if the image
print(cnts)
for c in cnts:
    x,y,w,h = cv2.boundingRect(c) 
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    ROI = original[y:y+h, x:x+w] 
    cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI) #naming the file 
    ROI_number += 1 

cv2.imshow('image', image) #to display 
cv2.waitKey(0) 