import cv2

cap=cv2.VideoCapture(r'C:\Users\rajat\OneDrive\Desktop\face detection border bounding\WIN_20210712_16_26_48_Pro.mp4')

while True:
    ret,frame =cap.read()

    cv2.imshow('frame',frame)
    cv2.waitKey(1)