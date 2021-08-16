import cv2

cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
cap = cv2.VideoCapture(r'C:\Users\rajat\OneDrive\Desktop\face detection border bounding\WIN_20210712_16_26_48_Pro.mp4')
#cap = cv2.VideoCapture(0) for webcam capture

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = cascade_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5) #to detect face
    print(detections)
    if(len(detections) > 0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # for (x,y,w,h) in detections:
    # 	frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Display the resulting frame
    cv2.imshow('frame',frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):  #press q to go out of the loop
        break

# When everything done, release the capture
cap.release() # changes the state to unlocked and returns immediately
cv2.destroyAllWindows()