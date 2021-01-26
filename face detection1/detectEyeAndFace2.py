## Detect eye and face from video from your mobile camera
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")





# install "IP Webcam" android app in your mobile and start server
# and then put here in place of my IP address , the IP address created by the server
url = "http://25.23.223.46:8080/video"
cap = cv2.VideoCapture(url)
# if you are using webcam from laptop , you can use 
# cap = cv2.VideoCapture(0)   ## This will take input from webcam


while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.03,6)

    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_frame =  frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray,1.03,6)
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(roi_frame,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    
    cv2.imshow("Eye detect Video",frame)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

