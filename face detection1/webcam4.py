# detect face from live video
import cv2

# install "IP Webcam" android app in your mobile and start server
# and then put here in place of my IP address , the IP address created by the server
# url = "http://25.141.86.250:8080/video"
url = "http://10.47.206.153:8080/video"

# if you are using webcam from laptop , you can use 
# cap = cv2.VideoCapture(0)   ## This will take input from webcam
cap = cv2.VideoCapture(url)

while True:
    ret,frame = cap.read()
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray,1.03,6)

    for x,y,w,h in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow("Face detection from video",frame)

    if cv2.waitKey(1)== ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
