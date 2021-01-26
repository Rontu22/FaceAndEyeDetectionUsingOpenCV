# how to detect eye from image ?
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


#url = ""
#cap = cv2.VideoCapture(0)

img = cv2.imread("bb.png")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.03,6)

for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
    roi_gray = gray[y:y+h,x:x+w]
    roi_img =  img[y:y+h,x:x+w]
    eye = eye_cascade.detectMultiScale(roi_gray)

    for ex,ey,ew,eh in eye:
        cv2.rectangle(roi_img,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

cv2.imshow("Eye detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
