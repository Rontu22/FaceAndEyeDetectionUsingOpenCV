import cv2
import matplotlib.pyplot as plt 


#how we can detect face from image


cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("aa.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = cascade.detectMultiScale(gray,1.01,10)

# scale factor is the ratio of their corresponding side
#  scale factor of 1.03 means to decrease the shape of value by 3% untill
# the face is found
# minimum neighbours = how much minimum distance your webcam find your face

for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("face detection",img)


# url = "http://25.141.86.250:8080/video"

# cap = cv2.VideoCapture(url)

# ret,frame = cap.read()

# plt.imshow(frame)

# plt.title("My first image captured")
# plt.show()

#cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()