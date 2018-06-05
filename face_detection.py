# Face-detection using haarcascade frontal face detector

face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img= cv2.imread("faces.jpg")

gray_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces= face_cascade.detectMultiScale( gray_img,scaleFactor=1.08,minNeighbors=5)
                                                                          
for x,y,w,h in faces:
    img= cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    
print(type(faces))

print(faces)

#resized= cv2.resize(img.shape[1]//3,img.shape[0]//3)

cv2.imshow("Gray",img)

cv2.waitKey(0)

cv2.destroyAllWindows()