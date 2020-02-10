import cv2
import sqlite3

camera = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

roll = input("enter the roll number: ")
name = input("enter the name: ")

#connection = sqlite3.connect('faces.db')
#query = "INSERT INTO students(name,roll) Values("+str(name)+","+str(roll)+")"
#connection.execute(query)
#connection.commit()
#connection.close()
i=0

while True:
     ret, frame = camera.read()
     gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     face = classifier.detectMultiScale(gray_frame,1.3,5)
     for x,y,w,h in face:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),5)
          cv2.imshow("myface", frame)
          cv2.imwrite("facedata/students."+str(roll)+"."+str(i)+".jpg",gray_frame[y:y+h,x:x+w])
          i = i+1
     if cv2.waitKey(1) & 0xFF == ord('q'):
          break
     elif i>100:
          break

camera.release()
cv2.destroyAllWindows()
     
     
















     

