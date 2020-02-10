import cv2
import numpy as np
import sqlite3

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
while True:
     ret, frame = camera.read()
     gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     face  = face_classifier.detectMultiScale(gray_frame,1.3,5)
     for x,y,w,h in face:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),10)
          faceid, confi = recognizer.predict(gray_frame[y:y+h,x:x+w])
          print(faceid, " ", confi)
          if faceid == int(1):
               cv2.putText(frame,"balaji",(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(255,255,255),10)
          if faceid == int(2):
               cv2.putText(frame,"viky",(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(255,255,255),10)
               
     cv2.imshow('my face',frame)
     k = cv2.waitKey(1)
     if k==ord('q'):
          break
camera.release()
cv2.destroyAllWindows()
                    
