import os
import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
imagepaths = [os.path.join("facedata",filename) for filename in os.listdir("facedata")]
samples = []
Ids = []
for imagepath in imagepaths:
     imgarray = cv2.imread(imagepath,0)
     Id = int(os.path.split(imagepath)[-1].split(".")[1])
     samples.append(imgarray)
     Ids.append(Id)

recognizer.train(samples,np.array(Ids))
recognizer.write("trainer.yml")
     









     
