# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 16:58:30 2021

@author: P41914
"""

import cv2
import dlib
import numpy as np

def getLandmarks(landmarks):
    keypoints = []
        
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        #cv2.circle(image, (x, y), 6, (255, 0, 0), -1)
        keypoints.append((x, y))
    return keypoints

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(".\\shape_predictor_68_face_landmarks.dat")

image = cv2.imread('.\\man_speaking.png')
cap = cv2.VideoCapture(0)
while(True):
    _, image = cap.read()
    faces = detector(image)
    keypoints = []
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        
        #Uncomment if face bounding rectangle should be shown
        #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        landmarks = predictor(image, face)
        
        keypoints = getLandmarks(landmarks)
        #keypoints = []
        
        mouth_kp = []
        
        for n in range(0, 67):
            x = keypoints[n][0]
            y = keypoints[n][1]
            cv2.circle(image, (x, y), 5, (20, 255, 20), 2)
    
    cv2.imshow("ImageWithLandmarks", image)
    key = cv2.waitKey(1)
    #'Escape' key to end
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()