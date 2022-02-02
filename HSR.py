# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 21:51:00 2021

@author: P41914
"""

import cv2
import dlib
import numpy as np
import math

def getLandmarks(landmarks):
    keypoints = []
        
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        #cv2.circle(image, (x, y), 6, (255, 0, 0), -1)
        keypoints.append((x, y))
    return keypoints

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def hasEyesOpen(landmarks, eye_range=range(36, 42)):
    if len(eye_range) != 6:
        print("Eye range is incorrect. Must have a range of exactly six!")
        return False
    
    eye_horizon = np.array(landmarks[39]) - np.array(landmarks[36])
    #, (38, 39), (40, 30)
    check_pairs = [(37, 36), (41, 36)]
    sum_angle = 0
    for pair in check_pairs:
        #eye_landmarks = landmarks[eye_range]
        end = pair[0]
        start = pair[1]
        eye_upper = np.array(landmarks[end]) - np.array(landmarks[start])
        angle = angle_between(eye_horizon, eye_upper)
        sum_angle += angle
        
    print(sum_angle * (180.0 / math.pi))
    return sum_angle
    
def hasEyesOpenAlt(landmarks, eye_range=range(36, 42)):
    if len(eye_range) != 6:
        print("Eye range is incorrect. Must have a range of exactly six!")
        return False
    
    eye_horizon = np.array(landmarks[39]) - np.array(landmarks[36])
    eye_vert = np.array(landmarks[41]) - np.array(landmarks[37])
    rat = np.linalg.norm(eye_vert) / np.linalg.norm(eye_horizon)
    print(rat)
    return rat

eye_right_range = range(36, 42)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(".\\shape_predictor_68_face_landmarks.dat")

image = cv2.imread('.\\10.jpg')

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
    hasEyesOpenAlt(keypoints)
    #keypoints = []
    
    mouth_kp = []
    start = 0
    end = 48
    eye_range = range(start, end)
    #eye_range = eye_right_range
    for n in eye_range:
        print(n)
        #if n >= start and n <= end:
        x = keypoints[n][0]
        y = keypoints[n][1]
        xt = keypoints[n+1][0]
        yt = keypoints[n+1][1]
        #cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
        #cv2.line(image, (x, y), (xt, yt), (20, 255, 20), 2)
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        mouth_kp.append((x, y))
    #cv2.line(image, (keypoints[end][0], keypoints[end][1]), (keypoints[start][0], keypoints[start][1]), (20, 255, 20), 2)
        #mouth_kp.append((x, y))
    
cv2.imwrite("man_base.jpg", image)
cv2.imshow("ImageWithLandmarks", image)
cv2.waitKey()
cv2.destroyAllWindows()