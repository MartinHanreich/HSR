# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:58:55 2021

@author: P41914
"""

import cv2
import dlib
import numpy as np
import os
import math
from os.path import isfile
from deepface import DeepFace

VIDEO_TO_PROCESS = '.\\videos\\jt.mp4'

FROM_VIDEO = True
#FROM_VIDEO = False

cv2.destroyAllWindows()
def getLandmarks(landmarks):
    keypoints = []
        
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        #cv2.circle(image, (x, y), 6, (255, 0, 0), -1)
        keypoints.append((x, y))
    return keypoints

def getRelativePoints(points):
    meanX = np.mean(points[:,0])
    meanY = np.mean(points[:,1])
    mean_point = np.array([meanX, meanY])
    diff_points = points - mean_point
    max_diff_point = np.max(diff_points)
    return diff_points / max_diff_point

def isUnique(keypoint_list, keypoints, threshold):
    keypoints = keypoints.astype('float32')
    for curr_keypoint in keypoint_list:
        curr_keypoint = curr_keypoint.astype('float32')
        diff = np.sum(np.abs(curr_keypoint - keypoints))
        #print(diff)
        if diff < threshold:
            return False
    return True

def getSaveString(relevantImagePaths, keypoints):
    saveString = ""
    for i in range(0, len(keypoints)):
        curr_keypoint = keypoints[i]
        saveString +=  relevantImagePaths[i]  + ";"
        for keypoint in curr_keypoint:
            x = keypoint[0]
            y = keypoint[1]
            saveString += str(x) + ";" + str(y) + ";"
        saveString += "\n"
    return saveString

def getSaveStringEmotion(relevantImagePaths, emotions):
    saveString = ""
    for i in range(0, len(emotions)):
        curr_emotion = emotions[i]
        saveString +=  relevantImagePaths[i]  + ";"
        for emotion in curr_emotion:
            saveString += str(emotion) + ";"
        saveString += "\n"
    return saveString


def readKeypoints():
    file = open("keypoints.txt", "r")
    read_keypoints = {}
    for line in file.readlines():
        text_parts = line.split(';')
        keypoints = []
        filename = text_parts[0]
        #The first text part is the corresponding file name
        for i in range(1, len(text_parts) - 1, 2):
            if text_parts[i] != "" and text_parts[i + 1] != "":
                x = float(text_parts[i])
                y = float(text_parts[i + 1])
                point = (x, y)
                keypoints.append(point)
        read_keypoints[filename] = keypoints
    return read_keypoints

def getClosestMatchFilename(keypoint_dict, keypoints, searchRange = None):
    if searchRange is None:
        searchRange = range(0, len(keypoints))
    keypoints = keypoints.astype('float32')
    curr_smallest = float('inf')
    keys = keypoint_dict.keys()
    filename = ""
    for key in keys:
        curr_keypoint = keypoint_dict[key]
        curr_keypoint = np.array(curr_keypoint, 'float32')
        #diff = np.sum(np.abs(curr_keypoint - keypoints))
        #res = cv2.estimateAffinePartial2D(curr_keypoint[searchRange], keypoints[searchRange])
        #transformedPoints = cv2.transform(curr_keypoint[searchRange], res[0])
        #res = np.matmul(curr_keypoint[searchRange][0], res)
        diff = getDistance(curr_keypoint[searchRange], keypoints[searchRange])
        #diff = np.sum(res[0])
        #print("AffinePartial")
        #print(res)
        #print(diff)
        if diff < curr_smallest:
            curr_smallest = diff
            filename = key
    return filename

def getDistance(first, second):
    return np.sum(np.abs(first - second))

def isEyeOpen(eye):
    A = getDistance(eye[1], eye[5])
    B = getDistance(eye[2], eye[4])
    C = getDistance(eye[0], eye[3])
    ratio = (A + B) / (2 * C)
    return ratio > 0.5

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
    sum_angle_deg = sum_angle * (180.0 / math.pi)
    print(sum_angle_deg)
    return sum_angle_deg > 60

def hasEyesOpenAlt(landmarks, eye_range=range(36, 42)):
    if len(eye_range) != 6:
        print("Eye range is incorrect. Must have a range of exactly six!")
        return False
    
    eye_horizon = np.array(landmarks[39]) - np.array(landmarks[36])
    eye_vert = np.array(landmarks[41]) - np.array(landmarks[37])
    rat = np.linalg.norm(eye_vert) / np.linalg.norm(eye_horizon)
    print(rat)
    return rat > 0.3

def getImagePaths():
    images_path = []
    IMG_PATH = ".\\imgs"
    for file in os.listdir(IMG_PATH):
        full_path = IMG_PATH + "\\" + file
        if isfile(full_path):
            images_path.append(full_path)
    return images_path

def readImages(images_path):
    images = []
    for path in images_path:
        images.append(cv2.imread(path))
    return images

def process():
    if not FROM_VIDEO:
        images_path = getImagePaths()
        images = readImages(images_path)
    #read_keypoints = readKeypoints()
    #read_keypoints = np.array(read_keypoints, 'float32')
    
    #Initialize the dlib components for the landmark detection
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(".\\shape_predictor_68_face_landmarks.dat")
    
    #Prepare video capture
    cap = cv2.VideoCapture(VIDEO_TO_PROCESS)
    prevImage = None
    count = 0
    savedKeypoints = []
    savedEmotions = []
    relevantImagePaths = []
    
    while(True):
        if FROM_VIDEO:
            ret, image = cap.read()
            if not ret:
                break
        else:
            if len(images) == 0:
                break
            image = images.pop(0)
        image_cp = image.copy()
        faces = detector(image_cp)
        if (prevImage is not None and len(faces) == 1) or not FROM_VIDEO:
            #analyze = DeepFace.analyze(image,actions=['emotion'], enforce_detection=False)
            #cv2.imshow('Image', image)
            #cv2.waitKey(0)
            rect = detector(image_cp)[0]
        
            # This creates a with 68 pairs of integer values — these values are the (x, y)-coordinates of the facial structures 
            landmarks = predictor(image_cp, rect)
            
            # Uses the function declared previously to get a list of the landmark coordinates
            landmarks_points = getLandmarks(landmarks)
        
            points = np.float32(landmarks_points)
            points_rel = getRelativePoints(points)
            
            #ind = getClosestMatchIndex(read_keypoints, points_rel)
            #read_img = cv2.imread(".\\imgs\\" + str(ind) + ".jpg")
            # and isEyeOpen(points[36:42])
            #
            if FROM_VIDEO:
                if isUnique(savedKeypoints, points_rel, 4.0) and hasEyesOpenAlt(landmarks_points):    
                    cv2.imshow('FrameCopy', image_cp)
                    #cv2.waitKey(0)
                    print("T")
                    hasEyesOpenAlt(landmarks_points)
                    imgPath = '.\\imgs\\' + str(count) + '.jpg'
                    cv2.imwrite(imgPath, image_cp)
                    count += 1
                    savedKeypoints.append(points_rel)
                    relevantImagePaths.append(imgPath)
                    #savedEmotions.append(list(analyze["emotion"].values()))
                    #print(analyze["dominant_emotion"])
            else:
                 savedKeypoints.append(points_rel)
                 relevantImagePaths.append(images_path[count])
                 #savedEmotions.append(list(analyze["emotion"].values()))
                 count += 1
            
        prevImage = image_cp.copy()
        key = cv2.waitKey(1)
        #'Escape' key to end
        if key == 27:
            break
    
    saveString = getSaveString(relevantImagePaths, savedKeypoints)
    file = open("keypoints.txt", "w")
    file.write(saveString)
    file.close()
    saveString = getSaveStringEmotion(relevantImagePaths, savedEmotions)
    file = open("emotions.txt", "w")
    file.write(saveString)
    file.close()
    cap.release()
    cv2.destroyAllWindows()

#process()
