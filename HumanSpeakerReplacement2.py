# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 23:04:30 2021

@author: P41914
"""

import cv2
import dlib
import numpy as np
from HSR_Processing import readKeypoints, getClosestMatchFilename, getRelativePoints

USE_CAM = False
#.\\man.jpg
REPLACEMENT_IMG = '.\\images\\ck.jpg'
#REPLACEMENT_IMG = ''
#Only used if USE_CAM = False
VIDEO_TO_PROCESS = '.\\videos\\sm.mp4'
FLIP = False

cv2.destroyAllWindows()

saved_keypoints = {}

if REPLACEMENT_IMG == "":
    read_keypoints = readKeypoints()

def get_index(arr):
    index = arr[0][0]
    #if arr[0]:
        #    index = arr[0][0]
    return index
        
def getLandmarks(landmarks):
    keypoints = []
        
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        #cv2.circle(image, (x, y), 6, (255, 0, 0), -1)
        keypoints.append((x, y))
    return keypoints

def get_cropped_mask(pt1, pt2, pt3):
    # Gets the delaunay triangles
    (x, y, widht, height) = cv2.boundingRect(np.array([pt1, pt2, pt3], np.int32))
    cropped_triangle = faceSwap[y: y+height, x: x+widht]
    cropped_mask = np.zeros((height, widht), np.uint8)

    # Fills triangle to generate the mask
    points = np.array([[pt1[0]-x, pt1[1]-y], [pt2[0]-x, pt2[1]-y], [pt3[0]-x, pt3[1]-y]], np.int32)
    cv2.fillConvexPoly(cropped_mask, points, 255)
    return cropped_mask

def draw_mouth(image, keypoints, color):
    for n in range(48, 67):
        print(n)
        if n >= 48 and n < 68:
            x = keypoints[n][0]
            y = keypoints[n][1]
            xt = keypoints[n+1][0]
            yt = keypoints[n+1][1]
            #cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
            cv2.line(image, (x, y), (xt, yt), color, 2)
            #mouth_kp.append((x, y))
    return image

def drawLines(points, image, color):
    for i in range(0, len(points) - 1):
        point = points[i]
        point_n = points[i + 1]
        cv2.line(image, point, point_n, color, 1)
    cv2.line(image, points[len(points) - 1], points[0], color, 1)
    return image

def getKeypointsForRanges(keypoints, list_ranges):
    sel_keypoints = []
    for i in range(0, 69):
        inside = False
        for rng in list_ranges:
            if i in rng:
                inside=True
        if inside:
            sel_keypoints.append(keypoints[i])
    return sel_keypoints
            
#Initialize the dlib components for the landmark detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(".\\shape_predictor_68_face_landmarks.dat")

#Prepare video capture
if USE_CAM:
    cap = cv2.VideoCapture(0)
    _, image = cap.read()
else:
    cap = cv2.VideoCapture(VIDEO_TO_PROCESS)
    _, image = cap.read()
    #image = cv2.imread('.\\man_speaking.png')


video = cv2.VideoWriter('.\\human_speaker_replacement.mp4', -1, 15, (image.shape[1], image.shape[0]))
img_prev = None
full_range = range(0, 68)
mouth_kp_range = range(60, 68)
mouth_kp_range_outer = range(48, 59)
eye_kp_range = range(36, 48)
no_mouth = range(0, 48)
no_inner_mouth = range(0, 59)
image_or = image.copy()
eye_range= range(36, 47)
outline_range = range(0, 26)
basenose_range = range(29, 30)
nose_range = range(27, 36)
top_range = range(20, 23)

run = 0
#try:
while(True):
    # Getting landmarks for the face that will have the first one swapped into
    if USE_CAM:
        ret, image = cap.read()
    else:
        #image = image_or.copy()
        ret, image = cap.read()
    if not ret:
        break
    if FLIP:
        image = cv2.flip(image, 1)
    #cv2.imwrite('.\\cont\\image_base_' + str(run) + '.jpg', image)
    faces = detector(image)
    if len(faces) == 1:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        rect2 = detector(image)[0]
    
        # This creates a with 68 pairs of integer values — these values are the (x, y)-coordinates of the facial structures 
        landmarks_2 = predictor(image, rect2)
        
        landmarks_points2 = []
        # Uses the function declared previously to get a list of the landmark coordinates
        landmarks_points2 = getLandmarks(landmarks_2)
    
        # Generates a convex hull for the second person
        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)
        convexhull_mouth = points2[mouth_kp_range]
        convexhull_mouth_outer = points2[mouth_kp_range_outer]
        
        """To fit the first person's face into the second one's body, we  will distort the triangles generated to so that they have the same dimentions of the ones created with the landmarks of the second person, this will warp the face of the first person to fit the facial features of the second one."""
    
        h = image.shape[0]
        w = image.shape[1]
        channels = image.shape[2]
        #lines_space_new_face = np.zeros((h, w, channels), np.uint8)
        body_new_face = np.zeros((h, w, channels), np.uint8)
        
        #body_new_face = body_cp.copy()
        
        #lines_space_mask = np.zeros((height_swap, width_swap), np.uint8)
        #for n in points2[mouth_kp_range]:
            #x = landmarks_points2[n][0]
            #y = landmarks_points2[n][1]
         #   cv2.circle(image, n, 5, (20, 255, 20), 2)
         
        ##Face swap
        #Load image for face swap
        points_rel = getRelativePoints(points2)
        if REPLACEMENT_IMG == "":
            file = getClosestMatchFilename(read_keypoints, points_rel, no_mouth)
            print(file)
            IMAGE_TO_LOAD = file;
        else:
            file = ""
            IMAGE_TO_LOAD = REPLACEMENT_IMG
        faceSwap = cv2.imread(IMAGE_TO_LOAD)
        #cv2.imshow('FaceSwap', faceSwap)
        faceSwap_gray = cv2.cvtColor(faceSwap, cv2.COLOR_BGR2GRAY)        
        height_swap, width_swap, _ = faceSwap.shape
        
        mask = np.zeros((faceSwap.shape[0], faceSwap.shape[1]), np.uint8)
        
        cv2.imshow('FaceSwap', faceSwap)
        if file not in saved_keypoints:
            faces = detector(faceSwap)
            keypoints = []
            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                
                #Uncomment if face bounding rectangle should be shown
                #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                landmarks = predictor(faceSwap, face)
                
                keypoints = getLandmarks(landmarks)
                saved_keypoints[file] = keypoints
                #keypoints = []
                
                #for n in range(0, 68):
                #    x = landmarks.part(n).x
                #    y = landmarks.part(n).y
                #    cv2.circle(image, (x, y), 6, (255, 0, 0), -1)
                #    keypoints.append((x, y))
        else:
            keypoints = saved_keypoints[file]
            
        if len(keypoints) == 0:
            print("keypoints empty!")
            continue
        
        points2 = np.array(keypoints, np.int32)
        convexHull = cv2.convexHull(points2[no_mouth])
        face_image_1 = cv2.bitwise_and(faceSwap, faceSwap, mask=mask)
        rect = cv2.boundingRect(convexHull)
        
        #Triangulate
        subdiv = cv2.Subdiv2D(rect) # Creates an instance of Subdiv2D
        considered_ranges = [outline_range, nose_range, mouth_kp_range_outer, eye_range]
        #considered_ranges = [outline_range, mouth_kp_range_outer]
        keypoints = getKeypointsForRanges(keypoints, considered_ranges)
        landmarks_points2 = getKeypointsForRanges(landmarks_points2, considered_ranges)
        subdiv.insert(keypoints) # Insert points into subdiv
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)
        indexes_triangles = []
        face_cp = cv2.cvtColor(faceSwap, cv2.COLOR_BGR2GRAY)
        
        points = np.array(keypoints, np.int32)
        
        for triangle in triangles :
            # Gets the vertex of the triangle
            pt1 = (triangle[0], triangle[1])
            pt2 = (triangle[2], triangle[3])
            pt3 = (triangle[4], triangle[5])
            
            # Draws a line for each side of the triangle
            #cv2.line(faceSwap, pt1, pt2, (255, 0, 0), 1,  0)
            #cv2.line(faceSwap, pt2, pt3, (255, 0, 0), 1,  0)
            #cv2.line(faceSwap, pt3, pt1, (255, 0, 0), 1,  0)
        
            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = get_index(index_pt1)
            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = get_index(index_pt2)
            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = get_index(index_pt3)
        
            # Saves coordinates if the triangle exists and has 3 vertices
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                vertices = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(vertices)
        #key = cv2.waitKey()
        if len(indexes_triangles) == 0:
            print("Not able to extract face from comparison image!")
            
        #Go through every triangle of the face to swap in
        for triangle in indexes_triangles:
        
             #if (triangle[0] in mouth_kp_range and 
             #    triangle[1] in mouth_kp_range and
             #    triangle[2] in mouth_kp_range): 
             #    continue
            
            if True or (triangle[0] not in top_range or triangle[1] not in top_range or triangle[2] not in top_range):
        #     # Coordinates of the first person's delaunay triangles
                 pt1 = keypoints[triangle[0]]
                 pt2 = keypoints[triangle[1]]
                 pt3 = keypoints[triangle[2]]
            
            #     # Gets the delaunay triangles
                #Get bounding rect for the current triangle
                 (x, y, widht, height) = cv2.boundingRect(np.array([pt1, pt2, pt3], np.int32))
                 
                 #Get the part of the image where the rectangle is located
                 cropped_triangle = faceSwap[y: y+height, x: x+widht]
            
            #     # Fills triangle to generate the mask
                 #cropped_mask = np.zeros((height, widht), np.uint8)
                 points = np.array([[pt1[0]-x, pt1[1]-y], [pt2[0]-x, pt2[1]-y], [pt3[0]-x, pt3[1]-y]], np.int32) 
                 #cv2.fillConvexPoly(cropped_mask, points, 255)
            
            #     # Draws lines for the triangles
            #     #cv2.line(lines_space_mask, pt1, pt2, 255)
            #     #cv2.line(lines_space_mask, pt2, pt3, 255)
            #     #cv2.line(lines_space_mask, pt1, pt3, 255)
            
            #     #lines_space = cv2.bitwise_and(faceSwap, faceSwap, mask=lines_space_mask)
            
            #     # Calculates the delaunay triangles of the second person's face
                 pt1 = landmarks_points2[triangle[0]]
                 pt2 = landmarks_points2[triangle[1]]
                 pt3 = landmarks_points2[triangle[2]]
                 
                 #bound_min = (0, 0)
                 #With images the y component is first while for the points it is switched
                 #bound_max = (body_new_face.shape[1], body_new_face.shape[0])
                 
                 #if (pt1 < bound_min or pt1 >= bound_max or 
                 #    pt2 < bound_min or pt2 >= bound_max or
                 #    pt3 < bound_min or pt3 >= bound_max):
                 #    continue
            
            #     # Gets the delaunay triangles
                 (x, y, widht, height) = cv2.boundingRect(np.array([pt1, pt2, pt3], np.int32))
                 cropped_mask2 = np.zeros((height,widht), np.uint8)
            
            #     # Fills triangle to generate the mask
                 points2 = np.array([[pt1[0]-x, pt1[1]-y], [pt2[0]-x, pt2[1]-y], [pt3[0]-x, pt3[1]-y]], np.int32)
                 cv2.fillConvexPoly(cropped_mask2, points2, 255)
            
            #     # Deforms the triangles to fit the subject's face : https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
                 points =  np.float32(points)
                 points2 = np.float32(points2)
                 M = cv2.getAffineTransform(points, points2)  # Warps the content of the first triangle to fit in the second one
                 dist_triangle = cv2.warpAffine(cropped_triangle, M, (widht, height), None, flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REFLECT_101)
                 dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=cropped_mask2)
                 
                 #M = cv2.getPerspectiveTransform(points, points2)
                 #dist_triangle = cv2.warpPerspective(cropped_triangle, M, (widht, height),flags=cv2.INTER_LINEAR)
                 #dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=cropped_mask2)
                 
            #     # Joins all the distorted triangles to make the face mask to fit in the second person's features
                 body_new_face_rect_area = body_new_face[y: y+height, x: x+widht]
                 body_new_face_rect_area_gray = cv2.cvtColor(body_new_face_rect_area, cv2.COLOR_BGR2GRAY)
            
            #     # Creates a mask
                 masked_triangle = cv2.threshold(body_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                 #print(len(masked_triangle))
                 if (masked_triangle[1].shape[0] != dist_triangle.shape[0] or
                     masked_triangle[1].shape[1] != dist_triangle.shape[1]):
                     continue
                 dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=masked_triangle[1])
            
                # Adds the piece to the face mask
                 body_new_face_rect_area = cv2.add(body_new_face_rect_area, dist_triangle)
                 body_new_face[y: y+height, x: x+widht] = body_new_face_rect_area
                 #cv2.imshow('FrameCopy', body_new_face)
                 #key = cv2.waitKey()
        
        
        # """Finally, we can swap the face masks:"""
        body_new_face = cv2.fillConvexPoly(body_new_face, convexhull_mouth, 0)
        body_new_face = drawLines(convexhull_mouth, body_new_face, (0, 0, 0))
        body_face_mask = np.zeros_like(image_gray)
        body_head_mask = cv2.fillConvexPoly(body_face_mask, convexhull2, 255)
        body_face_mask = cv2.bitwise_not(body_head_mask)
        body_face_mask = cv2.fillConvexPoly(body_face_mask, convexhull_mouth, 255)
        body_face_mask = drawLines(convexhull_mouth, body_face_mask, (255, 255, 255))
        
        
        #cv2.imshow('ManFace', body_face_mask)
        #cv2.imwrite('.\\cont\\man_mask_' + str(run) + '.jpg', body_face_mask)
        
        #cv2.imshow('FrameCopy', body_new_face)
        #key = cv2.waitKey(0)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        face_hsv = cv2.cvtColor(body_new_face, cv2.COLOR_BGR2HSV)
        face_hsv[0, :, :] = image_hsv[0, :, :]
        body_new_face = cv2.cvtColor(face_hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('Mask', body_new_face)
        #image = np.zeros((h, w, channels), np.uint8)
        body_maskless = cv2.bitwise_and(image, image, mask=body_face_mask)
        result = cv2.add(body_maskless, body_new_face)
        
        # #Make overlay fit
        # # Gets the center of the face for the body
        (x, y, widht, height) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x+x+widht)/2), int((y+y+height)/2))
        #body_face_mask = cv2.fillConvexPoly(body_face_mask, convexhull_mouth_outer, 255)
        #cv2.imshow('body_face_mask', cv2.bitwise_not(body_face_mask))
        seamlessclone = cv2.seamlessClone(result, image, cv2.bitwise_not(body_face_mask), center_face2, cv2.NORMAL_CLONE)
        for i in range(3):
            seamlessclone = cv2.seamlessClone(result, seamlessclone, cv2.bitwise_not(body_face_mask), center_face2, cv2.NORMAL_CLONE)
        
        image = result.copy()
        #cv2.imwrite('.\\cont\\man_image_' + str(run) + '.jpg', image)
        image = seamlessclone
        #cv2.imwrite('.\\cont\\man_seamless_' + str(run) + '.jpg', image)
        #image = draw_mouth(image, landmarks_points2)
        video.write(image)
        img_prev = image.copy()
    
    cv2.imshow('FrameCopy', image)
    #cv2.imwrite('.\\cont\\image_res' + str(run) + '.jpg', image)
    key = cv2.waitKey(1)
    run += 1
    #'Escape' key to end
    if key == 27:
        break
#except Exception as e:
#    print("Error occured")
#    print(e)
 
#if USE_CAM: 
cap.release()
video.release()
cv2.destroyAllWindows()
