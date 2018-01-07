from tracker import TrackManager
import numpy as np
import pandas as pd
import itertools

import cv2


cap = cv2.VideoCapture('vtest.avi')
#cap = cv2.VideoCapture('people.mp4')
#cap = cv2.VideoCapture('balls_m.mp4')
#cap = cv2.VideoCapture('cars2.mp4')
#cap = cv2.VideoCapture('cars.mp4')
#cap = cv2.VideoCapture('cars2.mp4')
#cap = cv2.VideoCapture('ant.mp4')
#cap = cv2.VideoCapture(0)


#Settings for the text to display trackerID
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (0,0,255)
lineType               = 2


ASSUME_SAME_PARTICLE_DISTANCE = 50

tracker = TrackManager()

mog = cv2.createBackgroundSubtractorMOG2()

#n=20
avg = []
seq_num = 0
kernel = np.ones((5,5),np.uint8)

prev_frame_raw = np.array([])
while(True):
#while(cap.isOpened()):

    current_frame_raw = np.array([])
    current_frame_feature = np.array([])

    ret, frame = cap.read()
    cv2.imshow('Outlined',  frame)
    foreground = mog.apply(frame)
    ret, threshold = cv2.threshold(foreground,125,255,cv2.THRESH_BINARY)
    erode = cv2.erode(threshold, kernel)
    foreground = cv2.bitwise_and(frame,frame,mask=erode)
    #keypoints = detector.detect(erode)
    #im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,255,0    ), cv2.DRAW_MATCES_FLAGS_DRAW_RICH_KEYPOINTS)
    im2, contours, hierarchy = cv2.findContours(erode,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
	if cv2.contourArea(cnt) > 75:
	    x,y,w,h = cv2.boundingRect(cnt)
	    current_frame_raw = np.append(current_frame_raw, [x+w/2,y+h/2])
	    target_image = frame[y:y+h, x:x+w]
	    
	    histb = cv2.calcHist([target_image],[0],None,[10], [0,10])
	    histg = cv2.calcHist([target_image],[1],None,[10], [0,10])
	    histr = cv2.calcHist([target_image],[2],None,[10], [0,10])
	    hist_combined = np.append(np.append(histb, histg), histr)
	    current_frame_feature = np.append(current_frame_feature, [hist_combined])
	    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    current_frame_raw = current_frame_raw.reshape(current_frame_raw.shape[0]/2,2)
    current_frame_feature = current_frame_feature.reshape(current_frame_feature.shape[0]/30,30)

    tracker.drop_lost_particles()
    tracker.new_frame() # used for track management
    tracker.match_particles(current_frame_raw, current_frame_feature, ASSUME_SAME_PARTICLE_DISTANCE)

    for part in tracker.get_particle_list():
	position = part.x
#	cv2.rectangle(frame,(int(position[0])-10,int(position[2])-10),(position[0]+10,position[2]+10),(255,125,0),2)
	if part.particle_state == "CONFIRMED":
	    cv2.putText(frame,str(part.trackID),(int(position[0]),int(position[2])), font,fontScale,fontColor, lineType)
	    x,x_v,y,y_v = position
	    cv2.rectangle(frame,(x+x_v,y+y_v),(x+x_v+10,y+y_v+10),(255,0,0),1)
	#if part.particle_state == "TENTATIVE":
	    #cv2.putText(frame,str(part.trackID),(int(position[0]),int(position[2])), font,fontScale,(125,125,23), lineType)
	#
	if len(part.path) > 1:
	    last = part.path[0]
	    for past in part.path:
		cv2.line(frame, (last[0], last[2]),(past[0], past[2]),(0,0,255))
		last = past

    cv2.imshow('Outlined',  frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
	    break
    if seq_num >= 794:
	break

    seq_num+=1
    #print seq_num
print tracker.particle_count
