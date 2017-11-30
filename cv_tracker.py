from tracker import TrackManager
import numpy as np
import pandas as pd
import itertools

import cv2


#cap = cv2.VideoCapture('vtest.avi')
cap = cv2.VideoCapture('people.mp4')
#cap = cv2.VideoCapture('balls_m.mp4')
#cap = cv2.VideoCapture('cars.mp4')
#cap = cv2.VideoCapture('cars2.mp4')


#Settings for the text to display trackerID
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (0,0,255)
lineType               = 2


ASSUME_SAME_PARTICLE_DISTANCE = 75
MOVING_AVG = False
MOG = True

tracker = TrackManager()

mog = cv2.createBackgroundSubtractorMOG2()

n=20
avg = []
seq_num = 0
kernel = np.ones((5,5),np.uint8)

prev_frame_raw = np.array([])
while(cap.isOpened()):

    current_frame_raw = np.array([])
    current_frame_feature = np.array([])

    if MOVING_AVG:

	ret, frame = cap.read()
	new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	if seq_num>=n:
	    avg[seq_num%n] = new_frame
	else:
	    avg.append(new_frame)

	background = np.mean(avg,axis=0)
	erode = cv2.erode(background,kernel)
	front =  erode - new_frame
	erode_f = np.clip(cv2.erode(front,kernel),0,1).astype(np.uint8)
	dilate = cv2.dilate(erode_f,kernel)

	foreground = cv2.bitwise_and(frame,frame,mask=dilate)

	im2, contours, hierarchy = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
	    if cv2.contourArea(cnt) > 150:
		    x,y,w,h = cv2.boundingRect(cnt)
		    current_frame_raw = np.append(current_frame_raw, [x+w/2,y+h/2])

		    current_frame_feature = np.append(current_frame_feature, [w,h])
		    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	
    if MOG:
	ret, frame = cap.read()
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
		current_frame_feature = np.append(current_frame_feature, [w,h])
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    current_frame_raw = current_frame_raw.reshape(current_frame_raw.shape[0]/2,2)
    current_frame_feature = current_frame_feature.reshape(current_frame_feature.shape[0]/2,2)

    #The raw locations from the previous and the current frame
    #prev_frame_raw = targets[(targets["Frame"] == frame_num - 1)][['x', 'y']].as_matrix()
    #current_frame_raw = targets[(targets["Frame"] == frame_num)][['x', 'y']].as_matrix()

    tracker.drop_lost_particles()
    tracker.new_frame() # used for track management

    #Iterate over the previous frame and the current frame, and compare which particles from last frame match the current frame
    for current_frame_index in range(len(current_frame_raw)):
        smalled_distance = np.inf
        smalled_distance_current_frame_index = -1
        for prev_particle in tracker.get_particle_list():

	    # There is a eucledian distance between the center of the objects 
            distance_eucledian = np.sqrt((prev_particle.get_predicted_x()[0]- 
                        current_frame_raw[current_frame_index][0])**2 + 
                        (prev_particle.get_predicted_x()[2] -
                        current_frame_raw[current_frame_index][1])**2)[0]
	    
	    
	    distance_feature = np.sqrt((prev_particle.width - 
	    current_frame_feature[current_frame_index][0])**2 + 
	    (prev_particle.height - current_frame_feature[current_frame_index][0])**2)

	    distance_combined = np.sqrt((distance_eucledian)**2 + distance_feature**2)
		#Find out which particle is the closest to the prediction
            if distance_eucledian < smalled_distance:
                smalled_distance_current_frame_index = current_frame_index
                smalled_distance = distance_combined

        # Update the prediction for the best particle match or create a new particle
	    #print ("Smallest: "+ str(smalled_distance))
	    if smalled_distance < ASSUME_SAME_PARTICLE_DISTANCE: 
		measurmet =current_frame_raw[smalled_distance_current_frame_index]
		feature =current_frame_feature[smalled_distance_current_frame_index]

		prev_particle.update_location([int(measurmet[0]),int(measurmet[1])])
		prev_particle.update_feature(feature)

#		cv2.rectangle(frame,(int(measurmet[0])-5,int(measurmet[1])-5),(int(measurmet[0])+5,int(measurmet[1])+5),(0,0,255),2)
		break

#	    print ("Measurment location raw:" + str(current_frame_raw[smalled_distance_current_frame_index]))
#	    print ("Measurment location Best:" + str(prev_particle.get_predicted_x()))
#           print ("Matched id: "+ str(prev_particle.trackID)+" to "+str(smalled_distance_current_frame_index))
	    #text_position = [int(prev_particle.get_predicted_x()[0]),int(prev_particle.get_predicted_x()[2])]
	    #cv2.rectangle(frame,(int(text_position[0]),int(text_position[1])),(text_position[0]+10,text_position[1]+10),(255,0,0),2)
	    #print ("Kalman Output: "+ str(prev_particle.get_predicted_x()[0])+ ","+str(prev_particle.get_predicted_x()[2]))
	    #cv2.putText(frame,str(prev_particle.trackID),(int(text_position[0]),int(text_position[1])), font,fontScale,fontColor, lineType)
        else:
            new_particle_id = tracker.new_particle(current_frame_raw[smalled_distance_current_frame_index], current_frame_feature[smalled_distance_current_frame_index])
            print ("Added new particle with ID:"+str(new_particle_id))

    #print ("Particles:"+str(len(tracker.get_particle_list()))) 
 #   print ("\n")

    for part in tracker.get_particle_list():
	position = part.x
#	cv2.rectangle(frame,(int(position[0])-10,int(position[2])-10),(position[0]+10,position[2]+10),(255,125,0),2)
	if part.particle_state == "CONFIRMED":
	    cv2.putText(frame,str(part.trackID),(int(position[0]),int(position[2])), font,fontScale,fontColor, lineType)
	#if part.particle_state == "TENTATIVE":
	    #cv2.putText(frame,str(part.trackID),(int(position[0]),int(position[2])), font,fontScale,(125,125,23), lineType)
	

	if len(part.path) > 1:
	    last = part.path[0]
	    #print (part.frame_detection_status)
	    for past in part.path:
		cv2.line(frame, (last[0], last[2]),(past[0], past[2]),(0,0,255))
		last = past
		#cv2.circle(frame,(past[0],past[2]),5,((part.trackID)%255, ((part.trackID*3)%255),(part.trackID*7)%255), -5)



    cv2.imshow('Outlined',  frame)
    #if cv2.waitKey(10) & 0xFF == ord('q'):
    if cv2.waitKey(10) & 0xFF == ord('q'):
	    break

