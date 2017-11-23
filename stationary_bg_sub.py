import numpy as np
import cv2

#https://stackoverflow.com/questions/10469235/opencv-apply-mask-to-a-color-image
def frame_difference(cap):
    ret, previous_frame= cap.read()
    background_frame =  cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2),np.uint8)
    while(cap.isOpened()):
        ret, frame = cap.read()
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_diff = cv2.absdiff(new_frame,background_frame)
	background_frame = new_frame
	
	ret, threshold = cv2.threshold(frame_diff,25,255,cv2.THRESH_BINARY)
        opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        dilate = cv2.dilate(opening,kernel)

	final = cv2.bitwise_and(frame,frame,mask=dilate)
	contours, hierarchy = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#	cv2.drawContours(frame, contours, -1, (0,255,0), 1)

	keypoints = detector.detect(opening)
	im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	for cnt in contours:
		if cv2.contourArea(cnt) > 150:
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


	cv2.imshow('Outlined - Contours',  frame)
	cv2.imshow('Outlines - Blobs',  im_with_keypoints)
	cv2.imshow('BG Extracted',  final)
	

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

#https://docs.opencv.org/2.4/modules/imgproc/doc/motion_analysis_and_object_tracking.html
#https://stackoverflow.com/questions/8855574/convert-ndarray-from-float64-to-integer
def moving_avg(cap, n=20):
    avg = []
    seq_num = 0
    kernel = np.ones((5,5),np.uint8)
    while(cap.isOpened()):
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

	contours, hierarchy = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		if cv2.contourArea(cnt) > 150:
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

	#cv2.drawContours(frame, contours, -1, (0,255,0), 1)

	cv2.imshow('Outlined',  frame)
	cv2.imshow('Foreground',  front)
	cv2.imshow('Background Extracted',  foreground)
	seq_num+=1
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

#http://answers.opencv.org/question/65005/in-python-how-can-i-reduce-a-list-of-contours-to-those-of-a-specified-size/
def MOG(c, MOG2=False):
    mog = cv2.BackgroundSubtractorMOG()
    if MOG2:
	    mog = cv2.BackgroundSubtractorMOG2()

    kernel = np.ones((3,3),np.uint8)
    detector = cv2.SimpleBlobDetector(params)
    while(cap.isOpened()):
        ret, frame = cap.read()
        foreground = mog.apply(frame)
	ret, threshold = cv2.threshold(foreground,125,255,cv2.THRESH_BINARY)
	erode = cv2.erode(threshold, kernel)

	foreground = cv2.bitwise_and(frame,frame,mask=erode)
	keypoints = detector.detect(erode)
	im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
	contours, hierarchy = cv2.findContours(erode,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		if cv2.contourArea(cnt) > 150:
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

	cv2.imshow('Contour Detection',  frame)
        cv2.imshow('Background Extracted',foreground)
        cv2.imshow('Blob Detection',im_with_keypoints)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 150;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 10
 
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87
 
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01
 
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)

cap = cv2.VideoCapture('vtest.avi')
#cap = cv2.VideoCapture('b1.mp4')
#cap = cv2.VideoCapture('video1.mp4')

#frame_difference(cap)
moving_avg(cap)
#MOG(cap, MOG2=False)

cap.release()
cv2.destroyAllWindows()

