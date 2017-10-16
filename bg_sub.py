import numpy as np
import time
import cv2

#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
def frame_difference(cap):
    ret, previous_frame= cap.read()
    background_frame =  cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    params = cv2.SimpleBlobDetector_Params()
     
    params.filterByArea = True
    params.filterByColor = False
    params.filterByCircularity =False 
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minArea = 250
    detector = cv2.SimpleBlobDetector_create(params)
    while(cap.isOpened()):
        ret, frame = cap.read()
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        forground_frame = new_frame - background_frame

        kernel = np.ones((2,2),np.uint8)
        erosion = cv2.erode(forground_frame,kernel,iterations = 2)
        dialate = cv2.dilate(erosion,kernel,iterations = 1)
        
        background_frame = new_frame

        ret, threshold = cv2.threshold(dialate,20,255,cv2.THRESH_BINARY)

        img8bit = (threshold).astype('uint8')
        keypoints = detector.detect(img8bit)

        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('frame', im_with_keypoints)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#https://www.learnopencv.com/blob-detection-using-opencv-python-c/
def mean_filter(cap,N):
    ret, first_frame= cap.read()
    bg_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY,1)
    frameNum = 0
    prevFrame = [bg_frame]*N

    detector = cv2.SimpleBlobDetector_create(params)
    while(cap.isOpened()):
        ret, frame = cap.read()
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prevFrame[frameNum%N] = new_frame

        bg_frame = np.mean(prevFrame,axis=0)
        fg_frame = new_frame - bg_frame

        kernel = np.ones((1,1),np.uint8)
        erosion = cv2.erode(fg_frame,kernel,iterations = 2)
        dialate = cv2.dilate(erosion,kernel,iterations = 1)
       
        ret, threshold = cv2.threshold(dialate,10,255,cv2.THRESH_BINARY)
        
        img8bit = (threshold).astype('uint8')


        keypoints = detector.detect(img8bit)

        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
         
         
        cv2.imshow('frame', im_with_keypoints)

        frameNum+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



#http://docs.opencv.org/3.1.0/db/d5c/tutorial_py_bg_subtraction.html
#http://docs.opencv.org/3.1.0/d7/d4d/tutorial_py_thresholding.html

def MOG(c):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    detector = cv2.SimpleBlobDetector_create(params)
    while(cap.isOpened()):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        ret, threshold = cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)
 
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(threshold,kernel,iterations = 1)
        dialate = cv2.dilate(erosion,kernel,iterations = 1)
        
        img8bit = (dialate).astype('uint8')


        keypoints = detector.detect(img8bit)

        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
         
        
        cv2.imshow('frame',im_with_keypoints)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap = cv2.VideoCapture('vtest.avi')

#blob params
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
params.filterByArea = True
params.filterByColor = False
params.filterByCircularity =False 
params.filterByConvexity = False
params.filterByInertia = False
params.minArea = 50

frame_difference(cap)
#MOG(cap)
#mean_filter(cap,N=5)

cap.release()
cv2.destroyAllWindows()

