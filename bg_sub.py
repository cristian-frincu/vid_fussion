import numpy as np
import cv2

#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
def frame_difference(cap):
    ret, previous_frame= cap.read()
    background_frame =  cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    while(cap.isOpened()):
        ret, frame = cap.read()
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        forground_frame = new_frame - background_frame

        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(forground_frame,kernel,iterations = 2)
        dialate = cv2.dilate(erosion,kernel,iterations = 1)
        
        background_frame = new_frame

        ret, threshold = cv2.threshold(dialate,20,255,cv2.THRESH_BINARY)
        im2, contours,hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im2,contours,-1,(0,255,0),3)


        cv2.imshow('frame',im2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def mean_filter(cap,N):
    ret, first_frame= cap.read()
    frameNum = 0
    prevFrame = [None]*N
    bg_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    while(cap.isOpened()):
        ret, frame = cap.read()
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prevFrame[frameNum%N] = new_frame

        if frameNum < N:
            for i in range(frameNum):
                bg_frame = cv2.add(bg_frame,prevFrame[i])
                bg_frame = cv2.divide(bg_frame,2)
        else:
            for i in range(N):
                bg_frame = cv2.add(prevFrame[i],bg_frame)
                bg_frame = cv2.divide(bg_frame,2)

        #print frameNum
        fg_frame = new_frame -bg_frame
        cv2.imshow('frame',bg_frame)

        frameNum+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



#http://docs.opencv.org/3.1.0/db/d5c/tutorial_py_bg_subtraction.html
#http://docs.opencv.org/3.1.0/d7/d4d/tutorial_py_thresholding.html

def MOG(c):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while(cap.isOpened()):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        ret, threshold = cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)
 
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(threshold,kernel,iterations = 1)
        dialate = cv2.dilate(erosion,kernel,iterations = 1)
        
        cv2.imshow('frame',dialate)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap = cv2.VideoCapture('vtest.avi')

#frame_difference(cap)
#MOG(cap)
mean_filter(cap,N=3)

cap.release()
cv2.destroyAllWindows()

