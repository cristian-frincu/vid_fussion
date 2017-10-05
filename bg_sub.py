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
        erosion = cv2.erode(forground_frame,kernel,iterations = 1)
        dialate = cv2.dilate(erosion,kernel,iterations = 1)
        
        background_frame = new_frame

        #cv2.imshow('frame',forground_frame)
        cv2.imshow('frame',dialate)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#http://docs.opencv.org/3.1.0/db/d5c/tutorial_py_bg_subtraction.html
def MOG(c):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while(cap.isOpened()):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        cv2.imshow('frame',fgmask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap = cv2.VideoCapture('vtest.avi')

#frame_difference(cap)
MOG(cap)

cap.release()
cv2.destroyAllWindows()
