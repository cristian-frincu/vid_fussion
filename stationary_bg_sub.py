import numpy as np
import cv2

def frame_difference(cap):
    ret, previous_frame= cap.read()
    background_frame =  cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3),np.uint8)

    while(cap.isOpened()):
        ret, frame = cap.read()
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        foreground_frame = new_frame-background_frame
        opening = cv2.morphologyEx(foreground_frame, cv2.MORPH_OPEN, kernel)
        foreground_frame = foreground_frame + opening
        background_frame = new_frame

        cv2.imshow('foreground_frame', foreground_frame)
        cv2.imshow('opening', opening)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap = cv2.VideoCapture('vtest.avi')
frame_difference(cap)

cap.release()
cv2.destroyAllWindows()

