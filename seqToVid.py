import cv2

output = "2b.mp4"

image_path = "2b/img_00000.bmp"
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape
print frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 30.0, (width, height))
i =0
for image in range(955):
    image_path = "2b/img_%05d.bmp"%i
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

   # cv2.imshow('video',frame)
   # if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
   #    break
    i+=1

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))
