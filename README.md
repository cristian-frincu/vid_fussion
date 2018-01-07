# Pedestrian Tracking from stationary camera

### ** Target detection **

Three different methods have been explored to detect targets from a video. They are all based on difference between frames. 
- Simple frame difference, where two different frames are compared, and the difference is considered a target.
- Average frame difference, where a sequence of frames is averaged, and that is considered background, everything else is considered foreground
- Mixture of Gaussians (MOG), where the distribution of color for each pixel is modeled using a mixture of gaussians, and any deviation from the model, is considered foreground

### ** Target tracking **

Tracking deals with the ability to match the targets across multiple video frames. This has been acomplished by using a Kalman filter for each target, and predicting it's position in the next frame. Neighrest neighbour has been used to further match the targets.

The program is modular and split in different classes, making it easy to test different settings and algorithms.

### ** Baysian Optimiation **

An attempt at tuning the Kalman filter automatically was done by manually counting the expected number of targets, and choosing paramers using a Baysian optimiation framework in order for the result to match the ground truth. Results can be improved, if the ground truth for each timestamp was counted, and more featured were taken into account. 


### ** Results on benchmark video

<a href="http://www.youtube.com/watch?feature=player_embedded&v=QEeDb1-t8IE" target="_blank"><img src="http://img.youtube.com/vi/QEeDb1-t8IE/0.jpg" 
alt="Video" width="480" height="360" border="10" /></a>

