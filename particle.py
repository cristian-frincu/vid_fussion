from scipy.linalg import inv
import numpy as np


def dot3(A, B, C):
	return np.dot(A, np.dot(B, C))

class TrackedParticle:
	trackID = 0
	frames_undetected = 0
	path = []

	# Width and height for each particle will also be used to improve the matching estimates
	width = -1
	height = -1
	
	#This is used for the logic to keep track of the m out of n logic for dropout
	# frame_detection_status will be an array holding 1 and 0 to indicate if it is detected or not
	frames_number = 0
	n = 15
	frame_detection_status = [-1]*n
	particle_state = "TENTATIVE" 

    #Kalman Filter parameter
	dt = 1.0
	R_var = 3000  # Measurment variance

	# x_pos, y_pos, x_vel,y_vel
	x = np.array([[0., 0., 0., 0.]]).T  # State
	P = np.eye(4)*2000  # State covariance

	F = np.array([[1., dt, 0, 0],
		     [0,  1, 0, 0],
		     [0, 0, 1, dt],
		     [0, 0, 0, 1]])  # Process Model

	Q = np.array([[3000., 250, 0, 0],
		     [250,  100, 0, 0],
		     [0, 0, 3000, 250],
		     [0, 0, 250, 100]])  # Process Varience



	H = np.array([[1., 0., 0., 0.],
		      [0., 0., 1., 0.]])  # Measurmnet conversion

	R = np.eye(2)*R_var

	I = np.eye(4)
	y = np.zeros((2,1))


#If the particle is not identified to belong to a TrackID, -1 represents unidentified
	def __init__(self,  z, width, height, trackID=-1):
		self.trackID = trackID
		self.x = np.array([[z[0], 0., z[1], 0.]]).T # Initialize the particle at the first measurment location
		self.width = width
		self.height = height

	def update_location(self,z):
		z = np.array([[z[0], z[1]]]).T
		
		#Kalman update
		self.y = z - np.dot(self.H, self.x)
		self.S = dot3(self.H, self.P,self.H.T) + self.R
		K = dot3(self.P, self.H.T, inv(self.S))
		self.x = self.x + np.dot(K, self.y)

		I_KH = self.I -np. dot(K, self.H)
		self.P = dot3(I_KH, self.P, I_KH.T) + dot3(K, self.R, K.T)


		self.path.append(self.x)
		self.reset_frames_undetected()
		# If the location has been updated, the particle must have matched another one , so it was detected

		#Kalman prediction to be implemented here
	def get_predicted_x(self):
		self.x = np.dot(self.F, self.x)
		self.P = dot3(self.F, self.P, self.F.T) + self.Q
		return self.x
		

	def update_feature(self, feature):
		self.width = feature[0]
		self.height = feature[1]


	# Gets called on each frame
	def increase_frames_undetected(self):
		self.frames_number+=1
		self.frame_detection_status[self.frames_number%self.n] = -1
		self.frames_undetected+= 1

	# Gets called when a new mesurment was matched to this particle
	def reset_frames_undetected(self):
		self.frame_detection_status[self.frames_number%self.n] = 1
		self.frames_undetected = 0 
		#if the same particle has been succesfully been associated n times, change from TENTATIVE to CONFIRMED
		if self.frame_detection_status.count(1)>= 8:
		    self.particle_state = "CONFIRMED"

	def reset_frame_status(self):
		self.frame_detection_status = [0]*self.n
