from scipy.linalg import inv
import numpy as np


def dot3(A, B, C):
	return np.dot(A, np.dot(B, C))

class TrackedParticle:
	trackID = 0
	frames_undetected = 0
	path = []

    #Kalman Filter parameter
	dt = 1.0
	R_var = 4.0  # Measurment variance
	Q_var = 2.0  # Process variance

	# x_pos, y_pos, x_vel,y_vel
	x = np.array([[0., 0., 0., 0.]]).T  # State
	P = np.eye(4)*200  # State covariance

	F = np.array([[1., dt, 0, 0],
		     [0,  1, 0, 0],
		     [0, 0, 1, dt],
		     [0, 0, 0, 1]])  # Process Model

	Q = np.eye(4)*Q_var

	H = np.array([[1., 0., 0., 0.],
		      [0., 0., 1., 0.]])  # Measurmnet conversion

	R = np.eye(2)*R_var
	I = np.eye(4)
	y = np.zeros((2,1))


#If the particle is not identified to belong to a TrackID, -1 represents unidentified
	def __init__(self,  z, trackID=-1):
		self.trackID = trackID
		self.x = np.array([[z[0], 0., z[1], 0.]]).T

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
		
	def increase_frames_undetected(self):
		self.frames_undetected+= 1

	def reset_frames_undetected(self):
		self.frames_undetected = 0 
