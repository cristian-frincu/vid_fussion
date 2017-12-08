from particle import TrackedParticle
import numpy as np

class TrackManager:

    def __init__(self):
	self.particle_list = []
	self.particle_count = 0

    def new_particle(self,x, features):
	particle = TrackedParticle(x,features)#, trackID = self.particle_count)
	particle.path = []
	particle.reset_frame_status()
	#self.particle_count+=1
	self.particle_list.append(particle)
	return self.particle_count -1
    
    def get_particle_list(self):
        return self.particle_list

    def new_frame(self):
        [frame.increase_frames_undetected() for frame in self.particle_list]

#Drop all the particles that have not been detected in n frames
    def drop_lost_particles(self, m= 5):
        temp_particle_list = []

        for particle in self.particle_list:
	    #Check if the particle has been confirmed and assign a trackID to it

	    if particle.particle_state == "CONFIRMED" and particle.trackID == -1:
		particle.trackID = self.particle_count
		self.particle_count+=1

	    #Implementation of m out of n
            if particle.frame_detection_status.count(1) + particle.frame_detection_status.count(0) >=  m:
	   # if particle.frames_undetected < 10:
                temp_particle_list.append(particle)
            else:
                #print ("Dropped ID: "+ str(particle.trackID))
                pass

        self.particle_list = temp_particle_list
    
    def match_particles(self, current_frame_raw, current_frame_feature, ASSUME_SAME_PARTICLE_DISTANCE = 50):
	for current_frame_index in range(len(current_frame_raw)):
	    smalled_distance = np.inf
	    smalled_distance_current_frame_index = -1

	    smalled_feature_distance = np.inf
	    smalled_feature_current_frame_id = -1
	    for prev_particle in self.get_particle_list():

		# There is a eucledian distance between the center of the objects 
		distance_eucledian = np.sqrt((prev_particle.get_predicted_x()[0]- 
			    current_frame_raw[current_frame_index][0])**2 + 
			    (prev_particle.get_predicted_x()[2] -
			    current_frame_raw[current_frame_index][1])**2)[0]
		
		
		distance_feature = sum(abs(prev_particle.histogram - current_frame_feature[current_frame_index]))

		distance_combined = np.sqrt((distance_eucledian)**2 + distance_feature**2)
		    #Find out which particle is the closest to the prediction
		if distance_eucledian < smalled_distance:
		    smalled_distance_current_frame_index = current_frame_index
		    smalled_distance = distance_eucledian

		if distance_feature < smalled_feature_distance:
		    smalled_feature_current_frame_index = current_frame_index
		    smalled_feature_distance = distance_feature

	    # Update the prediction for the best particle match or create a new particle
		if smalled_distance < ASSUME_SAME_PARTICLE_DISTANCE: #and smalled_feature_distance < 6000: 
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
		new_particle_id = self.new_particle(current_frame_raw[smalled_distance_current_frame_index], current_frame_feature[smalled_distance_current_frame_index])
		#print ("Added new particle with ID:"+str(new_particle_id))



