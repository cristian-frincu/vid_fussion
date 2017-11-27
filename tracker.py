from particle import TrackedParticle

class TrackManager:

    def __init__(self):
	self.particle_list = []
	self.particle_count = 0

    def new_particle(self,x, features):
	particle = TrackedParticle(x,features[0], features[1], trackID = self.particle_count)
	particle.path = []
	particle.reset_frame_status()
	self.particle_count+=1
	self.particle_list.append(particle)
	return self.particle_count -1
    
    def get_particle_list(self):
        return self.particle_list

    def new_frame(self):
        [frame.increase_frames_undetected() for frame in self.particle_list]

#Drop all the particles that have not been detected in n frames
    def drop_lost_particles(self, m=3, n=5):
        temp_particle_list = []

        for particle in self.particle_list:
            #if sum(particle.frame_detection_status) > 0:
	    if particle.frames_undetected < 10:
                temp_particle_list.append(particle)
            else:
                print ("Dropped ID: "+ str(particle.trackID))
                pass

        self.particle_list = temp_particle_list

