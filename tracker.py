from particle import TrackedParticle

class TrackManager:
    particle_list = []
    particle_count = 0

    def new_particle(self,x):
        particle = TrackedParticle(x, self.particle_count)
	particle.path = []
        self.particle_count+=1
        self.particle_list.append(particle)
        return self.particle_count -1
    
    def get_particle_list(self):
        return self.particle_list

    def new_frame(self):
        [frame.increase_frames_undetected() for frame in self.particle_list]

#Drop all the particles that have not been detected in n frames
    def drop_lost_particles(self, n=5):
        temp_particle_list = []

        for particle in self.particle_list:
            if particle.frames_undetected < n:
                temp_particle_list.append(particle)
            else:
                print ("Dropped ID: "+ str(particle.trackID))
                pass

        self.particle_list = temp_particle_list

