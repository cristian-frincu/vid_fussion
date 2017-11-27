from tracker import TrackManager
import numpy as np
import pandas as pd
import itertools

ASSUME_SAME_PARTICLE_DISTANCE =  100

targets = pd.read_csv("raw_data.csv",names=['Frame','x','y'])
tracker = TrackManager()

for frame_num in range(20):
#The raw locations from the previous and the current frame
    prev_frame_raw = targets[(targets["Frame"] == frame_num - 1)][['x', 'y']].as_matrix()
    current_frame_raw = targets[(targets["Frame"] == frame_num)][['x', 'y']].as_matrix()

    # Until we have some particles to match, get the last frame readings to be the particles be particles by default
   # if len(tracker.get_particle_list()) == 0:
    #[tracker.new_particle(x) for x in prev_frame_raw]

    tracker.drop_lost_particles(5)
    tracker.new_frame() # used for track management

    #Iterate over the previous frame and the current frame, and compare which particles from last frame match the current frame
    for current_frame_index in range(len(current_frame_raw)):
        smalled_distance = np.inf
        smalled_distance_current_frame_index = -1
        for prev_particle in tracker.get_particle_list():
            distance = np.sqrt((prev_particle.get_predicted_x()[0]- 
                        current_frame_raw[current_frame_index][0])**2 + 
                        (prev_particle.get_predicted_x()[1] -
                        current_frame_raw[current_frame_index][1])**2)

		#Find out which particle is the closest to the prediction
            if distance < smalled_distance:
                smalled_distance_current_frame_index = current_frame_index
                smalled_distance = distance

        # Update the prediction for the best particle match or create a new particle
        if smalled_distance < ASSUME_SAME_PARTICLE_DISTANCE: 
            prev_particle.update_location(current_frame_raw[smalled_distance_current_frame_index])
#            print ("Matched id: "+ str(prev_particle.trackID)+" to "+str(smalled_distance_current_frame_index))
            break
        else:
            new_particle_id = tracker.new_particle(current_frame_raw[smalled_distance_current_frame_index])
            print ("Added new particle with ID:"+str(new_particle_id))

    print ("Frame:"+str(frame_num)+" particles:"+str(len(tracker.get_particle_list()))) 

#for p in tracker.get_particle_list():
#    print (p.trackID)
#    print (p.path)

