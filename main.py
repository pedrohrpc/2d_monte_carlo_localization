import numpy as np
import cv2 as cv
from ParticleFilter import ParticleFilter
from Simulation import Simulation
import threading

speed = 5
fps = 30
scale_factor = 1

default_tendency = ParticleFilter.RIGHT_TENDENCY

simulation = Simulation(speed, scale_factor, fps, initial_h_tendency=default_tendency, default_tendency = default_tendency)

N = 50
w_limits = np.asarray([0,simulation.image_width])
h_limits = np.asarray([0,simulation.image_height])
pos_deviation = [10, 10, 10]
n_sensors = 7

### Real robot params
head_angle = 37 #degrees
vertical_fov = 58 #degrees
horizontal_fov = 87 #degrees
robot_height = 90 #centimeters

particle_filter = ParticleFilter(N,w_limits,h_limits, pos_deviation, n_sensors, horizontal_fov, vertical_fov, head_angle, robot_height, simulation.background_lines, 
                                 initial_h_tendency = default_tendency, default_tendency = default_tendency)

particle_filter_thread = threading.Thread(target=simulation.run_particle_filter, args=([particle_filter]), daemon=True)



while simulation.running:
    key = cv.waitKey(1)
    simulation.move_robot(key)
    if key & 0xFF == ord(' '): 
        particle_filter_thread.start() # Runing in a different thread for performance
    #     simulation.particle_filter_on = True
            # simulation.run_particle_filter(particle_filter)
    # if key & 0xFF == ord('p'): 
    #         particle_filter = ParticleFilter(N,w_limits,h_limits, pos_deviation, n_sensors, horizontal_fov, vertical_fov, head_angle, robot_height, simulation.background_lines, 
    #                              initial_h_tendency = default_tendency, default_tendency = default_tendency)
    #         simulation.run_particle_filter(particle_filter)
    simulation.get_visuals(particle_filter)
    # cv.namedWindow("Soccer Field", cv.WINDOW_NORMAL)
    # cv.resizeWindow("Soccer Field", int(simulation.image_width/scale_factor), int(simulation.image_height/scale_factor))
    cv.imshow("Soccer Field", simulation.field)

    # cv.imshow("probability_grid", simulation.probability_grid)
    # cv.imshow("probability_grid_mask 0", simulation.probability_grid_masks[0])
    # cv.imshow("probability_grid_mask 1", simulation.probability_grid_masks[1])
    # cv.imshow("probability_grid_mask 2", simulation.probability_grid_masks[2])
    # if particle_filter.probability_likehood != 1:
    #     print(f'p: {particle_filter.probability_likehood}, pl: {particle_filter.long_estimate}, ps: {particle_filter.short_estimate}')

    # robot_fov = fov_utils.get_particle_pov(simulation.background, simulation.robot_pos[0], simulation.robot_pos[1], -simulation.robot_pos[2], horizontal_fov, particle_filter.pov_min_range, particle_filter.pov_max_range)
    if type(simulation.robot_pov) != type(None):
        cv.imshow("Robot fov", simulation.robot_pov)

    simulation.tick()

simulation.results.save_result()
cv.destroyAllWindows()