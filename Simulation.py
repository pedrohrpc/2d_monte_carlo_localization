import numpy as np
import cv2 as cv
from numpy.random import randn, uniform, randint
from Line import Line
from ParticleFilter import ParticleFilter
from FieldGenerator import FieldGenerator
from fov_utils import fov_utils
from Results import Results
import time

class Simulation:

    def __init__(self, robot_speed, scale_factor, fps, 
                 initial_h_tendency = ParticleFilter.UNKNOWN_TENDENCY, initial_v_tendency = ParticleFilter.UNKNOWN_TENDENCY, default_tendency = ParticleFilter.UNKNOWN_TENDENCY):
        self.running = True
        self.fps = fps
        self.dt = 1/self.fps
        self.time = time.time()
        # self.tick()

        self.create_field(scale_factor)
        self.image_width = self.background.shape[1]
        self.image_height = self.background.shape[0]

        self.robot_found = False
        self.horizontal_tendency = initial_h_tendency
        self.vertical_tendency = initial_v_tendency
        self.default_tendency = default_tendency

        self.robot_speed = robot_speed * scale_factor
        self.field_offset = 100 * scale_factor
        self.robot_pos = self.random_pos_robot()
        self.last_robot_pos = np.copy(self.robot_pos)

        self.sensor_reset_possible = False
        self.probability_grid = np.zeros(self.field.shape)
        self.probability_grid_masks = [np.zeros(self.field.shape)]*3
        self.feature_center_mask = np.zeros(self.field.shape)
        self.feature_line_mask = np.zeros(self.field.shape)
        self.feature_direction_mask = np.zeros(self.field.shape)
        self.robot_pov = None

        ## Resultados
        self.results = Results('Test_6',columns=['robot_pos', 'estimated_mean', 'estimated_abs_deviation', 'positional_error',
                                          'robot_found', 'horizontal_tendency', 'vertical_tendency', 'resample_method', 'sensor_resetting', 'new_samples_amount', 'comment'])
        self.comment = 'starting - '

    def tick(self):
        delta_time = time.time()-self.time
        if 1/self.fps > delta_time: time.sleep(1/self.fps-delta_time)
        self.time = time.time()

    def run_particle_filter(self,particle_filter: ParticleFilter):
        while self.running:
            delta_pos = self.robot_feedback()
            particle_filter.predict_partices(delta_pos)
            self.robot_pov, self.probability_grid, self.probability_grid_masks = particle_filter.get_robot_sensor_data(self.background, self.robot_pos, self.feature_center, self.feature_line)
            particle_filter.test_particles()
            self.robot_found, self.horizontal_tendency, self.vertical_tendency = particle_filter.estimate_position()
            resample_method, new_samples_amount = particle_filter.resample_partiles()

            # print(f'Robot pos {self.robot_pos}')
            # print(f'Estimated pos {particle_filter.estimated_mean}')
            # print(f'Pos error  {particle_filter.estimated_abs_deviation}')
            print(f'Tendency: {self.horizontal_tendency} and {self.vertical_tendency}')

            # time.sleep(0.2)

            self.results.add_result(data=[self.robot_pos, particle_filter.estimated_mean, particle_filter.estimated_abs_deviation, self.get_positional_error(particle_filter.estimated_mean),
                                            self.robot_found, self.horizontal_tendency, self.vertical_tendency, resample_method, particle_filter.feature_in_fov, new_samples_amount, self.comment])
            self.comment = ''

    def get_positional_error(self, estimated_pos):
        error = self.robot_pos - estimated_pos
        error_abs = (error[0]**2+error[1]**2)**0.5
        return error_abs

    
    def get_visuals(self, particle_filter: ParticleFilter):
        self.field = np.copy(self.background)
        np.apply_along_axis(self.draw_particle,1,particle_filter.particles)

        for direction in particle_filter.sensors_directions:
            self.draw_line(direction,line_color=[0,150,0])
        for intersection in particle_filter.intersections:
            self.draw_point(intersection)

        self.draw_particle(self.robot_pos,15,[255,255,0],4)
        if self.robot_found: self.draw_particle(particle_filter.estimated_mean,15,[0,0,255],4)
    
    def draw_line(self,line, line_color = [150,150,0], thickness = 2):
        cv.line(self.robot_pov,line.init_point.astype(int),line.end_point.astype(int),line_color,thickness)

    def draw_point(self,particle, radius=5, particle_color=[0,255,0], thickness = 3):
        cv.circle(self.robot_pov,particle.astype(int),radius,particle_color,thickness)

    def draw_particle(self,particle, radius = 8, particle_color = [150,150,0], thickness = 2):
        cv.circle(self.field,particle[:2].astype(int),radius,particle_color,thickness)
        direction = np.asarray([np.cos(np.deg2rad(particle[2])),np.sin(np.deg2rad(particle[2]))])
        cv.line(self.field,particle[:2].astype(int),(particle[:2]+direction*radius).astype(int),particle_color,thickness)

    def robot_feedback(self):
        delta_pos = self.robot_pos-self.last_robot_pos
        self.last_robot_pos = np.copy(self.robot_pos)
        return delta_pos

    def random_pos_robot(self):
        if (self.horizontal_tendency == ParticleFilter.UNKNOWN_TENDENCY) and (self.vertical_tendency == ParticleFilter.UNKNOWN_TENDENCY):
            self.horizontal_tendency = self.default_tendency
            self.vertical_tendency = self.default_tendency
            
        if self.horizontal_tendency == ParticleFilter.LEFT_TENDENCY:
            pos_x = randint(self.field_offset,int(self.image_width/2))
        elif self.horizontal_tendency == ParticleFilter.RIGHT_TENDENCY:
            pos_x = randint(int(self.image_width/2),self.image_width-self.field_offset)
        else:
            pos_x = randint(self.field_offset,self.image_width-self.field_offset)
        
        if self.vertical_tendency == ParticleFilter.UP_TENDENCY:
            pos_y = randint(self.field_offset,int(self.image_height/2))
        elif self.vertical_tendency == ParticleFilter.DOWN_TENDENCY:
            pos_y = randint(int(self.image_height/2),self.image_height-self.field_offset)
        else:
            pos_y = randint(self.field_offset,self.image_height-self.field_offset)

        self.comment = 'robot kidnapping - '
        return np.asarray([pos_x,pos_y,randint(0,360)])

    def move_robot(self, key):
        if key & 0xFF == 27:
            print('exit')
            self.running = False
        if key & 0xFF == ord('r'):
            print('random position')
            self.robot_pos = self.random_pos_robot()
            self.last_robot_pos = np.copy(self.robot_pos)
        
        if key & 0xFF == ord('w'): 
            self.robot_pos[0] += self.robot_speed*np.cos(np.deg2rad(self.robot_pos[2]))
            self.robot_pos[1] += self.robot_speed*np.sin(np.deg2rad(self.robot_pos[2]))
        if key & 0xFF == ord('q'): 
            self.robot_pos[2] -= self.robot_speed/2
        if key & 0xFF == ord('e'): 
            self.robot_pos[2] += self.robot_speed/2
        self.robot_pos[2] = (self.robot_pos[2]+360)%360
    
    def create_field(self, scale_factor):

        self.background, self.background_lines, self.feature_line, self.feature_center= FieldGenerator.generate(scale_factor)
        self.field = np.copy(self.background)





        
