import numpy as np
import cv2 as cv
from numpy.random import randn, uniform, randint
from Line import Line
from fov_utils import fov_utils
import ParticleFilter

class ParticleFilter:
    LEFT_TENDENCY = 'left'
    RIGHT_TENDENCY = 'right'
    UP_TENDENCY = 'up'
    DOWN_TENDENCY = 'down'
    UNKNOWN_TENDENCY = 'unknown'

    LEFT_MASK = None
    RIGHT_MASK = None
    UP_MASK = None
    DOWN_MASK = None


    def __init__(self, N, w_limits, h_limits, pos_deviation, n_sensors, horizontal_fov, vertical_fov, head_angle, robot_height, feature_lines, 
                 initial_h_tendency = UNKNOWN_TENDENCY, initial_v_tendency = UNKNOWN_TENDENCY, default_tendency = UNKNOWN_TENDENCY):
        self.N = N
        self.w_limits = w_limits
        self.h_limits = h_limits
        self.out_of_screen_const = cv.norm(self.w_limits)**2+cv.norm(self.h_limits)**2 # Used in comparing distance when no line is found
        self.pos_deviation = pos_deviation

        # EMCL params
        self.prior_entropy = None
        self.current_entropy = None
        self.EMCL = True
        self.emcl_threshold = 0.25
        self.emcl_ess_constant = 2

        # A-MCL params
        self.p_fast = 1
        self.p_slow = 1
        self.n_fast = 0.1
        self.n_slow = 0.001
        self.amcl_ratio = 2
        self.AMCL= False

        # Real robot params
        self.fov = horizontal_fov
        self.pov_max_range = round(robot_height/np.tan((head_angle-vertical_fov/2)*np.pi/180))
        self.pov_min_range = round(robot_height/np.tan((head_angle+vertical_fov/2)*np.pi/180))
        # print(f'max {self.pov_max_range}')
        # print(f'min {self.pov_min_range}')
        # print(f'middle {round(robot_height/np.tan((head_angle)*np.pi/180))}')

        self.horizontal_tendency = initial_h_tendency
        self.vertical_tendency = initial_v_tendency
        self.default_tendency = default_tendency

        ParticleFilter.LEFT_MASK = cv.hconcat([np.ones((int(h_limits[1]),int(w_limits[1]/2),1),np.uint8),np.zeros((int(h_limits[1]),int(w_limits[1]/2),1),np.uint8)])*255
        ParticleFilter.RIGHT_MASK = cv.hconcat([np.zeros((int(h_limits[1]),int(w_limits[1]/2),1),np.uint8),np.ones((int(h_limits[1]),int(w_limits[1]/2),1),np.uint8)])*255
        ParticleFilter.UP_MASK = cv.vconcat([np.ones((int(h_limits[1]/2),int(w_limits[1]),1),np.uint8),np.zeros((int(h_limits[1]/2),int(w_limits[1]),1),np.uint8)])*255
        ParticleFilter.DOWN_MASK = cv.vconcat([np.ones((int(h_limits[1]/2),int(w_limits[1]),1),np.uint8),np.zeros((int(h_limits[1]/2),int(w_limits[1]),1),np.uint8)])*255


        # Testing params
        self.n_sensors = n_sensors
        self.feature_lines = feature_lines
        self.feature_in_fov = False
        self.sensors_result = []
        self.sensors_directions = []
        self.intersections = []

        # Creating particles
        self.particles = self.create_uniform_particles(N)
        self.weights = [1]*len(self.particles)
        self.probability_likehood = sum(self.weights)/self.N
        self.sensor_reset_particles = np.copy(self.particles)
        self.sensor_reset_weights = np.copy(self.weights)
        self.weights = np.divide(self.weights,sum(self.weights))

    def create_uniform_particles(self,N):
        particles = np.empty((N, 3))
        min_w, max_w = self.w_limits
        min_h, max_h = self.h_limits
        if self.horizontal_tendency == ParticleFilter.LEFT_TENDENCY: max_w = max_w/2
        if self.horizontal_tendency == ParticleFilter.RIGHT_TENDENCY: min_w = max_w/2
        if self.vertical_tendency == ParticleFilter.UP_TENDENCY: max_h = max_h/2
        if self.vertical_tendency == ParticleFilter.DOWN_TENDENCY: min_h = max_h/2

        particles[:, 0] = uniform(min_w, max_w, size=N)
        particles[:, 1] = uniform(min_h, max_h, size=N)
        particles[:, 2] = uniform(0, 360, size=N)
        return particles
    
    def predict_partices(self, delta_pos):
        self.particles[:, 2] += (self.pos_deviation[2]/2 * randn(self.N) + delta_pos[2] * (1 + randn(self.N) * self.pos_deviation[2]/100))
        self.particles[:, 2] = (self.particles[:, 2]+360)%360

        dist = (delta_pos[0]**2+delta_pos[1]**2)**0.5
        self.particles[:, 0] += (self.pos_deviation[0]/10 * randn(self.N) + dist*np.cos(np.deg2rad(self.particles[:, 2])) * (1 + randn(self.N) * self.pos_deviation[0]/100))
        self.particles[:, 1] += (self.pos_deviation[1]/10 * randn(self.N) + dist*np.sin(np.deg2rad(self.particles[:, 2])) * (1 + randn(self.N) * self.pos_deviation[1]/100))

    def test_particles(self):
        particle_sensors_result = np.apply_along_axis(self.read_from_sensors,1,self.particles,self.feature_lines)
        particles_score = np.apply_along_axis(self.compare_sensor_data,1,particle_sensors_result)
        self.weights = np.multiply(self.weights,particles_score)
        # self.normalized_weights = np.divide(self.weights,sum(self.weights))
        self.probability_likehood = sum(self.weights)/self.N
        self.weights = np.divide(self.weights,sum(self.weights))

        self.prior_entropy = self.current_entropy
        self.current_entropy = -np.sum(self.weights*np.log(self.weights))
        if type(self.prior_entropy) == type(None):
            self.prior_entropy = self.current_entropy

    def read_from_sensors(self,particle, feature_lines, is_robot = False):
        sensors_result = []
        sensors_directions = []
        intersections = []
        sensor_direction_spacing = self.fov/(self.n_sensors+1)
        angle = particle[2]-self.fov/2+sensor_direction_spacing

        for i in range(self.n_sensors):
            reading_direction = Line.get_line_from_slope(particle[:2],angle)
            reading = self.out_of_screen_const
            closest_intersection = None
            for feature in feature_lines:
                intersection = Line.get_intersection(reading_direction,feature)
                if type(intersection) != type(None): 
                    distance_to_intersection = np.linalg.norm(intersection-particle[:2])
                    if  (distance_to_intersection < reading) and (distance_to_intersection > self.pov_min_range) and (distance_to_intersection < self.pov_max_range):
                        reading = distance_to_intersection
                        reading_direction = Line(particle[:2],intersection)
                        closest_intersection = intersection
            
            if type(closest_intersection) != type(None):
                intersections.append(closest_intersection)
            sensors_directions.append(reading_direction)
            sensors_result.append(reading)
            angle += sensor_direction_spacing

        
        if is_robot:
            self.sensors_result = sensors_result
            self.sensors_directions = sensors_directions
            self.intersections = intersections
            return
        
        else: return sensors_result

    def check_for_features(self, robot_pos, feature_center):
        line_to_feature = Line(robot_pos[:2],feature_center)
        if (abs(robot_pos[2]-line_to_feature.slope) <= self.fov/2) or (abs(robot_pos[2]-line_to_feature.slope) >= (360-self.fov/2)):
            self.feature_in_fov = True
        else: self.feature_in_fov = False
        return self.feature_in_fov

    def get_robot_sensor_data(self, background, robot_pos, feature_center, feature_line):
        robot_pov = fov_utils.get_particle_pov(background, robot_pos, self.fov, self.pov_min_range, self.pov_max_range)
        result_pov, lines_in_pov, pov_feature_center = fov_utils.get_lines_in_pov(robot_pov)
            
        # Getting distance sensor data
        (pov_h, pov_w) = result_pov.shape[:2]
        robot_pov_position = [pov_w/2, self.pov_min_range+pov_h, -90]
        self.read_from_sensors(robot_pov_position,lines_in_pov,is_robot=True)

        # Getting features data
        probability_grid = np.zeros(background.shape, np.uint8)
        probability_grid_masks = [np.zeros(background.shape, np.uint8)]*3
        if pov_feature_center is not None:
            self.feature_in_fov = True
            distance_to_center, distance_to_line, angle_to_feature = self.get_features_data(robot_pov_position,lines_in_pov,pov_feature_center)
            probability_grid, probability_grid_masks = self.update_probability_grid(background.shape, feature_center, feature_line, distance_to_center, distance_to_line, angle_to_feature)
        else:
            # print('cant see feature')
            self.feature_in_fov = False

        return result_pov, probability_grid, probability_grid_masks

    def get_features_data(self,robot_pov_position,lines_in_pov: list[Line],feature_center):
        
        closest_distance_to_feature = self.out_of_screen_const
        for line in lines_in_pov:
            if line.dist_to_point(feature_center) < closest_distance_to_feature:
                pov_feature_line = line
                closest_distance_to_feature = line.dist_to_point(feature_center)


        distance_to_center = np.linalg.norm(robot_pov_position[:2]-feature_center)
        distance_to_line = pov_feature_line.dist_to_point(robot_pov_position[:2])
        angle_to_feature = ((270 + pov_feature_line.slope - Line(robot_pov_position[:2],feature_center).slope)*(-1)+360)%360
        # reset_robot_angle = angle_to_feature - (Line(robot_pov_position[:2],feature_center).slope- 270)
            
        return distance_to_center, distance_to_line, angle_to_feature

    def update_probability_grid(self,background_shape, feature_center, feature_line: Line,distance_to_center, distance_to_line, angle_to_feature):

        h, w, _ = background_shape
        blank_mask = np.zeros((h,w,1), np.uint8)
        percentage_error = 10

        feature_center_mask = np.copy(blank_mask)
        cv.circle(feature_center_mask,feature_center.astype(int),int(distance_to_center),255,max(1,int(distance_to_center*percentage_error/100)))

        feature_line_mask = np.copy(blank_mask)
        delta_x = -distance_to_line*np.sin(np.deg2rad(feature_line.slope))
        delta_y = distance_to_line*np.cos(np.deg2rad(feature_line.slope))
        delta = np.asarray([delta_x,delta_y])
        cv.line(feature_line_mask,(feature_line.init_point+delta).astype(int),(feature_line.end_point+delta).astype(int),255,max(1,int(distance_to_line*percentage_error/100)))
        cv.line(feature_line_mask,(feature_line.init_point-delta).astype(int),(feature_line.end_point-delta).astype(int),255,max(1,int(distance_to_line*percentage_error/100)))

        feature_direction_mask = np.copy(blank_mask)
        feature_direction_line = Line.join_lines(Line.get_line_from_slope(feature_center,angle_to_feature), Line.get_line_from_slope(feature_center,180+angle_to_feature))
        cv.line(feature_direction_mask,(feature_direction_line.init_point).astype(int),(feature_direction_line.end_point).astype(int),255,max(1,int(max(distance_to_line,distance_to_center)*percentage_error/100)))

        probability_grid = np.copy(blank_mask)
        probability_grid = cv.bitwise_and(feature_center_mask,feature_line_mask)
        probability_grid = cv.bitwise_and(probability_grid,feature_direction_mask)
        

        if (self.horizontal_tendency == ParticleFilter.UNKNOWN_TENDENCY) and (self.vertical_tendency == ParticleFilter.UNKNOWN_TENDENCY):
            self.horizontal_tendency = self.default_tendency
            self.vertical_tendency = self.default_tendency

        if self.horizontal_tendency == ParticleFilter.LEFT_TENDENCY: probability_grid = cv.bitwise_and(probability_grid, ParticleFilter.LEFT_MASK)
        if self.horizontal_tendency == ParticleFilter.RIGHT_TENDENCY: probability_grid = cv.bitwise_and(probability_grid, ParticleFilter.RIGHT_MASK)
        if self.vertical_tendency == ParticleFilter.UP_TENDENCY: probability_grid = cv.bitwise_and(probability_grid, ParticleFilter.UP_MASK)
        if self.vertical_tendency == ParticleFilter.DOWN_TENDENCY: probability_grid = cv.bitwise_and(probability_grid, ParticleFilter.DOWN_MASK)

        distance_error = max(int(distance_to_center*percentage_error/100),int(distance_to_line*percentage_error/100))
        if distance_error%2 == 0: distance_error+=1
        probability_grid = cv.GaussianBlur(probability_grid,(distance_error,distance_error),distance_error)

        not_zero_indexes = cv.findNonZero(probability_grid)
        if type(not_zero_indexes) != type(None):
            not_zero_indexes = not_zero_indexes.reshape(-1, not_zero_indexes.shape[-1])

            self.sensor_reset_particles = np.empty((len(not_zero_indexes), 3))
            self.sensor_reset_particles[:,0] = not_zero_indexes[:,0]
            self.sensor_reset_particles[:,1] = not_zero_indexes[:,1]
            self.sensor_reset_particles[:,2] = np.apply_along_axis(Line.get_direction,1,self.sensor_reset_particles,feature_center.astype(int))
            self.sensor_reset_particles = np.round(self.sensor_reset_particles,3)

            self.sensor_reset_weights = np.apply_along_axis(ParticleFilter.get_pixel_value,1,self.sensor_reset_particles,probability_grid)/255
            self.sensor_reset_weights = np.divide(self.sensor_reset_weights,sum(self.sensor_reset_weights))
        else:
            self.feature_in_fov = False
        return probability_grid, [feature_center_mask,feature_line_mask,feature_direction_mask]

    def get_pixel_value(pixel,img):
        return img[int(pixel[1]),int(pixel[0])]

    def compare_sensor_data(self, sensor_particle):
        diff_array = np.abs(np.divide((self.sensors_result-sensor_particle),(self.sensors_result+sensor_particle)))
        # diff_array = np.nan_to_num(diff_array,nan=1)
        
        score = np.prod(np.subtract([1],diff_array))
        return score

    def estimate_position(self):
        self.estimated_mean = np.average(self.particles, weights=self.weights, axis=0).astype(int)
        self.estimated_var  = np.average((self.particles - self.estimated_mean)**2, weights=self.weights, axis=0)
        self.estimated_deviation = (self.estimated_var)**0.5
        self.estimated_abs_deviation = (self.estimated_deviation[0]**2 + self.estimated_deviation[1]**2)**0.5
        if (self.estimated_abs_deviation < 100) and (self.ess() > self.emcl_ess_constant*self.emcl_threshold*self.N):
            if self.estimated_mean[0] < self.w_limits[1]*0.45: self.horizontal_tendency = ParticleFilter.LEFT_TENDENCY
            elif self.estimated_mean[0] > self.w_limits[1]*0.55: self.horizontal_tendency = ParticleFilter.RIGHT_TENDENCY
            else: self.horizontal_tendency = ParticleFilter.UNKNOWN_TENDENCY
            if self.estimated_mean[1] < self.h_limits[1]*0.45: self.vertical_tendency = ParticleFilter.UP_TENDENCY
            elif self.estimated_mean[1] > self.h_limits[1]*0.55: self.vertical_tendency = ParticleFilter.DOWN_TENDENCY
            else: self.vertical_tendency = ParticleFilter.UNKNOWN_TENDENCY
            robot_found = True
        else:
            robot_found = False
        self.emcl_entropy_labda = min(1,(self.current_entropy-self.prior_entropy)/self.prior_entropy)
        return robot_found, self.horizontal_tendency, self.vertical_tendency
    
    def resample_partiles(self):
        # # print('---------------------')
        # self.p_fast += self.n_fast*(self.probability_likehood-self.p_fast)
        # self.p_slow += self.n_slow*(self.probability_likehood-self.p_slow)
        # # print(f'Samples A-MCL: {int(self.N*min(1,max(0,1-self.amcl_ratio*self.p_fast/self.p_slow)))}')
        # if self.AMCL:
        #     new_samples_amount = int(self.N*min(1,max(0,1-self.amcl_ratio*self.p_fast/self.p_slow)))


        # resample_method = ''

        # if self.EMCL:
            
        print(f'ESS {round(self.ess(),3)}, threshold {round(self.emcl_ess_constant*self.emcl_threshold*self.N,3)}')
        print(f'Prior entropy {round(self.prior_entropy,3)}, current entropy {round(self.current_entropy,3)}, lambda = {round(self.emcl_entropy_labda,3)}')
        print(f'Likehood  {round(self.probability_likehood,3)}, threshold {round(self.emcl_threshold,3)}')
        if self.ess() < self.emcl_ess_constant*self.emcl_threshold*self.N:
            new_samples_amount = int(self.N-self.ess())
            resample_method = 'ESS insuficiente'
        elif self.emcl_entropy_labda >= 0.4:
            new_samples_amount = int((1-self.emcl_entropy_labda)*(self.N-self.ess()))
            resample_method = 'Variacao de entropia'
        elif self.probability_likehood < self.emcl_threshold:
            new_samples_amount = self.N
            resample_method = 'Qualidade de sensor insuficiente'
        else:
            new_samples_amount = 0
            resample_method = 'Amostras aceitaveis'

        # print(f'Samples EMCL: {new_samples_amount}')

        old_samples_amount = self.N-new_samples_amount

        if self.feature_in_fov:
            # Sensor Reseting
            indexes_sensor = ParticleFilter.systematic_resample(new_samples_amount,self.sensor_reset_weights)
            new_samples_particles = self.sensor_reset_particles[indexes_sensor]

            # Importance Resampling
            indexes_importance = ParticleFilter.systematic_resample(old_samples_amount,self.weights)
            old_samples_particles = self.particles[indexes_importance]

            self.particles[:new_samples_amount] = new_samples_particles
            self.particles[new_samples_amount:] = old_samples_particles
        else:
            new_samples_amount = int(new_samples_amount/10)
            old_samples_amount = self.N-new_samples_amount

            # Uniform distribution
            new_samples_particles = self.create_uniform_particles(new_samples_amount)

            # Importance Resampling
            indexes_importance = ParticleFilter.systematic_resample(old_samples_amount,self.weights)
            old_samples_particles = self.particles[indexes_importance]

            self.particles[:new_samples_amount] = new_samples_particles
            self.particles[new_samples_amount:] = old_samples_particles

        self.weights = [1]*len(self.particles)

        return resample_method, new_samples_amount

    def systematic_resample(N, weights):

        positions = (np.random.random() + np.arange(N)) / N
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes


    def ess(self):
        return 1. / np.sum(np.square(self.weights))
    