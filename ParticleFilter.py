import numpy as np
import cv2 as cv
from filterpy.monte_carlo import systematic_resample
from numpy.random import randn, uniform
from math import dist
from fov_utils import fov_utils as fov



class ParticleFilter():

    # Contructor
    def __init__(self, N, field_cv, fov_angle, fov_min_range, fov_max_range, player_position_deviation, player_angle_deviation, 
                 initial_position_known = False, initial_position = [0,0,0], standardDeviation = [0,0,0]):
        self.N = N
        self.fov_angle = fov_angle
        self.fov_min_range = fov_min_range
        self.fov_max_range = fov_max_range
        self.field_cv = field_cv
        self.field_h, self.field_w, c = self.field_cv.shape
        self.result_field_cv = self.field_cv.copy()

        self.player_position_deviation = player_position_deviation #Centimeters
        self.player_angle_deviation = player_angle_deviation #Degrees

        # Creating initial particles
        self.standardDeviation = standardDeviation
        if initial_position_known:
            self.particles = self.create_gaussian_particles(initial_position)
        else:
            self.particles = self.create_uniform_particles()

        # Creating weights array
        self.weights = [1]*len(self.particles)
        self.weights = np.divide(self.weights,sum(self.weights))

    # Gera particulas com distribuicao uniforme
    def create_uniform_particles(self):
        particles = np.empty((self.N, 3))
        particles[:, 0] = uniform(0, self.field_w, size=self.N)
        particles[:, 1] = uniform(0, self.field_h, size=self.N)
        particles[:, 2] = uniform(0, 360, size=self.N)
        particles[:, 2] += 360
        particles[:, 2] %= 360
        print(particles[:,0].max())
        return particles.astype(int)

    # Gera particulas com distribuicao gaussiana, utilizando uma media e um desvio padrao
    def create_gaussian_particles(self, initial_position):
        particles = np.empty((self.N, 3))
        particles[:, 0] = initial_position[0] + (randn(self.N) * self.standardDeviation[0])
        particles[:, 1] = initial_position[1] + (randn(self.N) * self.standardDeviation[1])
        particles[:, 2] = initial_position[2] + (randn(self.N) * self.standardDeviation[2])
        particles[:, 2] += 360
        particles[:, 2] %= 360
        return particles.astype(int)

    # Preve a posicao das particulas de acordo com a movimentacao do robo, com um erro
    def predict(self, delta_x, delta_y, delta_angle, deviation_x, deviation_y, deviation_angle):
        self.particles[:, 2] += (delta_angle + (randn(self.N) * deviation_angle)).astype(int)
        self.particles[:, 2] %= 360

        self.particles[:, 0] += (delta_x + (randn(self.N) * deviation_x)).astype(int)
        self.particles[:, 1] += (delta_y + (randn(self.N) * deviation_y)).astype(int)

    def predict(self, delta_pos, delta_angle):
        self.particles[:, 2] += (delta_angle + (randn(self.N) * self.player_angle_deviation)).round().astype(int)
        self.particles[:, 2] %= 360

        dist = delta_pos + (randn(self.N) * self.player_position_deviation)
        self.particles[:, 0] += (np.cos(self.particles[:, 2]*np.pi/180) * dist).round().astype(int)
        self.particles[:, 1] += (np.sin(self.particles[:, 2]*np.pi/180) * dist).round().astype(int)

    # Testa todas as particulas para atualizar seus pesos (utiliza a get_partice_pov())
    def testParticles(self, robot_fov):

        # for i, particle in enumerate(self.particles):
        #     # Gets a matrix with the point of view (POV) of the particle            
        #     particle_pov = fov.get_particle_pov(self.field_cv, particle[0], particle[1], particle[2], self.fov_angle, self.fov_min_range, self.fov_max_range, low_res = True)
            
        #     # Compares the particles pov to the robots pov to get a value that represents how similar they are
        #     # diff = (1-(np.sum(np.abs(np.subtract(robot_fov,particle_pov))))/(robot_fov.shape[0]*robot_fov.shape[1]/(180/self.fov_angle)))**2
        #     diff = 1/(np.sum(np.abs(np.subtract(robot_fov,particle_pov))))

        #     # Multplies the particles weight to the value obtained
        #     self.weights[i] *= diff

        self.robot_pov = robot_fov
        new_weights = np.apply_along_axis(self.testParticle,1,self.particles)
        self.weights = np.multiply(self.weights,new_weights)
        
        # Normalizes the weights
        self.weights = np.divide(self.weights,sum(self.weights))

    def testParticle(self,particle):
        # TODO: Testar a abordagem de distância
        # (max_y, max_x) = self.field_cv.shape[:2]
        # if (particle[0] > max_x) or (particle[0] < 0) or (particle[1] > max_y) or (particle[1] < 0):
        #     diff = 1e-10
        # else:
        particle_pov = fov.get_particle_pov(self.field_cv, particle[0], particle[1], particle[2], self.fov_angle, self.fov_min_range, self.fov_max_range)
        diff = 1/(np.sum(np.abs(np.subtract(self.robot_pov,particle_pov)))+1e-50)
        # print(diff)
        return diff    

    # Utilizada para verificar a necessidade de resample
    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    # Realiza o resample
    def resample_from_index(self):
        indexes = systematic_resample(self.weights)

        # resample according to indexes
        self.particles[:] = self.particles[indexes]
        self.weights.fill(1.0 / self.N)

    # Estima a posicao do robo atraves da media e variancia ponderada das particulas (nao estima a direcao)
    def estimate(self):
        pos = self.particles[:, 0:2]
        self.mean = np.average(pos, weights=self.weights, axis=0).astype(int)
        self.var  = np.average((pos - self.mean)**2, weights=self.weights, axis=0)
        self.deviation = (self.var)**0.5
        self.abs_deviation = (self.deviation[0]**2 + self.deviation[1]**2)**0.5


    def get_metrics_from_lines(lines, particle):
        metrics = []
        for line in lines:
            slope = round(((line.slope-particle[2]+90)+360)%360,3)
            metrics.append((line.dist_to_point([particle[0], particle[1]]),slope))
        dtype = [('dist', float), ('slope', float)]
        metrics = np.array(metrics,dtype=dtype)
        metrics = np.sort(metrics,order='dist')

        return metrics


    #### Utilities ####

    def particlesVisualMirror(self, thickness = 3, estimate_localization_color = [0,100,200]):
        self.result_field_cv = self.field_cv.copy()
        np.apply_along_axis(self.drawParticle,1,self.particles)
        cv.circle(self.result_field_cv,(int(self.mean[0]),int(self.mean[1])),int(self.abs_deviation),estimate_localization_color,thickness)

    def drawParticle(self, particle, size = 2, thickness = 2):
        particle_color = [255,0,0]
        cv.circle(self.result_field_cv,(particle[0],particle[1]),size,particle_color,thickness)
        # cv.line(self.result_field_cv,(particle[0],particle[1]),(int(particle[0]+size*3*np.cos(particle[2]*np.pi/180)),int(particle[1]+size*3*np.sin(particle[2]*np.pi/180))),particle_color,thickness)