import numpy as np
import cv2 as cv
from filterpy.monte_carlo import systematic_resample
from numpy.random import randn, uniform
from math import dist
from fov_utils import fov_utils as fov



class ParticleFilter():

    # Contructor
    def __init__(self, N, field_cv, fov_angle, fov_min_range, fov_max_range, player_position_deviation, player_angle_deviation, 
                 initial_position_known = False, initial_position = [0,0,0], standardDeviation = [0,0,0], xRange = 0, yRange = 0, headingRange = 0):
        self.N = N
        self.fov_angle = fov_angle
        self.fov_min_range = fov_min_range
        self.fov_max_range = fov_max_range
        self.field_cv = field_cv
        self.result_field_cv = self.field_cv.copy()

        self.player_position_deviation = player_position_deviation #Centimeters
        self.player_angle_deviation = player_angle_deviation #Degrees

        # Creating initial particles
        self.standardDeviation = standardDeviation
        if initial_position_known:
            self.particles = self.create_gaussian_particles(initial_position)
        else:
            self.particles = self.create_uniform_particles(xRange, yRange, headingRange)

        # Creating weights array
        self.weights = [1]*len(self.particles)
        self.weights = np.divide(self.weights,sum(self.weights))

    # Gera particulas com distribuicao uniforme
    def create_uniform_particles(self, xRange, yRange, headingRange):
        particles = np.empty((self.N, 3))
        particles[:, 0] = uniform(xRange[0], xRange[1], size=self.N)
        particles[:, 1] = uniform(yRange[0], yRange[1], size=self.N)
        particles[:, 2] = uniform(headingRange[0], headingRange[1], size=self.N)
        particles[:, 2] += 360
        particles[:, 2] %= 360
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
    def predict(self, delta_pos, delta_angle):
        self.particles[:, 2] += (delta_angle + (randn(self.N) * self.player_angle_deviation)).astype(int)
        self.particles[:, 2] %= 360

        dist = delta_pos + (randn(self.N) * self.player_position_deviation)
        self.particles[:, 0] += (np.cos(self.particles[:, 2]*np.pi/180) * dist).astype(int)
        self.particles[:, 1] += (np.sin(self.particles[:, 2]*np.pi/180) * dist).astype(int)

    # Testa todas as particulas para atualizar seus pesos (utiliza a get_partice_pov())
    def testParticles(self, robot_fov):

        for i, particle in enumerate(self.particles):
            # Gets a matrix with the point of view (POV) of the particle            
            particle_pov = fov.get_particle_pov(self.field_cv, particle[0], particle[1], particle[2], self.fov_angle, self.fov_min_range, self.fov_max_range, low_res = True)
            
            if particle_pov.size == 0: print('4')
            # Compares the particles pov to the robots pov to get a value that represents how similar they are
            diff = (1-(np.sum(np.abs(np.subtract(robot_fov,particle_pov))))/(robot_fov.shape[0]*robot_fov.shape[1]/(180/self.fov_angle)))**2

            # Multplies the particles weight to the value obtained
            self.weights[i] *= diff

        # Normalizes the weights
        self.weights = np.divide(self.weights,sum(self.weights))

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



    #### Utilities ####

    def particlesVisualMirror(self, size = 2, thickness = 3, particle_color = [255,0,0], estimate_localization_color = [0,100,200]):
        self.result_field_cv = self.field_cv.copy()
        for particle in self.particles:
            cv.circle(self.result_field_cv,(particle[0],particle[1]),size,particle_color,thickness)
            cv.line(self.result_field_cv,(particle[0],particle[1]),(int(particle[0]+size*3*np.cos(particle[2]*np.pi/180)),int(particle[1]+size*3*np.sin(particle[2]*np.pi/180))),particle_color,thickness)

        cv.circle(self.result_field_cv,(int(self.mean[0]),int(self.mean[1])),int(self.abs_deviation),estimate_localization_color,thickness)


    def drawParticles(self, drawFov=False):
        for particle in self.particles:
            coloredField = self.drawParticle(particle, drawFov=drawFov)

        return coloredField
    
    def drawParticle(self, particle, drawFov=True, color=[255,0,0], robo=False):
        if robo: size = 3
        else: size = 2

        cv.circle(self.field_cv,(particle[0],particle[1]),size,color,size)

        if drawFov:
            cv.line(self.field_cv,(int(particle[0]+self.fov_min_range*np.cos(particle[2]*np.pi/180+self.fov_angle/2)),int(particle[1]+self.fov_min_range*np.sin(particle[2]*np.pi/180+self.fov_angle/2))),(int(particle[0]+self.fov_max_range*np.cos(particle[2]*np.pi/180+self.fov_angle/2)),int(particle[1]+self.fov_max_range*np.sin(particle[2]*np.pi/180+self.fov_angle/2))),[150,0,0],size)
            cv.line(self.field_cv,(int(particle[0]+self.fov_min_range*np.cos(particle[2]*np.pi/180-self.fov_angle/2)),int(particle[1]+self.fov_min_range*np.sin(particle[2]*np.pi/180-self.fov_angle/2))),(int(particle[0]+self.fov_max_range*np.cos(particle[2]*np.pi/180-self.fov_angle/2)),int(particle[1]+self.fov_max_range*np.sin(particle[2]*np.pi/180-self.fov_angle/2))),[150,0,0],size)
            cv.ellipse(self.field_cv, (particle[0],particle[1]), (self.fov_min_range,self.fov_min_range), particle[2]*np.pi/180, int(particle[2]-180*self.fov_angle/(np.pi*2)), int(particle[2]+180*self.fov_angle/(np.pi*2)), [150,0,0], size)
            cv.ellipse(self.field_cv, (particle[0],particle[1]), (self.fov_max_range,self.fov_max_range), particle[2]*np.pi/180, int(particle[2]-180*self.fov_angle/(np.pi*2)), int(particle[2]+180*self.fov_angle/(np.pi*2)), [150,0,0], size)
        else:
            cv.line(self.field_cv,(particle[0],particle[1]),(int(particle[0]+size*3*np.cos(particle[2]*np.pi/180)),int(particle[1]+size*3*np.sin(particle[2]*np.pi/180))),color,size)

        return self.field_cv