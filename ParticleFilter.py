import numpy as np
from filterpy.monte_carlo import systematic_resample
from numpy.random import randn, uniform
from math import dist
from fov_utils import fov_utils as fov


class ParticleFilter():

    # Contructor
    def __init__(self, N, fov_angle, fov_min_range, fov_max_range, initial_position_known = False, mean = [0,0,0], standardDeviation = [0,0,0], xRange = 0, yRange = 0, headingRange = 0):
        self.N = N
        self.fov_angle = fov_angle
        self.fov_min_range = fov_min_range
        self.fov_max_range = fov_max_range

        # Creating initial particles
        if initial_position_known:
            self.particles = self.create_gaussian_particles(mean, standardDeviation)
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
    def create_gaussian_particles(self, mean, standardDeviation):
        particles = np.empty((self.N, 3))
        particles[:, 0] = mean[0] + (randn(self.N) * standardDeviation[0])
        particles[:, 1] = mean[1] + (randn(self.N) * standardDeviation[1])
        particles[:, 2] = mean[2] + (randn(self.N) * standardDeviation[2])
        particles[:, 2] += 360
        particles[:, 2] %= 360
        return particles.astype(int)

    # Preve a posicao das particulas de acordo com a movimentacao do robo, com um erro
    def predict(self, passo, erro):
        self.particles[:, 2] += (passo[1] + (randn(self.N) * erro[1])).astype(int)
        self.particles[:, 2] %= 360

        dist = passo[0] + (randn(self.N) * erro[0])
        self.particles[:, 0] += (np.cos(self.particles[:, 2]*np.pi/180) * dist).astype(int)
        self.particles[:, 1] += (np.sin(self.particles[:, 2]*np.pi/180) * dist).astype(int)

    # (NOT NEEDED ANYMORE)
    def testParticlesOld(self, answer):
        for i, particle in enumerate(self.particles):
            particleAnswer = []
            particleAnswer = self.checkFOV(particle)
            
            '''if len(answer) != 0:
                checking = ParticleFilter.compareIntersectionLists(answer,particleAnswer)/len(answer)
            elif len(particleAnswer) != 0:
                checking = 1/(1+len(particleAnswer))
            else:
                checking = 1'''
            
            intersectionsAnswer = 0
            intersectionsParticle = 0

            for intersection in answer:
                intersectionsAnswer += intersection[2]

            for intersection in particleAnswer:
                intersectionsParticle += intersection[2]

            if intersectionsAnswer == 0 and intersectionsParticle == 0:
                checking = 1
            else:
                checking = 1 - abs(intersectionsAnswer-intersectionsParticle)*2/max([intersectionsAnswer,intersectionsParticle])

            self.weights[i] *= checking
            self.weights[i] += 1e-300

        self.weights = np.divide(self.weights,sum(self.weights))


    # Testa todas as particulas para atualizar seus pesos (utiliza a get_partice_pov())
    def testParticles(self, robot_fov, field_cv):

        for i, particle in enumerate(self.particles):
            # Gets a matrix with the point of view (POV) of the particle            
            particle_pov = fov.get_particle_pov(field_cv, particle[0], particle[1], particle[2], self.fov_angle, self.fov_min_range, self.fov_max_range, low_res = True)
            
            if particle_pov.size == 0: print('4')
            # Compares the particles pov to the robots pov to get a value that represents how similar they are
            diff = (1-(np.sum(np.abs(np.subtract(robot_fov,particle_pov))))/(robot_fov.shape[0]*robot_fov.shape[1]/(180/self.fov_angle)))**2

            # Multplies the particles weight to the value obtained
            self.weights[i] *= diff

        # Normalizes the weights
        self.weights = np.divide(self.weights,sum(self.weights))

    # Verifica quais intersecoes estao na linha de visao de uma unica particula (NOT NEEDED ANYMORE)
    def checkFOV(self, particle):
        seen = []
        for intersection in self.intersections:
            distance = dist((particle[0],particle[1]),intersection[0])
            distX = particle[0] - intersection[0][0]
            distY = particle[1] - intersection[0][1]
            distance = (distX**2 + distY**2)**0.5
            
            if distance <= self.fov_max_range and distance >= self.fov_min_range:

                if distX == 0 and distY<0:
                    angle = np.pi/2
                elif distX == 0 and distY>0:
                    angle = -np.pi/2
                elif distY == 0 and distX<0:
                    angle = 0
                elif distY == 0 and distX>0:
                    angle = np.pi
                elif distX<0:
                    angle = np.arctan(distY/distX)
                elif distX>0:
                    angle = np.arctan(distY/distX) - np.pi*np.sign(np.arctan(distY/distX))
                else: 
                    pass

                angle = (angle + 2*np.pi)%(2*np.pi)

                limLeft = (particle[2]*np.pi/180+self.fov_angle/2)%(2*np.pi)
                limRight = (particle[2]*np.pi/180-self.fov_angle/2)%(2*np.pi)
                if limLeft >= limRight and angle <= limLeft and angle >= limRight:
                    seen.append(intersection)
                elif limLeft < limRight and (angle <= limLeft or angle >= limRight):
                    seen.append(intersection)

        return seen

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



    #### Utilities ####

    # Utilizada para atualizar a posicao do robo 2D simulado
    # def moveRobot(robot, passo):
    #     robot[2] += passo[1]
    #     robot[0] += int(passo[0]*np.cos(robot[2]*np.pi/180))
    #     robot[1] += int(passo[0]*np.sin(robot[2]*np.pi/180))
    #     return robot

    # Utilizada para calcular a quantidade de elementos presentes nas duas listas (NOT NEEDED ANYMORE)
    def compareIntersectionLists(list1, list2):
        result = 0
        for intersection1 in list1:
            for intersection2 in list2:
                if intersection1[2] == intersection2[2]:
                    result += 1
                    list2.remove(intersection2)
        return result