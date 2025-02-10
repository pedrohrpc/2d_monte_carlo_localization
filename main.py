# Example file showing a circle moving on screen
import pygame
import numpy as np
import cv2 as cv
import threading
from fov_utils import fov_utils as fov
from ParticleFilter import ParticleFilter as pf


class simulation:
    def __init__(self, fov_angle, fov_min_range, fov_max_range, visual_particles = True, visual_fov = True, fps = 60):
        # pygame setup
        pygame.init()
        self.field = pygame.image.load("2d_localization/images/soccer_field.jpg")
        self.field_w, self.field_h = self.field.get_size()
        self.screen = pygame.display.set_mode((self.field_w, self.field_h))
        self.clock = pygame.time.Clock()
        self.running = True
        self.fps = fps

        # Parameters
        self.dt = 0
        self.base_speed_esc = 50
        self.forward_speed_esc = self.base_speed_esc
        self.speed_ang = 50
        self.acceleration_esc = 50
        self.acceleration_ang = 50
        self.player = pygame.image.load("2d_localization/images/robot.png")
        self.player = pygame.transform.rotate(self.player, 90)
        self.player_w, self.player_h = self.player.get_size()

        self.particle_icon = pygame.image.load("2d_localization/images/particle.png")
        self.particle_icon_w, self.particle_icon_h = self.particle_icon.get_size()

        self.field_center_pos = pygame.Vector2(self.field_w/2, self.field_h/2)
        
        self.player_angle = 0
        self.player_pos = pygame.Vector2(self.field_center_pos)


        ### OPENCV FOV PARAMETERS ###
        self.fov_angle = fov_angle #degrees
        self.fov_min_range = fov_min_range #centimeters
        self.fov_max_range = fov_max_range #centimeters
        self.field_cv = cv.imread("2d_localization/images/soccer_field.jpg")
        self.particles_visual_ready = False

        ## Visual features (for debugging)
        self.visual_particles = visual_particles
        self.visual_fov = visual_fov

    # Used to rotate a image, in this case the player
    def blitRotate(surf, image, pos, originPos, angle):

        # offset from pivot to center
        image_rect = image.get_rect(topleft = (pos[0] - originPos[0], pos[1]-originPos[1]))
        offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center
        
        # roatated offset from pivot to center
        rotated_offset = offset_center_to_pivot.rotate(-angle)

        # roatetd image center
        rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)

        # get a rotated image
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)

        # rotate and blit the image
        surf.blit(rotated_image, rotated_image_rect)

    def update_sim(self):
        self.screen.blit(self.field, (0,0))
        #The player walking in the field
        simulation.blitRotate(self.screen, self.player, self.player_pos,(self.player_w/2, self.player_h/2), self.player_angle) 

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.player_pos.y -= np.sin(np.radians(self.player_angle)) * self.forward_speed_esc * self.dt
            self.player_pos.x += np.cos(np.radians(self.player_angle)) * self.forward_speed_esc * self.dt

        if keys[pygame.K_s]:
            self.player_pos.y += np.sin(np.radians(self.player_angle)) * self.base_speed_esc * self.dt
            self.player_pos.x -= np.cos(np.radians(self.player_angle)) * self.base_speed_esc * self.dt

        if keys[pygame.K_a]:
            self.player_pos.y -= np.sin(np.radians(self.player_angle+90)) * self.base_speed_esc * self.dt
            self.player_pos.x += np.cos(np.radians(self.player_angle+90)) * self.base_speed_esc * self.dt

        if keys[pygame.K_d]:
            self.player_pos.y += np.sin(np.radians(self.player_angle+90)) * self.base_speed_esc * self.dt
            self.player_pos.x -= np.cos(np.radians(self.player_angle+90)) * self.base_speed_esc * self.dt

        if keys[pygame.K_q]:
            self.player_angle += self.speed_ang * self.dt
        if keys[pygame.K_e]:
            self.player_angle -= self.speed_ang * self.dt


        # adding angular acceleration
        if (keys[pygame.K_q] or keys[pygame.K_e]) and self.speed_ang < 150:
            self.speed_ang += self.acceleration_ang * self.dt
        if (keys[pygame.K_q] == False and keys[pygame.K_e] == False):
            self.speed_ang = 50

        # adding escalar acceleration (forward)
        if keys[pygame.K_w] and self.forward_speed_esc < 150:
            self.forward_speed_esc += self.acceleration_esc * self.dt
        if keys[pygame.K_w] == False:
            self.forward_speed_esc = self.base_speed_esc

        if keys[pygame.K_SPACE]:
            self.player_pos = pygame.Vector2(self.field_center_pos)
            self.player_angle = 0

        if keys[pygame.K_ESCAPE]:
            self.running == False

        pygame.display.flip()

        # limits FPS to 60
        self.dt = self.clock.tick(self.fps) / 1000

    def robot_fov(self):
        # Calc the are that will be cropped
        robot_pov = fov.get_particle_pov(self.field_cv, self.player_pos[0], self.player_pos[1], self.player_angle, self.fov_angle, self.fov_min_range, self.fov_max_range, low_res = False)

        robot_pov_low_res = fov.get_particle_pov(self.field_cv, self.player_pos[0], self.player_pos[1], self.player_angle, self.fov_angle, self.fov_min_range, self.fov_max_range, low_res = True)
        
        if self.visual_fov:
            cv.imshow("Robot Field of View", robot_pov)
            cv.imshow("Robot Field of View Low Resolution", robot_pov_low_res)
        cv.waitKey(1)

        self.robot_pov = robot_pov_low_res
    
    def particle_result(self, particle_x, particle_y, particle_angle,robot_fov):
        particle_pov = fov.get_particle_pov(self.field_cv, pygame.Vector2(particle_x, particle_y), particle_angle, self.fov_angle, self.fov_min_range, self.fov_max_range, low_res = True)
        diff = (1-(np.sum(np.abs(np.subtract(robot_fov,particle_pov))))/(robot_fov.shape[0]*robot_fov.shape[1]/(180/self.fov_angle)))**2
        diff_percent = diff*100

        return particle_pov, diff_percent

    def runParticleFilter(self, particleFilter: pf, robotVariaton = None, desvioPos = None, desvioAngle = None):
        # Atualiza a posicao das particulas de acordo com o robo
        # particleFilter.predict(robotVariaton,(desvioPos,desvioAngle))
        # Testa o input das particulas (update)
        particleFilter.testParticles(self.robot_pov, self.field_cv)

        # Verifica se é necessario resample
        if particleFilter.neff() < (N/2):
            particleFilter.resample_from_index()

        # Calcula a media e variancia
        particleFilter.estimate()
        

    def update_particles_visual(self, robot_fov, particleFilter: pf):
        aux_count = 1
        weights = []
        # print('particles: ')
        if self.visual_particles:
            horizontal_strip = np.ones((int(robot_fov.shape[0]/10),robot_fov.shape[1]), robot_fov.dtype)
            l_concat = horizontal_strip.copy()
            r_concat = horizontal_strip.copy()
        
        for particle in particleFilter.particles:
            particle_pov, diff_percent = self.particle_result(particle[0], particle[1], particle[2],robot_fov)

            if self.visual_particles:
                print(f'partice {aux_count}:\nsum: {np.sum(particle_pov)}\ndiff: {diff_percent}%')
                if (aux_count<=N/2):
                    l_concat = np.concatenate((l_concat,particle_pov,horizontal_strip), axis=0)
                else:
                    r_concat = np.concatenate((r_concat,particle_pov,horizontal_strip), axis=0)
            aux_count += 1
            
            weights.append(diff_percent/100)

        if self.visual_particles:
            vertical_strip = np.ones((l_concat.shape[0],int(l_concat.shape[1]/10)), robot_fov.dtype)
            concat_result = np.concatenate((l_concat,vertical_strip,r_concat), axis=1)
            self.particles_visual = concat_result
            self.particles_visual_ready = True

        particleFilter.weights = weights
        particleFilter.weights = np.divide(particleFilter.weights,sum(particleFilter.weights))

if __name__ == '__main__':
    ### Simulation params
    exec_freq = 1 # times per second
    aux_count = 0
    player_initial_x = 200
    player_initial_y = 300
    player_initial_angle = 0
    playerInitPosKnown = True
    player_x_deviation = 10
    player_y_deviation = 10
    player_angle_deviation = 10

    ### Real robot params
    fov_max_range = 500 #centimeters
    fov_min_range = 50 #centimeters
    fov_angle = 90 #degrees

    ### Particle filter params
    N = 10
    # camHeight = 80
    # camAngle = np.pi/4
    # fov = (3/3) * np.pi
    # minRange = camAngle * np.tan(camAngle-fov/2)
    # maxRange = camAngle * np.tan(camAngle+fov/4)

    sim = simulation(fov_angle, fov_min_range, fov_max_range, visual_particles=False, visual_fov=False, fps = 30)
    particleFilter = pf(N,fov_angle,fov_min_range,fov_max_range, initial_position_known=playerInitPosKnown, 
                        mean = [player_initial_x,player_initial_y,player_initial_angle], standardDeviation = [player_x_deviation,player_y_deviation,player_angle_deviation], 
                        xRange = [0, sim.field_w], yRange = [0, sim.field_h], headingRange = [0, 360])



    while sim.running:
        # pygame.QUIT event means the user clicked X to close your window
        pygame.display.set_caption(f'Robot position - X: {int(sim.player_pos[0])} Y: {int(sim.player_pos[0])}')

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim.running = False

        sim.update_sim()
        robot_fov = sim.robot_fov()
        # print(robot_fov.max())
        # print(robot_fov.shape[0]*robot_fov.shape[1])

        particle_filter_thread = threading.Thread(target=sim.runParticleFilter, args=([particleFilter]), daemon=True)


        if aux_count == sim.fps/exec_freq:
            particle_filter_thread.start() # Runing in a different thread (for performance)
            # sim.runParticleFilter(particleFilter) # Running on the same thread (for debugging)
            print(f'Best estimate: \nMean: {particleFilter.mean}\nDeviation: {particleFilter.deviation}')
            print(f'Neff: {particleFilter.neff()}')
            print(f'N/2: {N/2}')
            aux_count = 0
        
        if sim.particles_visual_ready and sim.visual_particles:
            cv.imshow(f'Particles', sim.particles_visual)
        cv.waitKey(1)
        
        aux_count += 1

    pygame.quit()