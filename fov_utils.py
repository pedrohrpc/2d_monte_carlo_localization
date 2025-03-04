import cv2 as cv
import numpy as np

class fov_utils:

    # Gets the trapezoid that limits the robot FOV
    def get_fov_trapezoid(robot_angle, fov_angle, min_range, max_range):

        x, y = (max_range, max_range)
        vertice_1 = (int(x+min_range*np.cos(np.radians(robot_angle+fov_angle/2))), int(y-min_range*np.sin(np.radians(robot_angle+fov_angle/2))))
        vertice_2 = (int(x+max_range*np.cos(np.radians(robot_angle+fov_angle/2))), int(y-max_range*np.sin(np.radians(robot_angle+fov_angle/2))))
        vertice_3 = (int(x+max_range*np.cos(np.radians(robot_angle-fov_angle/2))), int(y-max_range*np.sin(np.radians(robot_angle-fov_angle/2))))
        vertice_4 = (int(x+min_range*np.cos(np.radians(robot_angle-fov_angle/2))), int(y-min_range*np.sin(np.radians(robot_angle-fov_angle/2))))
        

        return np.array([vertice_1, vertice_2, vertice_3, vertice_4])

    # Crops said FOV from the field to represent player's POV and returns it as a cv2 image
    def get_fov(field, x, y, range, fov_area):
        border = 2*range
        field_contrast = cv.addWeighted(field, 3, np.zeros(field.shape, field.dtype), 1, -240) 
        expanded_field = cv.copyMakeBorder(field_contrast, top=border, bottom=border, left=border, right=border, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])
        x+= border
        y+= border
        cropped_field = expanded_field[(int(y)-range):(int(y)+range),(int(x)-range):(int(x)+range)]
        mask = np.zeros_like(cropped_field)
        # mask = cv.circle(mask, (range,range), range, (255,255,255), -1)
        cv.fillConvexPoly(mask, fov_area, (255,255,255))
        robot_fov = cv.bitwise_and(cropped_field,mask)
        
        return robot_fov

    # Used to rotate an cv2 image, in this case the player's POV for better perspective
    def cv_rotate(image, angle, center = None, scale = 1.0):
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        # Perform the rotation
        M = cv.getRotationMatrix2D(center, angle, scale)
        rotated = cv.warpAffine(image, M, (w, h))

        return rotated

    # Reduces the resolution of the FOV to make it lighter
    def to_low_res(image,compress_ratio):
        (h, w) = image.shape[:2]
        gray_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        blured_image = cv.GaussianBlur(gray_image, (compress_ratio,compress_ratio), 0)
        # blured_image = (gray_image)
        compressed_image = cv.resize(blured_image, (int(w/compress_ratio), int(h/compress_ratio)))
        compressed_image = compressed_image/255
        return compressed_image
    
    
    def get_particle_pov(field_cv, particle_x, particle_y, particle_angle, fov_angle, fov_min_range, fov_max_range, low_res = True):
        # Calc the are that will be cropped
        fov_area = fov_utils.get_fov_trapezoid(particle_angle, fov_angle, fov_min_range, fov_max_range)
        # Get the fov (area)
        robot_fov = fov_utils.get_fov(field_cv, particle_x, particle_y, fov_max_range, fov_area)

        # Rotate fov for comparrison
        pov = fov_utils.cv_rotate(robot_fov, 90-particle_angle)[int(fov_max_range*(1-np.sin(np.radians(fov_angle/2)))):fov_max_range-fov_min_range,int(fov_max_range*(1-np.cos(np.radians(fov_angle/2)))):int(fov_max_range*(1+np.cos(np.radians(fov_angle/2))))]

        # Using low resolution for particle comparison
        if low_res: pov = fov_utils.to_low_res(pov,compress_ratio=5)
        
        return pov
    
