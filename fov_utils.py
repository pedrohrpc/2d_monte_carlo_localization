import cv2 as cv
import numpy as np
from Line import Line

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
    
    
    def get_particle_pov(field_cv, particle, fov_angle, fov_min_range, fov_max_range, low_res = True):

        [particle_x, particle_y, particle_angle] = particle
        particle_angle = -particle_angle
        # Calc the are that will be cropped
        x, y = (fov_max_range, fov_max_range)
        vertice_1 = (int(x+fov_min_range*np.cos(np.radians(particle_angle+fov_angle/2))), int(y-fov_min_range*np.sin(np.radians(particle_angle+fov_angle/2))))
        vertice_2 = (int(x+fov_max_range*np.cos(np.radians(particle_angle+fov_angle/2))), int(y-fov_max_range*np.sin(np.radians(particle_angle+fov_angle/2))))
        vertice_3 = (int(x+fov_max_range*np.cos(np.radians(particle_angle-fov_angle/2))), int(y-fov_max_range*np.sin(np.radians(particle_angle-fov_angle/2))))
        vertice_4 = (int(x+fov_min_range*np.cos(np.radians(particle_angle-fov_angle/2))), int(y-fov_min_range*np.sin(np.radians(particle_angle-fov_angle/2))))
        fov_area = np.array([vertice_1, vertice_2, vertice_3, vertice_4])

        gray_field = cv.cvtColor(field_cv,cv.COLOR_BGR2GRAY)
        ret, gray_field = cv.threshold(gray_field,130,255,cv.THRESH_BINARY)
        # gray_field = gray_field+50
        # gray_field = cv.dilate(gray_field,np.ones((7, 7), np.uint8))

        # Get the fov (area)
        border = 2*fov_max_range
        # field_contrast = cv.addWeighted(field_cv, 3, np.zeros(field_cv.shape, field_cv.dtype), 1, -240) 
        expanded_field = cv.copyMakeBorder(gray_field, top=border, bottom=border, left=border, right=border, borderType=cv.BORDER_CONSTANT, value=[0])
        particle_x+= border
        particle_y+= border
        cropped_field = expanded_field[(int(particle_y)-fov_max_range):(int(particle_y)+fov_max_range),(int(particle_x)-fov_max_range):(int(particle_x)+fov_max_range)]
        mask = np.zeros_like(cropped_field)
        cv.fillConvexPoly(mask, fov_area, (255))
        particle_fov = cv.bitwise_and(cropped_field,mask)

        # Rotate fov for comparrison
        (h, w) = particle_fov.shape[:2]
        center = (w / 2, h / 2)
        # Perform the rotation
        M = cv.getRotationMatrix2D(center, 90-particle_angle, 1)
        pov = cv.warpAffine(particle_fov, M, (w, h))
        cropped_pov = pov[int(fov_max_range*(1-np.sin(np.radians(fov_angle/2)))):fov_max_range-fov_min_range,int(fov_max_range*(1-np.cos(np.radians(fov_angle/2)))):int(fov_max_range*(1+np.cos(np.radians(fov_angle/2))))]
        
        return cropped_pov

    def get_lines_in_pov(robot_pov: np.ndarray, slope_threshold = 50, dist_treshold = 20):

        # edges = cv.Canny(robot_pov, 50, 200, None, 3)
        edges = cv.erode(robot_pov,np.ones((3,3)))

        edges_colored = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

        linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 85, None, 20, 1)
        linesFiltered = []
        if linesP is not None:
            # print(f'number of lines: {len(linesP)}')
            for line in linesP:
                init_point = np.asarray([line[0][0],line[0][1]])
                end_point = np.asarray([line[0][2],line[0][3]])
                lineL = Line(init_point,end_point)
                match = False

                for lineR in linesFiltered:
                    if (abs(lineL.slope - lineR.slope) <= slope_threshold) or (abs(lineL.slope - lineR.slope) >= (360-slope_threshold)):
                        smallest_distance_between_line_points = min(cv.norm(lineL.init_point-lineR.end_point),cv.norm(lineL.init_point-lineR.init_point),cv.norm(lineL.end_point-lineR.end_point),cv.norm(lineL.end_point-lineR.init_point))
                        if (lineL.dist_to_point(lineR.init_point) <= dist_treshold) and (smallest_distance_between_line_points <= dist_treshold):
                            linesFiltered.remove(lineR)
                            linesFiltered.append(Line.join_lines(lineL,lineR))
                            match = True

                if not match:
                    linesFiltered.append(lineL)

            for line in linesFiltered:
                cv.line(edges_colored, line.init_point, line.end_point, (0,0,255), 2)
        
        circles = cv.HoughCircles(robot_pov,cv.HOUGH_GRADIENT_ALT,1.5,1000,param1=300,param2=0.6,minRadius=50,maxRadius=100)
        feature_center = None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv.circle(edges_colored,(i[0],i[1]),i[2],(0,0,255),2)
                cv.circle(edges_colored,(i[0],i[1]),4,(0,0,255),4)
                feature_center = np.asarray([i[0],i[1]])
                
        

        return edges_colored, linesFiltered, feature_center
    
