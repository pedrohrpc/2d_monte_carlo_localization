import cv2 as cv
import numpy as np
import Line

class Line:

    def __init__(self, init_point, end_point, is_feature = False):
        self.init_point = np.asarray(init_point)
        self.end_point = np.asarray(end_point)

        self.length = np.linalg.norm(self.end_point-self.init_point)

        self.is_feature = is_feature
        if is_feature: self.feature_center = init_point + (end_point-init_point)/2

        if self.end_point[0]==self.init_point[0]:
            self.slope = 90
        else:
            self.slope = round(np.rad2deg(np.arctan((self.end_point[1]-self.init_point[1])/((self.end_point[0]-self.init_point[0])))),3)
            if (self.end_point[0]-self.init_point[0])<0:
                self.slope += 180
            else:
                self.slope = (self.slope+360)%360

    def dist_to_point(self,point):
        point = np.asarray([point[0],point[1]])

        dist = np.abs(np.cross(self.end_point-self.init_point, self.init_point-point))/np.linalg.norm(self.end_point-self.init_point)

        return round(dist,3)
    
    def closest_point_in_line(self,point):
        direction = self.end_point-self.init_point
        a = (np.sum(direction*(point-self.init_point)))/cv.norm(direction)**2
        closest_point = self.init_point+a*direction
        return closest_point
    
    def get_points(self):
        return [self.init_point, self.end_point]
    
    def get_line_from_slope(origin_point,angle,lenght=5000):
        end_point = origin_point + lenght * np.asarray([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
        new_line = Line(origin_point,end_point)
        return new_line

    def get_intersection(line1: Line, line2: Line):
        direction1 = line1.end_point - line1.init_point
        direction2 = line2.end_point - line2.init_point

        if (np.cross(direction1,direction2) == 0): return None
        

        intersection = line1.init_point + direction1 * (np.cross(line2.init_point-line1.init_point,direction2))/np.cross(direction1,direction2)
        if ((np.linalg.norm(line1.init_point-intersection) > line1.length) or (np.linalg.norm(line1.end_point-intersection) > line1.length)): return None
        if ((np.linalg.norm(line2.init_point-intersection) > line2.length) or (np.linalg.norm(line2.end_point-intersection) > line2.length)): return None

        return intersection.astype(int)

    def get_direction(particle,point):
        direction = Line(particle[:2],point).slope
        return direction


    def join_lines(line1: Line, line2: Line):
        [p1, p2] = line1.get_points()
        [p3, p4] = line2.get_points()
        points = [p2,p3,p4]
        init_point = p1
        end_point = p1
        for point in points:
            if abs(line1.slope)<=45:
                if point[0] < init_point[0]:
                    init_point = point
                if point[0] > end_point[0]:
                    end_point = point
            else:
                if point[1] < init_point[1]:
                    init_point = point
                if point[1] > end_point[1]:
                    end_point = point
        new_line = Line(np.asarray(init_point),np.asarray(end_point))
        return new_line
    


    def __str__(self):
        return f'Init: {self.init_point} End: {self.end_point} Slope: {self.slope}'



