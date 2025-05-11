import cv2 as cv
import numpy as np

class Line:

    # def __init__(self, point_array):
    #     self.initPoint = np.asarray([point_array[0],point_array[1]])
    #     self.endPoint = np.asarray([point_array[2],point_array[3]])

    #     self.length = np.linalg.norm(self.endPoint-self.initPoint)

    #     if self.endPoint[0]==self.initPoint[0]:
    #         self.slope = 90
    #     else:
    #         self.slope = np.rad2deg(np.arctan((self.endPoint[1]-self.initPoint[1])/((self.endPoint[0]-self.initPoint[0]))))

    def __init__(self, init_point, end_point):
        self.initPoint = init_point
        self.endPoint = end_point

        self.length = np.linalg.norm(self.endPoint-self.initPoint)

        if self.endPoint[0]==self.initPoint[0]:
            self.slope = 90
        else:
            self.slope = round(np.rad2deg(np.arctan(-(self.endPoint[1]-self.initPoint[1])/((self.endPoint[0]-self.initPoint[0])))),3)
            self.slope = (self.slope+360)%360

    def dist_to_point(self,point):
        point = np.asarray([point[0],point[1]])

        dist = np.abs(np.cross(self.endPoint-self.initPoint, self.initPoint-point))/np.linalg.norm(self.endPoint-self.initPoint)

        return round(dist,3)
    
    def get_points(self):
        return [self.initPoint, self.endPoint]

    def join_lines(line1, line2):
        [p1, p2] = line1.get_points()
        [p3, p4] = line2.get_points()
        points = [p2,p3,p4]
        initPoint = p1
        endPoint = p1
        for point in points:
            if abs(line1.slope)<=45:
                if point[0] < initPoint[0]:
                    initPoint = point
                if point[0] > endPoint[0]:
                    endPoint = point
            else:
                if point[1] < initPoint[1]:
                    initPoint = point
                if point[1] > endPoint[1]:
                    endPoint = point
        new_line = Line(initPoint,endPoint)
        return new_line
    
    def __str__(self):
        return f'Init: {self.initPoint} End: {self.endPoint} Slope: {self.slope}'



