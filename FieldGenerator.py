import cv2 as cv
import numpy as np
from numpy import sin, cos
import ParticleFilter as pf
from Line import Line
import time


class FieldGenerator():


    def generate(scale_factor):
        # Empty field green field
        field_width = (600 + 2 * 100) * scale_factor
        field_lenght = (900 + 2 * 100) * scale_factor
        field = np.zeros((field_width, field_lenght,3),np.uint8)
        field[:] = [0,120,0]

        lineWidth = 5 * scale_factor
        field_lines = FieldGenerator.get_field_lines(scale_factor)
        feature_line, feature_center = FieldGenerator.get_field_features(scale_factor)
        center_feature_diameter = 150 * scale_factor

        for line in field_lines:
            cv.line(field,line.init_point,line.end_point,[255,255,255],lineWidth)

        cv.circle(field,feature_center,int(center_feature_diameter/2),[255,255,255],lineWidth)

        return field, field_lines, feature_line, feature_center
    
    def get_field_lines(scale_factor):
            
        fieldLenght = 900 * scale_factor
        fieldWidth = 600 * scale_factor
        goalDepth = 60 * scale_factor
        goalWidth = 260 * scale_factor
        goalAreaDepth = 100 * scale_factor
        goalAreaWidth = 300 * scale_factor
        padding = 100 * scale_factor
        penaltyAreaDepth = 200 * scale_factor
        penaltyAreaWidth = 500 * scale_factor

        #Intersections
        nwField = np.asarray([padding,padding])
        neField = np.asarray([padding+fieldLenght,padding])
        swField = np.asarray([padding,padding+fieldWidth])
        seField = np.asarray([padding+fieldLenght,padding+fieldWidth])

        middleN = np.asarray([int(padding+fieldLenght/2),padding])
        middleS = np.asarray([int(padding+fieldLenght/2),padding+fieldWidth])
        middle = np.asarray([int(padding+fieldLenght/2),int(padding+fieldWidth/2)])

        nwLGoalArea = np.asarray([padding,int(padding+fieldWidth/2-goalAreaWidth/2)])
        neLGoalArea = np.asarray([padding+goalAreaDepth,int(padding+fieldWidth/2-goalAreaWidth/2)])
        swLGoalArea = np.asarray([padding,int(padding+fieldWidth/2+goalAreaWidth/2)])
        seLGoalArea = np.asarray([padding+goalAreaDepth,int(padding+fieldWidth/2+goalAreaWidth/2)])

        nwRGoalArea = np.asarray([padding+fieldLenght-goalAreaDepth,int(padding+fieldWidth/2-goalAreaWidth/2)])
        neRGoalArea = np.asarray([padding+fieldLenght,int(padding+fieldWidth/2-goalAreaWidth/2)])
        swRGoalArea = np.asarray([padding+fieldLenght-goalAreaDepth,int(padding+fieldWidth/2+goalAreaWidth/2)])
        seRGoalArea = np.asarray([padding+fieldLenght,int(padding+fieldWidth/2+goalAreaWidth/2)])

        nwLGoal = np.asarray([padding-goalDepth,int(padding+fieldWidth/2-goalWidth/2)])
        neLGoal = np.asarray([padding,int(padding+fieldWidth/2-goalWidth/2)])
        swLGoal = np.asarray([padding-goalDepth,int(padding+fieldWidth/2+goalWidth/2)])
        seLGoal = np.asarray([padding,int(padding+fieldWidth/2+goalWidth/2)])

        nwRGoal = np.asarray([padding+fieldLenght,int(padding+fieldWidth/2-goalWidth/2)])
        neRGoal = np.asarray([padding+fieldLenght+goalDepth,int(padding+fieldWidth/2-goalWidth/2)])
        swRGoal = np.asarray([padding+fieldLenght,int(padding+fieldWidth/2+goalWidth/2)])
        seRGoal = np.asarray([padding+fieldLenght+goalDepth,int(padding+fieldWidth/2+goalWidth/2)])

        nwLPenaltyArea = np.asarray([padding,int(padding+fieldWidth/2-penaltyAreaWidth/2)])
        neLPenaltyArea = np.asarray([padding+penaltyAreaDepth,int(padding+fieldWidth/2-penaltyAreaWidth/2)])
        swLPenaltyArea = np.asarray([padding,int(padding+fieldWidth/2+penaltyAreaWidth/2)])
        seLPenaltyArea = np.asarray([padding+penaltyAreaDepth,int(padding+fieldWidth/2+penaltyAreaWidth/2)])

        nwRPenaltyArea = np.asarray([padding+fieldLenght-penaltyAreaDepth,int(padding+fieldWidth/2-penaltyAreaWidth/2)])
        neRPenaltyArea = np.asarray([padding+fieldLenght,int(padding+fieldWidth/2-penaltyAreaWidth/2)])
        swRPenaltyArea = np.asarray([padding+fieldLenght-penaltyAreaDepth,int(padding+fieldWidth/2+penaltyAreaWidth/2)])
        seRPenaltyArea = np.asarray([padding+fieldLenght,int(padding+fieldWidth/2+penaltyAreaWidth/2)])

        # Lines
        fieldLines = []
        fieldLines.append(Line(nwField,neField))
        fieldLines.append(Line(nwField,swField))
        fieldLines.append(Line(swField,seField))
        fieldLines.append(Line(neField,seField))

        fieldLines.append(Line(nwLGoalArea,neLGoalArea))
        fieldLines.append(Line(neLGoalArea,seLGoalArea))
        fieldLines.append(Line(swLGoalArea,seLGoalArea))

        fieldLines.append(Line(nwRGoalArea,neRGoalArea))
        fieldLines.append(Line(nwRGoalArea,swRGoalArea))
        fieldLines.append(Line(swRGoalArea,seRGoalArea))

        fieldLines.append(Line(nwLPenaltyArea,neLPenaltyArea))
        fieldLines.append(Line(neLPenaltyArea,seLPenaltyArea))
        fieldLines.append(Line(swLPenaltyArea,seLPenaltyArea))

        fieldLines.append(Line(nwRPenaltyArea,neRPenaltyArea))
        fieldLines.append(Line(nwRPenaltyArea,swRPenaltyArea))
        fieldLines.append(Line(swRPenaltyArea,seRPenaltyArea))

        fieldLines.append(Line(nwLGoal,neLGoal))
        fieldLines.append(Line(nwLGoal,swLGoal))
        fieldLines.append(Line(swLGoal,seLGoal))

        fieldLines.append(Line(nwRGoal,neRGoal))
        fieldLines.append(Line(neRGoal,seRGoal))
        fieldLines.append(Line(swRGoal,seRGoal))

        fieldLines.append(Line(middleN,middleS))

        return fieldLines

    def get_field_features(scale_factor):
        fieldLenght = 900 * scale_factor
        fieldWidth = 600 * scale_factor
        padding = 100 * scale_factor

        middleN = np.asarray([int(padding+fieldLenght/2),padding])
        middleS = np.asarray([int(padding+fieldLenght/2),padding+fieldWidth])
        middle = np.asarray([int(padding+fieldLenght/2),int(padding+fieldWidth/2)])

        feature_line = Line(middleN,middleS)
        feature_center = middle

        return feature_line, feature_center