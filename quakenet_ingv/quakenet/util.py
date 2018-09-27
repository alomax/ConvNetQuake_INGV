'''
Created on 3 Apr 2018

@author: anthony
'''


import math



def distance2classification(distance, num_dist_classes):
    
    dist_id = int(round(float(num_dist_classes) * math.pow(distance / 180.0, 0.5)))
    dist_id = min(max(dist_id, 0), num_dist_classes - 1)
   
    return dist_id


def classification2distance(dist_id, num_dist_classes):
    
    distance = 180.0 * math.pow(float(dist_id) / float(num_dist_classes), 2.0)
   
    return distance


def magntiude2classification(magnitude, num_mag_classes):
    
    mstep = 9.5 / float(num_mag_classes - 1);
    mag_id = int(round(magnitude / mstep))
    mag_id = min(max(mag_id, 0), num_mag_classes - 1)
   
    return mag_id


def classification2magnitude(mag_id, num_mag_classes):
    
    mstep = 9.5 / float(num_mag_classes - 1);
    magnitude = mstep * float(mag_id)
   
    return magnitude


def depth2classification(depth, num_depth_classes):
    
    depth_id = int(round(float(num_depth_classes) * math.pow(depth / 700.0, 0.5)))
    depth_id = min(max(depth_id, 0), num_depth_classes - 1)
   
    return depth_id


def classification2depth(depth_id, num_depth_classes):
    
    depth = 700.0 * math.pow(float(depth_id) / float(num_depth_classes), 2.0)
   
    return depth


def azimuth2classification(azimuth, num_az_classes):
    
    mstep = 360.0 / float(num_az_classes);
    mag_id = int(round(azimuth / mstep))
    mag_id = min(max(mag_id, 0), num_az_classes - 1)
   
    return mag_id


def classification2azimuth(az_id, num_az_classes):
    
    mstep = 360.0 / float(num_az_classes);
    azimuth = mstep * float(az_id)
   
    return azimuth



