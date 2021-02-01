# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 20:36:26 2020

@author: Administrator
"""
import arcpy
upstream_pntGeom = arcpy.PointGeometry(arcpy.Point(100.09,38.36),arcpy.SpatialReference(4326)) #2
middle_pntGeom = arcpy.PointGeometry(arcpy.Point(98.9,40.108333325),arcpy.SpatialReference(4326)) #middle
down_pntGeom = arcpy.PointGeometry(arcpy.Point(100.857,42.081),arcpy.SpatialReference(4326)) #1

#
# A list of coordinate pairs
#
pointList = [[100.09,38.36],[98.9,40.108333325],[100.857,42.081]]

# Create an empty Point object
#
point = arcpy.Point()

# A list to hold the PointGeometry objects
#
pointGeometryList = []
# For each coordinate pair, populate the Point object and create
#  a new PointGeometry
for pt in pointList:
    point.X = pt[0]
    point.Y = pt[1]

    pointGeometry = arcpy.PointGeometry(point,arcpy.SpatialReference(4326))
    pointGeometryList.append(pointGeometry)

# Create a copy of the PointGeometry objects, by using pointGeometryList
#  as input to the CopyFeatures tool.
filename = r"E:\modis.lst.interp\shape\Experiment.shape\center_point.shp"
arcpy.CopyFeatures_management(pointGeometryList, filename)

#substitute your centroid x/y coordinates 
#pixel resolution
PIXEL_SIZE = 0.008333333 

radius_list = [25,50,75,100]
for radius in radius_list:
    diameter = radius*2
    #substitute the distance from your circles verticies to the centroid 
    circleGeom = upstream_pntGeom.buffer(PIXEL_SIZE*radius) 
    filename = r"E:\modis.lst.interp\shape\Experiment.shape\upstream_%d.shp"%diameter
    # copying to a .shp will force densification  
    arcpy.CopyFeatures_management(circleGeom, filename) 
    
    #substitute the distance from your circles verticies to the centroid 
    circleGeom = middle_pntGeom.buffer(PIXEL_SIZE*radius) 
    filename = r"E:\modis.lst.interp\shape\Experiment.shape\middle_%d.shp"%diameter
    # copying to a .shp will force densification  
    arcpy.CopyFeatures_management(circleGeom, filename) 
    
    #substitute the distance from your circles verticies to the centroid 
    circleGeom = down_pntGeom.buffer(PIXEL_SIZE*radius) 
    filename = r"E:\modis.lst.interp\shape\Experiment.shape\down_%d.shp"%diameter
    # copying to a .shp will force densification  
    arcpy.CopyFeatures_management(circleGeom, filename) 