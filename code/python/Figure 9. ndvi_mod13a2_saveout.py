# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 20:13:51 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
import os
import arcpy

arcpy.env.workspace = "E:/modis.lst.interp/hrb.2018.C6/ndvi.MOD13A2/600.661/tif"
rasters = arcpy.ListRasters("*","tif")
out_path = "E:/modis.lst.interp/hrb.2018.C6/ndvi.MOD13A2/600.661/cal.ndvi/"
for raster in rasters:
    (filepath, fullname) = os.path.split(raster)
    (prename, suffix) = os.path.splitext(fullname)
    print(prename)
    arcpy.CheckOutExtension("ImageAnalyst") #检查许可
    arcpy.CheckOutExtension("spatial") #检查许可
    whereClause = "VALUE = -3000" #无效值
    outSetNull = arcpy.sa.SetNull(raster, raster, whereClause) * 0.0001 #去除无效值并乘以0.0001
    #outname=r"E:\MODISNDVI\BYNDVI\try1.tif" #输出路径
    outSetNull.save(out_path + prename + '_ndvi_qcwxz.tif') #保存数据
    print('over')