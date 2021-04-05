# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 20:04:05 2016

@author: Tan Junlei
Process MOD11A1,MOD11B1,MOD13A2 Data: Mosaic, Resample, Clip
Interpolated view_time view angle
Dir:
"""
import os,arcpy,time,datetime
import LST_Interp_Modules

def start_end_DOY(year):
    # Creating two objects for 1st January and 31st December of that particular year 
    # For that we are passing one argument (1) year
    # Return start day of year and end day of year in Integer  
    # Output Fromat is YYYYDOY, i.e. input 2001, output is 2001001 and 2001365
    startday = int(datetime.datetime(year, 1, 1).strftime('%Y%j'))
    endday  = int(datetime.datetime(year, 12, 31).strftime('%Y%j'))
    return startday,endday

starttime = time.time()
arcpy.env.overwriteOutput = True  # output file should be overwrite default
#### Boundary of Heihe River Basin
shapedir = 'E:/modis.lst.interp/shape/hrb.600.661'
shape_abspath = os.path.join(os.path.abspath(shapedir),'heihe_rectangle.shp')
####================ Common Variables =========================================
spatial_subset = '97.0 37.491666887 101.9999998 43.0'  #boundary coordinate
SPATIAL_UL_CORNER = '43.0 97.0'
SPATIAL_LR_CORNER = '37.491666887 101.9999998'
DATUM = 'WGS84' #coordinate system
OUTPUT_PIXEL_SIZE = 0.008333333 #pixel resolution
fillvalue = '65535'  #filled value when outside of boundary
year = 2012 # year
# startday,endday like 2007001
startday,endday = start_end_DOY(year);

####------Input data file------
dem_datadir = 'E:/modis.lst.interp/dem'  # dem
mod11a1_hdf_abspath = 'K:/modis/6/hrb/MOD11A1/%d' %(year) # MOD11A1 1Km lst hdf files
mod11b1_hdf_abspath = 'K:/modis/6/hrb/MOD11B1/%d' %(year) # MOD11B1 6Km lst hdf files
mod13a2_hdf_abspath = 'K:/modis/6/hrb/MOD13A2/%d' %(year) # MOD13A2 NDVI hdf files

####------OutPut data file ------
targetdir = 'E:/modis.lst.interp/dem/hrb.600.661' # output dem,aspect,slope size: 650*600
mod11a1_outdir = 'E:/modis.lst.interp/hrb.%d.C6/lst.MOD11A1/600.661' %(year)
mod11b1_outdir = 'E:/modis.lst.interp/hrb.%d.C6/lst.MOD11B1/600.661' %(year)
mod13a2_outdir = 'E:/modis.lst.interp/hrb.%d.C6/ndvi.MOD13A2/600.661'%(year)
mod11a1_int_dir = 'E:/modis.lst.interp/hrb.%d.C6/lst.MOD11A1/600.661/tif' %(year) # original lst file(integer type)
mod11b1_int_dir = 'E:/modis.lst.interp/hrb.%d.C6/lst.MOD11B1/600.661/tif' %(year) # original lst file(integer type)
mod11a1_real_dir = 'E:/modis.lst.interp/hrb.%d.C6/lst.MOD11A1/600.661/lst.ori' %(year) # Output real lst file(float type,K)


####------MYD INput/OutPut data file ------
# myd11a1_hdf_abspath = 'E:/modis/6/hrb/MYD11A1/%d' %(year) # MYD11A1 1Km lst hdf files
# myd11b1_hdf_abspath = 'E:/modis/6/hrb/MYD11B1/%d' %(year) # MYD11B1 6Km lst hdf files
# myd13a2_hdf_abspath = 'E:/modis/6/hrb/MYD13A2/%d' %(year) # MYD13A2 NDVI hdf files
# myd11a1_outdir = '../../hrb.%d.C6/lst.MYD11A1/600.661' %(year)
# myd11b1_outdir = '../../hrb.%d.C6/lst.MYD11B1/600.661' %(year)
# myd13a2_outdir = '../../hrb.%d.C6/ndvi.MYD13A2/600.661' %(year)
# myd11a1_int_dir = '../../hrb.%d.C6/lst.MYD11A1/600.661/tif' %(year) # original lst file(integer type)
# myd11b1_int_dir = '../../hrb.%d.C6/lst.MYD11B1/600.661/tif' %(year) # original lst file(integer type)
# myd11a1_real_dir = '../../hrb.%d.C6/lst.MYD11A1/600.661/lst.ori' %(year) # real lst file(float type,K)

#####==============================================================================
#####=======================Process DEM Data=======================================
##### relative dir of input and output
#dem_global = 'SRTM_1km.tif'     # global dem from SRTM
#dem_abspath = os.path.abspath(os.path.join(dem_datadir,dem_global))
#dem_hrb = 'hrb.dem.tif'      # output clipped dem file name
#if os.path.exists(targetdir) == False:
#    os.mkdir(targetdir)
#output_dem = os.path.abspath(os.path.join(targetdir,dem_hrb))
#slope_hrb = 'hrb.slope.tif'  # output clipped slope file name
#aspect_hrb = 'hrb.aspect.tif'  # output clipped slope file name
#
##print Output_dem
##### Process: Clip...
##### Extent of the file, Output PIXEL: 661*600  
#arcpy.Clip_management(dem_abspath,spatial_subset,output_dem,
#                      shape_abspath,"-32768","NONE","MAINTAIN_EXTENT")                      
#
##### Set local variables
#inRaster = output_dem
#outMeasurement = 'DEGREE'
#zFactor = '0.0000089992800575954'
##### Check out the ArcGIS Spatial Analyst extension license
#arcpy.CheckOutExtension("Spatial")
##### Execute Slope
#slop_abspath = os.path.abspath(os.path.join(targetdir,slope_hrb))
#outSlope = arcpy.gp.Slope_sa(inRaster, slop_abspath, outMeasurement, zFactor)
## Execute Aspect
#aspect_abspath = os.path.abspath(os.path.join(targetdir,aspect_hrb))
#outAspect = arcpy.gp.Aspect_sa(inRaster,aspect_abspath)
#print "DEM Data Processed to dem, slope,aspect Well Done"
###===========================================================================
###============= Process Data: MOD11A1, MOD13A2, MOD11B1======================
###================Process MOD11A1 Data=======================================
lstout_abspath = os.path.abspath(mod11a1_outdir)
lsttif_abspath = os.path.join(lstout_abspath,'tif')
lsttifclp_abspath = os.path.join(lstout_abspath,'clp')
temp_abspath = os.path.join(lstout_abspath,'tmp')
if os.path.exists(lstout_abspath) == False:
    os.makedirs(lstout_abspath)
    os.makedirs(lsttif_abspath)
    os.makedirs(lsttifclp_abspath)
    os.makedirs(temp_abspath)
## Create .prm file    
product_shortname = 'MOD11A1'
SPECTRAL_SUBSET = '1 1 1 1 1 1 1 1 0 0 0 0'
mrt_parameter = LST_Interp_Modules.MRT_PRM_FILE(product_shortname,mod11a1_hdf_abspath,SPECTRAL_SUBSET,
                                                SPATIAL_UL_CORNER,SPATIAL_LR_CORNER,DATUM,OUTPUT_PIXEL_SIZE)
mrt_parameter.Create() 
## Run Mosaic and Clip Process
mosaic_spectral = '1 1 1 1 1 1 1 1 0 0 0'
MOD11A1_pro = LST_Interp_Modules.MODIS_Process(product_shortname,startday,endday,mosaic_spectral,spatial_subset,
                                              fillvalue,mod11a1_hdf_abspath,lsttif_abspath,temp_abspath,shape_abspath,lsttifclp_abspath)
MOD11A1_pro.mosaic()
MOD11A1_pro.del_tempfile(mod11a1_hdf_abspath,'.txt')
####MOD11A1_pro.clip()
####===========================================================================
##### Processe MOD11B1 Data
lst6kmout_abspath=os.path.abspath(mod11b1_outdir)
lst6kmtif_abspath = os.path.join(lst6kmout_abspath,'tif')
lst6kmtifclp_abspath = os.path.join(lst6kmout_abspath,'clp')
temp_abspath = os.path.join(lst6kmout_abspath,'tmp')
if os.path.exists(lst6kmout_abspath) == False:
    os.makedirs(lst6kmout_abspath)
    os.makedirs(lst6kmtif_abspath)
    os.makedirs(lst6kmtifclp_abspath)
    os.makedirs(temp_abspath)
    
product_shortname = 'MOD11B1'
SPECTRAL_SUBSET = '1 1 1 1 1 1 1 1 1 1 1 1 1 1'
mrt_parameter = LST_Interp_Modules.MRT_PRM_FILE(product_shortname,mod11b1_hdf_abspath,SPECTRAL_SUBSET,
                                    SPATIAL_UL_CORNER,SPATIAL_LR_CORNER,DATUM,OUTPUT_PIXEL_SIZE)
mrt_parameter.Create() 

# Run Mosaic and Clip Process
mosaic_spectral = '1 1 1 1 1 1 1 1 0 0 0 1 1 1 0 0 0 0 0 0 1 1'
MOD11B1_pro = LST_Interp_Modules.MODIS_Process(product_shortname,startday,endday,mosaic_spectral,spatial_subset,fillvalue,
                                              mod11b1_hdf_abspath,lst6kmtif_abspath,temp_abspath,shape_abspath,lst6kmtifclp_abspath)
MOD11B1_pro.mosaic()
MOD11B1_pro.del_tempfile(mod11b1_hdf_abspath,'.txt')
###MOD11B1_pro.clip()
####=============================================================================
#### Processe MOD13A2 Data
ndviout_abspath = os.path.abspath(mod13a2_outdir)
ndvitif_abspath = os.path.join(ndviout_abspath,'tif')
ndvitifclp_abspath = os.path.join(ndviout_abspath,'clp')
temp_abspath = os.path.join(ndviout_abspath,'tmp')
if os.path.exists(ndviout_abspath) == False:
    os.makedirs(ndviout_abspath)
    os.makedirs(ndvitif_abspath)
    os.makedirs(ndvitifclp_abspath)
    os.makedirs(temp_abspath)
    
product_shortname = 'MOD13A2'
SPECTRAL_SUBSET = '1 0 0 0 0 0 0 0 0 0 0 0'
mrt_parameter = LST_Interp_Modules.MRT_PRM_FILE(product_shortname,mod13a2_hdf_abspath,SPECTRAL_SUBSET,
                                                SPATIAL_UL_CORNER,SPATIAL_LR_CORNER,DATUM,OUTPUT_PIXEL_SIZE)
mrt_parameter.Create() 
#### Run Mosaic and Clip Process
mosaic_spectral = '1 0 0 0 0 0 0 0 0 0 0 0'
MOD13A2_pro = LST_Interp_Modules.MODIS_Process(product_shortname,startday,endday,mosaic_spectral,spatial_subset,fillvalue,
                                              mod13a2_hdf_abspath,ndvitif_abspath,temp_abspath,shape_abspath,ndvitifclp_abspath)
MOD13A2_pro.mosaic()
MOD13A2_pro.del_tempfile(mod13a2_hdf_abspath,'.txt')
#MOD13A2_pro.clip()
endtime = time.time()
consumedhours = (endtime-starttime)/3600
print 'This Step Consumed Hours: '+ str('%.2f'% consumedhours)

################################################################################
################################################################################
arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True
####===========================================================================
####==================== Process Data: MOD11A1, MOD11B1========================
####=============================Common variables==============================
mod11a1_int_abspath = os.path.abspath(mod11a1_int_dir)
mod11a1_real_abspath = os.path.abspath(mod11a1_real_dir)
mod11b1_int_abspath = os.path.abspath(mod11b1_int_dir)
if (os.path.exists(mod11a1_real_abspath)==False):
    os.makedirs(mod11a1_real_abspath)  
####================Compute MOD11A1 from int to real temperature(K)============
arcpy.env.workspace = os.path.abspath(mod11a1_int_abspath)
rasters = arcpy.ListRasters('*.LST_Day_1km.tif','TIF')
for raster in rasters:  
    inraster = os.path.join(mod11a1_int_abspath,raster)
    Output_Raster_Dataset = os.path.join(mod11a1_real_abspath,raster)
    if os.path.isfile(Output_Raster_Dataset):
      os.remove(Output_Raster_Dataset)
    # Process: Raster Calculator
    raster_data = arcpy.sa.Times(inraster,0.02)
    raster_data.save(Output_Raster_Dataset)
rasters = arcpy.ListRasters('*.LST_Night_1km.tif','TIF')
for raster in rasters:  
    inraster = os.path.join(mod11a1_int_abspath,raster)
    Output_Raster_Dataset = os.path.join(mod11a1_real_abspath,raster)
    if os.path.isfile(Output_Raster_Dataset):
      os.remove(Output_Raster_Dataset)
    # Process: Raster Calculator
    raster_data = arcpy.sa.Times(inraster,0.02)
    raster_data.save(Output_Raster_Dataset)
print "MOD11A1 original LST were computed to real LST  Well Done"
####===============Process MOD11A1 ----Interpolation===========================
arcpy.env.workspace = mod11a1_int_abspath
# Local variables:
time_lt = arcpy.ListRasters('*view_time.tif','TIF')
angl_lt = arcpy.ListRasters('*view_angl.tif','TIF')
for input_dvt in time_lt:
    input_dvt_abspath = os.path.join(mod11a1_int_abspath,input_dvt) 
    LST_Interp_Modules.MOD11A1B1_ViewTimeAngl_Interp(input_dvt_abspath)
    #print input_dvt
for input_dangl in angl_lt:
    input_dangl_abspath = os.path.join(mod11a1_int_abspath,input_dangl) 
    LST_Interp_Modules.MOD11A1B1_ViewTimeAngl_Interp(input_dangl_abspath)
print 'MOD11A1 view_time angle were interpolated'
#####=======================Process MOD11B1----Interpolation===================
arcpy.env.workspace = mod11b1_int_abspath
angl_lt = arcpy.ListRasters('*view_angl.tif','TIF')
for input_dangl in angl_lt:
    input_dangl_abspath = os.path.join(mod11b1_int_abspath,input_dangl) 
    LST_Interp_Modules.MOD11A1B1_ViewTimeAngl_Interp(input_dangl_abspath)
print 'MOD11B1 view_angle were interpolated'

# Local variables:
time_lt = arcpy.ListRasters('*view_time.tif','TIF')
for input_dvt in time_lt:
    input_dvt_abspath = os.path.join(mod11b1_int_abspath,input_dvt) 
    LST_Interp_Modules.MOD11A1B1_ViewTimeAngl_Interp(input_dvt_abspath)
print 'MOD11B1 view_time were interpolated'
# Local variables:
emis_lt = arcpy.ListRasters('*Emis_29.tif','TIF')
for input_emi in emis_lt:
    input_emi_abspath = os.path.join(mod11b1_int_abspath,input_emi) 
    LST_Interp_Modules.MOD11A1B1_Emissivity_Interp(input_emi_abspath)
emis_lt = arcpy.ListRasters('*Emis_31.tif','TIF')
for input_emi in emis_lt:
    input_emi_abspath = os.path.join(mod11b1_int_abspath,input_emi) 
    LST_Interp_Modules.MOD11A1B1_Emissivity_Interp(input_emi_abspath)
emis_lt = arcpy.ListRasters('*Emis_32.tif','TIF')
for input_emi in emis_lt:
    input_emi_abspath = os.path.join(mod11b1_int_abspath,input_emi) 
    LST_Interp_Modules.MOD11A1B1_Emissivity_Interp(input_emi_abspath)    
print 'MOD11B1 Day_Emissivity were interpolated'



#####===========================================================================
#####============= Process Data: MYD11A1, MYD13A2, MYD11B1======================
#####===========================================================================
#####================Process MYD11A1 Data=======================================
#lstout_abspath = os.path.abspath(myd11a1_outdir)
#lsttif_abspath = os.path.join(lstout_abspath,'tif')
#lsttifclp_abspath = os.path.join(lstout_abspath,'clp')
#temp_abspath = os.path.join(lstout_abspath,'tmp')
#if os.path.exists(lstout_abspath) == False:
#    os.makedirs(lstout_abspath)
#    os.makedirs(lsttif_abspath)
#    os.makedirs(lsttifclp_abspath)
#    os.makedirs(temp_abspath)
#
#####==================== Create .prm file  ====================================  
#product_shortname = 'MYD11A1'
#SPECTRAL_SUBSET = "1 1 1 1 1 1 1 1 0 0 0 0"
#mrt_parameter = LST_Interp_Modules.MRT_PRM_FILE(product_shortname,myd11a1_hdf_abspath,SPECTRAL_SUBSET,
#                                                SPATIAL_UL_CORNER,SPATIAL_LR_CORNER,DATUM,OUTPUT_PIXEL_SIZE)
#mrt_parameter.Create() 
#
## Run Mosaic and Clip Process
#mosaic_spectral = '1 1 1 1 1 1 1 1 0 0 0'
#MOD11A1_pro = LST_Interp_Modules.MODIS_Process(product_shortname,startday,endday,mosaic_spectral,spatial_subset,
#                                               fillvalue,myd11a1_hdf_abspath,lsttif_abspath,temp_abspath,shape_abspath,lsttifclp_abspath)
#MOD11A1_pro.mosaic()
#MOD11A1_pro.del_tempfile(myd11a1_hdf_abspath,'.txt')
#####MOD11A1_pro.clip()
#####===========================================================================
#####==================Processe MYD11B1 Data====================================
#lst6kmout_abspath=os.path.abspath(myd11b1_outdir)
#lst6kmtif_abspath = os.path.join(lst6kmout_abspath,'tif')
#lst6kmtifclp_abspath = os.path.join(lst6kmout_abspath,'clp')
#temp_abspath = os.path.join(lst6kmout_abspath,'tmp')
#if os.path.exists(lst6kmout_abspath) == False:
#    os.makedirs(lst6kmout_abspath)
#    os.makedirs(lst6kmtif_abspath)
#    os.makedirs(lst6kmtifclp_abspath)
#    os.makedirs(temp_abspath)
#    
#product_shortname = 'MYD11B1'
#SPECTRAL_SUBSET = '1 1 1 1 1 1 1 1 1 1 1 1 1 1'
#mrt_parameter = LST_Interp_Modules.MRT_PRM_FILE(product_shortname,myd11b1_hdf_abspath,SPECTRAL_SUBSET,
#                                    SPATIAL_UL_CORNER,SPATIAL_LR_CORNER,DATUM,OUTPUT_PIXEL_SIZE)
#mrt_parameter.Create() 
#
##### Run Mosaic and Clip Process
#mosaic_spectral = '1 1 1 1 1 1 1 1 0 0 0 1 1 1 0 0 0 0 0 0 1 1'
#MOD11B1_pro = LST_Interp_Modules.MODIS_Process(product_shortname,startday,endday,mosaic_spectral,spatial_subset,fillvalue,
#                                               myd11b1_hdf_abspath,lst6kmtif_abspath,temp_abspath,shape_abspath,lst6kmtifclp_abspath)
#MOD11B1_pro.mosaic()
#MOD11B1_pro.del_tempfile(myd11b1_hdf_abspath,'.txt')
#####MOD11B1_pro.clip()
##
# ####=============================================================================
# #####======================== Processe MYD13A2 Data============================
# ndviout_abspath = os.path.abspath(myd13a2_outdir)
# ndvitif_abspath = os.path.join(ndviout_abspath,'tif')
# ndvitifclp_abspath = os.path.join(ndviout_abspath,'clp')
# temp_abspath = os.path.join(ndviout_abspath,'tmp')
# if os.path.exists(ndviout_abspath) == False:
#     os.makedirs(ndviout_abspath)
#     os.makedirs(ndvitif_abspath)
#     os.makedirs(ndvitifclp_abspath)
#     os.makedirs(temp_abspath)
    
# product_shortname = 'MYD13A2'
# SPECTRAL_SUBSET = "1 0 0 0 0 0 0 0 0 0 0 0"
# mrt_parameter = LST_Interp_Modules.MRT_PRM_FILE(product_shortname,myd13a2_hdf_abspath,SPECTRAL_SUBSET,
#                                                 SPATIAL_UL_CORNER,SPATIAL_LR_CORNER,DATUM,OUTPUT_PIXEL_SIZE)
# mrt_parameter.Create() 

# ####=================Run Mosaic and Clip Process===============================
# mosaic_spectral = '1 0 0 0 0 0 0 0 0 0 0 0'
# MOD13A2_pro = LST_Interp_Modules.MODIS_Process(product_shortname,startday,endday,mosaic_spectral,spatial_subset,fillvalue,
#                                                myd13a2_hdf_abspath,ndvitif_abspath,temp_abspath,shape_abspath,ndvitifclp_abspath)
# MOD13A2_pro.mosaic()
# MOD13A2_pro.del_tempfile(myd13a2_hdf_abspath,'.txt')
# ####MOD13A2_pro.clip()
# endtime = time.time()
# consumedhours = (endtime-starttime)/3600
# print 'This Step Consumed Hours: '+ str('%.2f'% consumedhours)

#######===========================================================================
#######==================== Process Data: MYD11A1, MYD11B1========================
#######=============================Common variables==============================
######=======Input file folder
#myd11a1_int_abspath = os.path.abspath(myd11a1_int_dir)
#myd11a1_real_abspath = os.path.abspath(myd11a1_real_dir)
#myd11b1_int_abspath = os.path.abspath(myd11b1_int_dir)
#if (os.path.exists(myd11a1_real_abspath)==False):
#    os.makedirs(myd11a1_real_abspath)  
######================Compute MYD11A1 from int to real temperature(K)============
#arcpy.env.workspace = os.path.abspath(myd11a1_int_dir)
#rasters = arcpy.ListRasters('*.LST_Day_1km.tif','TIF')
#for raster in rasters:  
#    inraster = os.path.join(myd11a1_int_abspath,raster)
#    Output_Raster_Dataset = os.path.join(myd11a1_real_abspath,raster)
#    if os.path.isfile(Output_Raster_Dataset):
#       os.remove(Output_Raster_Dataset)
#    # Process: Raster Calculator
#    ####raster_data = arcpy.sa.Times(inraster,0.02)-273.15
#    raster_data = arcpy.sa.Times(inraster,0.02)
#    raster_data.save(Output_Raster_Dataset)
#rasters = arcpy.ListRasters('*.LST_Night_1km.tif','TIF')
#for raster in rasters:  
#    inraster = os.path.join(myd11a1_int_abspath,raster)
#    Output_Raster_Dataset = os.path.join(myd11a1_real_abspath,raster)
#    if os.path.isfile(Output_Raster_Dataset):
#       os.remove(Output_Raster_Dataset)
#    # Process: Raster Calculator
#    raster_data = arcpy.sa.Times(inraster,0.02)
#    raster_data.save(Output_Raster_Dataset)
#print "MYD11A1 Original LST  were compute to real LST"
#####===============Process MYD11A1 ----Interpolation===========================
#arcpy.env.workspace = myd11a1_int_abspath
## Local variables:
#time_lt = arcpy.ListRasters('*view_time.tif','TIF')
#for input_dvt in time_lt:
#    input_dvt_abspath = os.path.join(myd11a1_int_abspath,input_dvt) 
#    LST_Interp_Modules.MOD11A1B1_ViewTimeAngl_Interp(input_dvt_abspath)
#    #print input_dvt
#angl_lt = arcpy.ListRasters('*view_angl.tif','TIF')
#for input_dangl in angl_lt:
#    input_dangl_abspath = os.path.join(myd11a1_int_abspath,input_dangl) 
#    LST_Interp_Modules.MOD11A1B1_ViewTimeAngl_Interp(input_dangl_abspath)
#print 'MYD11A1 view_time angle were interpolated'
#####================Process MOD11B1----Interpolation===========================
#arcpy.env.workspace = myd11b1_int_abspath
#angl_lt = arcpy.ListRasters('*view_angl.tif','TIF')
#for input_dangl in angl_lt:
#    input_dangl_abspath = os.path.join(myd11b1_int_abspath,input_dangl) 
#    LST_Interp_Modules.MOD11A1B1_ViewTimeAngl_Interp(input_dangl_abspath)
#print 'MYD11B1 view_angl angle were interpolated'
#
## Local variables:
#time_lt = arcpy.ListRasters('*view_time.tif','TIF')
#for input_dvt in time_lt:
#    input_dvt_abspath = os.path.join(myd11b1_int_abspath,input_dvt) 
#    LST_Interp_Modules.MOD11A1B1_ViewTimeAngl_Interp(input_dvt_abspath)
#print 'MYD11B1 view_time were interpolated'
#
## Local variables:
#emis_lt = arcpy.ListRasters('*Emis_29.tif','TIF')
#for input_emi in emis_lt:
#    input_emi_abspath = os.path.join(myd11b1_int_abspath,input_emi) 
#    LST_Interp_Modules.MOD11A1B1_Emissivity_Interp(input_emi_abspath)
#emis_lt = arcpy.ListRasters('*Emis_31.tif','TIF')
#for input_emi in emis_lt:
#    input_emi_abspath = os.path.join(myd11b1_int_abspath,input_emi) 
#    LST_Interp_Modules.MOD11A1B1_Emissivity_Interp(input_emi_abspath)
#emis_lt = arcpy.ListRasters('*Emis_32.tif','TIF')
#for input_emi in emis_lt:
#    input_emi_abspath = os.path.join(myd11b1_int_abspath,input_emi) 
#    LST_Interp_Modules.MOD11A1B1_Emissivity_Interp(input_emi_abspath)    
#print 'MYD11B1 Day_Emissivity were interpolated'