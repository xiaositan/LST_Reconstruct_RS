# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 07:31:11 2020

@author: Administrator
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import arcpy
import inspect
import collections
import pandas as pd
#from osgeo import gdal
#Owed by: http://blog.csdn.net/chunyexiyu
#direct get the input name from called function code
def retrieve_name_ex(var):
    stacks = inspect.stack()
    try:
        callFunc = stacks[1].function
        code = stacks[2].code_context[0]
        startIndex = code.index(callFunc)
        startIndex = code.index("(", startIndex + len(callFunc)) + 1
        endIndex = code.index(")", startIndex)
        return code[startIndex:endIndex].strip()
    except:
        return ""
def outputVar(var):
    print("{} = {}".format(retrieve_name_ex(var),var))

def RasterInfo(rasterfile):
#==============================================================================
    # #==========================
    # #===GDAL method============
    # gdal.UseExceptions()
    # ds = gdal.Open(FILE_NAME)
    # band = ds.GetRasterBand(1)
    # dataarray = band.ReadAsArray()
    # nrows,ncols = qaarray.shape
    # x0,dx,dxdy,y0,dydx,dy = ds.GetGeoTransform()
    # x1 = x0+dx*ncols
    # y1 = y0+dy*nrows
    # ============================
    # ===arcpy method============
     refraster = arcpy.Raster(rasterfile)
     # corner cordinates(Lower Left  Corner Longitude(Xmin) and Latitude(Ymin))
     # corner cordinates(Upper Right Corner Max Longitude(Xmax) and Latitude(Ymax))
     crncords = [refraster.extent.XMin,refraster.extent.YMin,refraster.extent.XMax,refraster.extent.YMax]
     cellSize = refraster.meanCellWidth
     data = arcpy.RasterToNumPyArray(rasterfile)
     nrows,ncols = data.shape
     return data,cellSize,crncords
#==============================================================================
def RasterPlot_wgs84_p1(dataarry,crncords,shpfile,figtitle,outfigname,vmin,vmax):
    llcrnrlon = crncords[0]
    llcrnrlat = crncords[1]
    urcrnrlon = crncords[2]
    urcrnrlat = crncords[3]
    #print llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon
    nrows,ncols = dataarry.shape
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

    plt.rc('font', **font)
    fig = plt.figure(figsize=(5,8),tight_layout=False)
    ax = fig.add_subplot(111)
    mask = dataarry<-9998
    dataarry=np.ma.masked_array(dataarry,mask=mask)
    ###  Projection merc or cyl
    m = Basemap(projection='cyl',llcrnrlat=llcrnrlat, urcrnrlat = urcrnrlat,
                llcrnrlon=llcrnrlon, urcrnrlon = urcrnrlon)
#    m.drawparallels(np.linspace(int(llcrnrlat), int(urcrnrlat), int(urcrnrlat)-int(llcrnrlat)+1),
#                    linewidth = 0.1,labels=[1, 0, 0, 0], fmt="%d",dashes=[1, 6])
#    m.drawmeridians(np.linspace( int(urcrnrlon),int(llcrnrlon),int(urcrnrlon)-int(llcrnrlon)+1), 
#                    linewidth = 0.1, labels=[0, 0, 0, 1], fmt="%d",dashes=[1, 6])
    m.drawparallels(np.linspace(llcrnrlat, urcrnrlat, 2),
                    linewidth = 0.1,labels=[1, 0, 0, 0], fmt="%.1f",dashes=[1, 6])
    m.drawmeridians(np.linspace( urcrnrlon,llcrnrlon,2), 
                    linewidth = 0.1, labels=[0, 0, 0, 1], fmt="%.1f",dashes=[1, 6])    
    longitude = np.linspace(llcrnrlon,urcrnrlon,ncols)
    latitude = np.linspace(urcrnrlat,llcrnrlat,nrows)
    # create grids and compute map projection coordinates for lon/lat grid
    x, y = m(*np.meshgrid(longitude, latitude))
    #======colormaps invert as surffix  _r;
    #======colormaps: brg_r, nipy_spectral_r, gist_rainbow, gist_ncar
#    cmap = mpl.cm.Spectral_r
#    cmap.set_bad('K')
    cmap = mpl.cm.jet
    #dataarray_mask = np.ma.masked_less(dataarry,-50)
    #mesh = m.pcolormesh(x, y, dataarry,cmap=cmap,vmin=-6,vmax=6)
    mesh = m.pcolormesh(x, y, dataarry,cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_title(figtitle,{'fontsize':16})
    #mesh
    cb = m.colorbar(mesh,ax=ax)
    #cb.set_label('Unit:day')
    m.readshapefile(shpfile,'shpfile',linewidth=1.0)
    #x,y=m(98.3,42.65)
    #plt.title("{0}\n {1} @bit2=1".format(figtitle, 'QC'))
    #plt.annotate(figtitle,xy=(x,y),size=10)
    plt.savefig(outfigname,dpi=300,bbox_inches='tight',transparent=False)
    plt.close()
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")

doys = [2018205]
diameters = [200]## set diameter pixels of circle [50,100,150,200]
pointList = [[100.09,38.36],[98.9,40.108333325],[100.857,42.081]]
pointList_up = [[100.09,38.36]]
sites = ['up','middle','down']
#pointList = [[100.857,42.081]]
#sites = ['down']
dem_path = 'E:/modis.lst.interp/dem/SRTM_1km.tif'
ndvi_path = 'E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/cal.ndvi/MOD13A2_2018209.1_km_16_days_NDVI_ndvi_qcwxz.tif'
ori_path = 'E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/Experiment1/ori/'
## shape file of HeiHe River Basin Boundary
##shpfile = os.path.abspath( '../../shape/heihe_huangwei')
data_ori1,cellSize1,crncords1 = RasterInfo(ndvi_path)
dsc = arcpy.Describe(ndvi_path)
sr= dsc.SpatialReference  #obtain the spatial reference of original lst image

#for doy in doys:
#    for experiment_point in pointList_up:
#        site = sites[pointList_up.index(experiment_point)]
#        out_dem_path = 'E:/modis.lst.interp/hrb.2018.C6/Experiment1/difference/dem/%s.%d.Day/'%(site,doy)
#        out_ndvi_path = 'E:/modis.lst.interp/hrb.2018.C6/Experiment1/difference/ndvi/%s.%d.Day/'%(site,doy)
#        for diameter in diameters:        
#            shape_file ='E:/modis.lst.interp/shape/Experiment.shape/%s_%d.shp'%(site,diameter)
#            #print(shape_file)
#            #### ---- DEM Clip and Plot
#            #print(out_path_clip)
#            out_dem_basename_nodata = "dem_%d_1km_%d_pixels_nodata.tif" %(doy,diameter)
#            out_dem_filename_nodata = os.path.join(out_dem_path,out_dem_basename_nodata) 
#            ## plot the difference Error map
#            outExtCircle = arcpy.sa.ExtractByCircle(dem_path,arcpy.Point(experiment_point[0],experiment_point[1]),diameter/2*cellSize1,'INSIDE')
#            arcpy.DefineProjection_management(outExtCircle,sr)  #set the spatial reference for outExtCircle
#            outExtCircle.save(out_dem_filename_nodata)                                   
#            errormean = arcpy.GetRasterProperties_management(out_dem_filename_nodata,'MEAN')
#            STD = arcpy.GetRasterProperties_management(out_dem_filename_nodata,'STD')
#            print 'ERROR Mean = %s, Standard Deviation = %s'%(errormean.getOutput(0),STD.getOutput(0)) 
#            
#            out_dem_path_clip = 'E:/modis.lst.interp/hrb.2018.C6/Experiment1/difference.clip/dem/%s.%d.Day/'%(site,doy)
#            out_dem_filename_nodata_clip = os.path.join(out_dem_path_clip,out_dem_basename_nodata) 
#            arcpy.Clip_management(out_dem_filename_nodata,'#',out_dem_filename_nodata_clip,shape_file,"-9999","NONE","MAINTAIN_EXTENT")                    
#            outfigname = "dem_%d_1km_%d_pixels_nodata.jpg" %(doy,diameter)
#            outfigname = os.path.join(out_dem_path_clip,outfigname)
#            figtitle = '%d D=%d'%(doy,diameter)
#            data,cellSize,crncords = RasterInfo(out_dem_filename_nodata_clip)
#            vmin =1400
#            vmax =5000
#            shape_file ='E:/modis.lst.interp/shape/Experiment.shape/%s_%d'%(site,diameter)
#            RasterPlot_wgs84_p1(data,crncords,shape_file,figtitle,outfigname,vmin,vmax)  
#            
#            #### ---- NDVI Clip and Plot
#            shape_file ='E:/modis.lst.interp/shape/Experiment.shape/%s_%d.shp'%(site,diameter)
#            out_ndvi_basename_nodata = "ndvi_%d_1km_%d_pixels_nodata.tif" %(doy,diameter)
#            out_ndvi_filename_nodata = os.path.join(out_ndvi_path,out_ndvi_basename_nodata) 
#            ## plot the difference Error map
#            outExtCircle = arcpy.sa.ExtractByCircle(ndvi_path,arcpy.Point(experiment_point[0],experiment_point[1]),diameter/2*cellSize1,'INSIDE')
#            arcpy.DefineProjection_management(outExtCircle,sr)  #set the spatial reference for outExtCircle
#            outExtCircle.save(out_ndvi_filename_nodata)                                   
#            errormean = arcpy.GetRasterProperties_management(out_ndvi_filename_nodata,'MEAN')
#            STD = arcpy.GetRasterProperties_management(out_ndvi_filename_nodata,'STD')
#            print 'ERROR Mean = %s, Standard Deviation = %s'%(errormean.getOutput(0),STD.getOutput(0)) 
#            
#            out_ndvi_path_clip = 'E:/modis.lst.interp/hrb.2018.C6/Experiment1/difference.clip/ndvi/%s.%d.Day/'%(site,doy)
#            #print(out_path_clip)
#            out_ndvi_basename_nodata = "ndvi_%d_1km_%d_pixels_nodata.tif" %(doy,diameter)
#            out_ndvi_filename_nodata_clip = os.path.join(out_ndvi_path_clip,out_ndvi_basename_nodata)  
#            arcpy.Clip_management(out_ndvi_filename_nodata,'#',out_ndvi_filename_nodata_clip,shape_file,"-9999","NONE","MAINTAIN_EXTENT")                    
#            outfigname = "ndvi_%d_1km_%d_pixels_nodata.jpg" %(doy,diameter)
#            outfigname = os.path.join(out_ndvi_path_clip,outfigname)
#            figtitle = '%d D=%d'%(doy,diameter)
#            data,cellSize,crncords = RasterInfo(out_ndvi_filename_nodata_clip)
#            vmin =0
#            vmax =1
#            shape_file ='E:/modis.lst.interp/shape/Experiment.shape/%s_%d'%(site,diameter)
#            RasterPlot_wgs84_p1(data,crncords,shape_file,figtitle,outfigname,vmin,vmax)    


def group_by_varible(File_path, mask):
    Data_raster = arcpy.Raster(File_path)
    Data_array = arcpy.RasterToNumPyArray(Data_raster)
    Data_array_mask = np.ma.masked_array(Data_array,mask=mask)
#    print np.ma.mean(LST_error_data_mask)
#    print np.ma.max(LST_error_data_mask)
#    print np.ma.min(LST_error_data_mask)
    
    return np.ma.mean(Data_array_mask),np.ma.std(Data_array_mask)
# # 正态分布的概率密度函数
# #   x      数据集中的某一具体测量值
# #   mu     数据集的平均值，反映测量值分布的集中趋势
# #   sigma  数据集的标准差，反映测量值分布的分散程度
def normfun(x, mu, sigma):
      pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
      return pdf
dem_path = 'E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/Experiment1/difference.clip/dem/up.2018205.Day/dem_2018205_1km_200_pixels_nodata.tif'
ndvi_path = 'E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/Experiment1/difference.clip/ndvi/up.2018205.Day/ndvi_2018205_1km_200_pixels_nodata.tif'
LST_error_Day_path = 'E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/Experiment1/difference.clip/up.2018205.Day/MOD11A1_2018205.LST_Day_1km_200_pixels_diff_nodata.tif'
font = {'family' : 'Palatino Linotype',
    'weight' : 'normal',
    'size'   : 18}    
plt.rc('font', **font)  
####===========================================================================
####------------------------------------
fig = plt.figure(figsize=[22,11],dpi=300)

####-------Plot the LST Error data histogram and Normal distribution---------------
ax = fig.add_subplot(231)
array = arcpy.RasterToNumPyArray(LST_error_Day_path,nodata_to_value = 9999)             
hist, bins = np.histogram(array, bins = np.arange(-8.5,9.5),density=True)
# Plot the histogram
plt.bar(bins[:-1]+0.5, hist, width = 0.5,fc='b')
plt.xticks(range(-8,9))
plt.xlim(-8.5, 8.5)
plt.ylabel('Frequency')
plt.xlabel('Reconstruct Error')
ax.text(-8,0.195,'(a)', fontsize=18, ha='left',va='bottom', wrap=True)
bins1 = np.arange(-8.5,8.5,0.1)
errormean = arcpy.GetRasterProperties_management(LST_error_Day_path,'MEAN').getOutput(0)
error_STD = arcpy.GetRasterProperties_management(LST_error_Day_path,'STD').getOutput(0)
pdf = normfun(bins1,float(errormean),float(error_STD))
plt.plot(bins1,pdf,color='b')
print 'hist', hist,
temp = {'bins':range(-8,9),'hist':hist}
df_histgram = pd.DataFrame(temp)
df_histgram.to_csv('E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/Experiment1/difference.clip/UP_LST_Error_Histogram_Statistic.Day.csv')

####---------------------------------------------------------------------------
dem_raster = arcpy.Raster(dem_path)
dem_data = arcpy.RasterToNumPyArray(dem_raster)
mask_1400to1800 = ~((dem_data >1400) & (dem_data<=1800))
mask_1800to2200 = ~((dem_data >1800) & (dem_data<=2200))
mask_2200to2600 = ~((dem_data >2200) & (dem_data<=2600))
mask_2600to3000 = ~((dem_data >2600) & (dem_data<=3000))
mask_3000to3400 = ~((dem_data >3000) & (dem_data<=3400))
mask_3400to3800 = ~((dem_data >3400) & (dem_data<=3800))
mask_3800to4200 = ~((dem_data >3800) & (dem_data<=4200))
mask_4200to4600 = ~((dem_data >4200) & (dem_data<=4600))
mask_4600to5000 = ~((dem_data >4600) & (dem_data<=6000))
availableData_list = [~mask_1400to1800,~mask_1800to2200,~mask_2200to2600,~mask_2600to3000,
                      ~mask_3000to3400,~mask_3400to3800,~mask_3800to4200,~mask_4200to4600,~mask_4600to5000]
dem_data_no = np.sum(dem_data>0);
#print(dem_data[availableData_list[0]].shape)

# mask_dict = dict(mask_1400to1800=mask_1400to1800,mask_1800to2200=mask_1800to2200,mask_2200to2600=mask_2200to2600,
#                  mask_2600to3000=mask_2600to3000,mask_3000to3400=mask_3000to3400,mask_3400to3800=mask_3400to3800,
#                  mask_3800to4200=mask_3800to4200,mask_4200to4600=mask_4200to4600,mask_4600to5000=mask_4600to5000,)
mask_dict = {'1.4-1.8':mask_1400to1800,'1.8-2.2': mask_1800to2200,'2.2-2.6':mask_2200to2600,
                 '2.6-3.0': mask_2600to3000,'3.0-3.4': mask_3000to3400,'3.4-3.8': mask_3400to3800,
                 '3.8-4.2': mask_3800to4200,'4.2-4.6': mask_4200to4600,'4.6-5.0': mask_4600to5000}
items = mask_dict.items()
items.sort() 

print 'DEM'

dem_range_list = []
lst_error_mean_list = []
lst_error_std_list = []
dem_mean_list = []
dem_std_list = []
ndvi_mean_list = []
ndvi_std_list = []
percentage_list = []
for (key,value) in items:
    dem_range_list.append(key)
    index = items.index((key,value))
    percentage_list.append(np.sum(availableData_list[index])/float(dem_data_no)*100)
    lst_error_mean,lst_error_std=group_by_varible(LST_error_Day_path,value)
    lst_error_mean_list.append(lst_error_mean)
    lst_error_std_list.append(lst_error_std)
    dem_mean,dem_std = group_by_varible(dem_path,value)
    dem_mean_list.append(dem_mean)
    dem_std_list.append(dem_std)
    ndvi_mean,ndvi_std = group_by_varible(ndvi_path,value)
    ndvi_mean_list.append(ndvi_mean)
    ndvi_std_list.append(ndvi_std)
   # print key,lst_error_mean,lst_error_std,dem_mean,dem_std,ndvi_mean,ndvi_std
temp = {'Range':dem_range_list,'lst_error_mean':lst_error_mean_list,
        'lst_error_std':lst_error_std_list,'dem_mean': dem_mean_list,'dem_std':dem_std_list,
        'ndvi_mean':ndvi_mean_list,'ndvi_std':ndvi_std_list,'Percentage':percentage_list}
df_dem = pd.DataFrame(temp)
df_dem.to_csv('E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/Experiment1/difference.clip/dem/dem_LST_Error_Statistic.Day.csv')
#print(df_dem)
                
ax1 = fig.add_subplot(232)
bar1 = plt.bar(range(1,10),df_dem['Percentage'],color='blue',width=0.5,label='Percentage')
plt.xticks(range(1,10),df_dem['Range'],fontsize =12)
plt.xlabel('DEM Range')
ax1.text(0.5,27,'(b)', fontsize=18, ha='left',va='bottom', wrap=True)
ax1.tick_params(axis='y',labelsize=18,colors='blue') # y轴
ax1.set_ylim([0, 30])
#第二纵轴的设置和绘图
ax2=ax1.twinx()
line1, = plt.plot(range(1,10),df_dem['lst_error_mean'],color='black',marker='o',label='LST Error')
plt.axhline(y=1,ls=':',c='black')
plt.axhline(y=-1,ls=':',c='black')

ax1.yaxis.set_label_position("right")
#ax1.set_ylabel('Percentage(%)',color='blue')
ax1.yaxis.tick_right()
ax2.yaxis.set_label_position("left")
ax2.yaxis.tick_left()
ax2.set_ylabel('LST Error(K)')
ax2.set_yticks(range(-4,3))
ax2.set_ylim([-4, 3])
plt.legend((bar1,line1),('Percentage','LST Error'),loc='upper right',fontsize=12)

#### ==========================================================================
####---------------------------------------------------------------------------   
ndvi_raster = arcpy.Raster(ndvi_path)
ndvi_data = arcpy.RasterToNumPyArray(ndvi_raster)
mask_00to01 = ~((ndvi_data >=0.0) & (ndvi_data<=0.1))
mask_01to02 = ~((ndvi_data >0.1) & (ndvi_data<=0.2))
mask_02to03 = ~((ndvi_data >0.2) & (ndvi_data<=0.3))
mask_03to04 = ~((ndvi_data >0.3) & (ndvi_data<=0.4))
mask_04to05 = ~((ndvi_data >0.4) & (ndvi_data<=0.5))
mask_05to06 = ~((ndvi_data >0.5) & (ndvi_data<=0.6))
mask_06to07 = ~((ndvi_data >0.6) & (ndvi_data<=0.7))
mask_07to08 = ~((ndvi_data >0.7) & (ndvi_data<=0.8))
mask_08to09 = ~((ndvi_data >0.8) & (ndvi_data<=1.0))
mask_dict = {'0.0-0.1': mask_00to01,'0.1-0.2': mask_01to02,'0.2-0.3': mask_02to03,
                 '0.3-0.4': mask_03to04,'0.4-0.5': mask_04to05,'0.5-0.6': mask_05to06,
                 '0.6-0.7': mask_06to07,'0.7-0.8':mask_07to08,'0.8-0.9':mask_08to09}
availableData_list = [~mask_00to01,~mask_01to02,~mask_02to03,~mask_03to04,
                      ~mask_04to05,~mask_05to06,~mask_06to07,~mask_07to08,~mask_08to09]
ndvi_data_no = np.sum(ndvi_data>=0);

items = mask_dict.items()
items.sort() 
print 'NDVI' 
ndvi_range_list = []
lst_error_mean_list = []
lst_error_std_list = []
dem_mean_list = []
dem_std_list = []
ndvi_mean_list = []
ndvi_std_list = []
percentage_list = []
for (key,value) in items:
    ndvi_range_list.append(key)
    index = items.index((key,value))
    percentage_list.append(np.sum(availableData_list[index])/float(ndvi_data_no)*100)
    lst_error_mean,lst_error_std=group_by_varible(LST_error_Day_path,value)
    lst_error_mean_list.append(lst_error_mean)
    lst_error_std_list.append(lst_error_std)
    dem_mean,dem_std = group_by_varible(dem_path,value)
    dem_mean_list.append(dem_mean)
    dem_std_list.append(dem_std)
    ndvi_mean,ndvi_std = group_by_varible(ndvi_path,value)
    ndvi_mean_list.append(ndvi_mean)
    ndvi_std_list.append(ndvi_std)
    #print key,lst_error_mean,lst_error_std,dem_mean,dem_std,ndvi_mean,ndvi_std
temp = {'Range':ndvi_range_list,'lst_error_mean':lst_error_mean_list,
        'lst_error_std':lst_error_std_list,'dem_mean': dem_mean_list,'dem_std':dem_std_list,
        'ndvi_mean':ndvi_mean_list,'ndvi_std':ndvi_std_list,'Percentage':percentage_list}
df_ndvi = pd.DataFrame(temp)
df_ndvi.to_csv('E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/Experiment1/difference.clip/ndvi/ndvi_LST_Error_Statistic.Day.csv')
#print(df_ndvi)
####------------------------------------
ax1 = fig.add_subplot(233)
bar1 = plt.bar(range(1,10),df_ndvi['Percentage'],color='blue',width=0.5,label='Percentage')
plt.xticks(range(1,10),df_ndvi['Range'],fontsize =12)
plt.xlabel('NDVI Range')
ax1.tick_params(axis='y',labelsize=18,colors='blue') # y轴
ax1.set_ylim([0, 30])
ax1.text(0.5,27,'(c)', fontsize=18, ha='left',va='bottom', wrap=True)
#第二纵轴的设置和绘图
ax2=ax1.twinx()
line1, = plt.plot(range(1,10),df_ndvi['lst_error_mean'],color='black',marker='o',label='LST Error')
plt.axhline(y=1,ls=':',c='black')
plt.axhline(y=-1,ls=':',c='black')

ax1.yaxis.set_label_position("right")
ax1.set_ylabel('Percentage(%)',color='blue')
ax1.yaxis.tick_right()
ax2.yaxis.set_label_position("left")
ax2.yaxis.tick_left()
#ax2.set_ylabel('LST Error(K)')
ax2.set_yticks(range(-4,3))
ax2.set_ylim([-4, 3])
plt.legend((bar1,line1),('Percentage','LST Error'),loc='upper right',fontsize=12)

####===========================================================================
####=====================     Night    ========================================
LST_error_Night_path = 'E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/Experiment1/difference.clip/up.2018205.Night/MOD11A1_2018205.LST_Night_1km_200_pixels_diff_nodata.tif'
font = {'family' : 'Palatino Linotype',
    'weight' : 'normal',
    'size'   : 18}    
plt.rc('font', **font)  
####-------Plot the LST Error data histogram and Normal distribution---------------
ax = fig.add_subplot(234)
array = arcpy.RasterToNumPyArray(LST_error_Night_path,nodata_to_value = 9999)             
hist, bins = np.histogram(array, bins = np.arange(-8.5,9.5),density=True)
# Plot the histogram
plt.bar(bins[:-1]+0.5, hist, width = 0.5,fc='b')
plt.xticks(range(-8,9))
plt.xlim(-8.5, 8.5)
plt.ylabel('Frequency')
plt.xlabel('Reconstruct Error')
ax.text(-8,0.30,'(d)', fontsize=18, ha='left',va='bottom', wrap=True)
bins = np.arange(-8.5,8.5,0.1)
errormean = arcpy.GetRasterProperties_management(LST_error_Day_path,'MEAN').getOutput(0)
error_STD = arcpy.GetRasterProperties_management(LST_error_Day_path,'STD').getOutput(0)
pdf = normfun(bins,float(errormean),float(error_STD))
plt.plot(bins,pdf,color='b')
print 'hist', hist,
temp = {'bins':range(-8,9),'hist':hist}
df_histgram = pd.DataFrame(temp)
df_histgram.to_csv('E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/Experiment1/difference.clip/UP_LST_Error_Histogram_Statistic.Night.csv')

####---------------------------------------------------------------------------
dem_raster = arcpy.Raster(dem_path)
dem_data = arcpy.RasterToNumPyArray(dem_raster)
mask_1400to1800 = ~((dem_data >1400) & (dem_data<=1800))
mask_1800to2200 = ~((dem_data >1800) & (dem_data<=2200))
mask_2200to2600 = ~((dem_data >2200) & (dem_data<=2600))
mask_2600to3000 = ~((dem_data >2600) & (dem_data<=3000))
mask_3000to3400 = ~((dem_data >3000) & (dem_data<=3400))
mask_3400to3800 = ~((dem_data >3400) & (dem_data<=3800))
mask_3800to4200 = ~((dem_data >3800) & (dem_data<=4200))
mask_4200to4600 = ~((dem_data >4200) & (dem_data<=4600))
mask_4600to5000 = ~((dem_data >4600) & (dem_data<=6000))
availableData_list = [~mask_1400to1800,~mask_1800to2200,~mask_2200to2600,~mask_2600to3000,
                      ~mask_3000to3400,~mask_3400to3800,~mask_3800to4200,~mask_4200to4600,~mask_4600to5000]
dem_data_no = np.sum(dem_data>0);
#print(dem_data[availableData_list[0]].shape)

# mask_dict = dict(mask_1400to1800=mask_1400to1800,mask_1800to2200=mask_1800to2200,mask_2200to2600=mask_2200to2600,
#                  mask_2600to3000=mask_2600to3000,mask_3000to3400=mask_3000to3400,mask_3400to3800=mask_3400to3800,
#                  mask_3800to4200=mask_3800to4200,mask_4200to4600=mask_4200to4600,mask_4600to5000=mask_4600to5000,)
mask_dict = {'1.4-1.8':mask_1400to1800,'1.8-2.2': mask_1800to2200,'2.2-2.6':mask_2200to2600,
                 '2.6-3.0': mask_2600to3000,'3.0-3.4': mask_3000to3400,'3.4-3.8': mask_3400to3800,
                 '3.8-4.2': mask_3800to4200,'4.2-4.6': mask_4200to4600,'4.6-5.0': mask_4600to5000}
items = mask_dict.items()
items.sort() 

print 'DEM'

dem_range_list = []
lst_error_mean_list = []
lst_error_std_list = []
dem_mean_list = []
dem_std_list = []
ndvi_mean_list = []
ndvi_std_list = []
percentage_list = []
for (key,value) in items:
    dem_range_list.append(key)
    index = items.index((key,value))
    percentage_list.append(np.sum(availableData_list[index])/float(dem_data_no)*100)
    lst_error_mean,lst_error_std=group_by_varible(LST_error_Night_path,value)
    lst_error_mean_list.append(lst_error_mean)
    lst_error_std_list.append(lst_error_std)
    dem_mean,dem_std = group_by_varible(dem_path,value)
    dem_mean_list.append(dem_mean)
    dem_std_list.append(dem_std)
    ndvi_mean,ndvi_std = group_by_varible(ndvi_path,value)
    ndvi_mean_list.append(ndvi_mean)
    ndvi_std_list.append(ndvi_std)
    #print key,lst_error_mean,lst_error_std,dem_mean,dem_std,ndvi_mean,ndvi_std
temp = {'Range':dem_range_list,'lst_error_mean':lst_error_mean_list,
        'lst_error_std':lst_error_std_list,'dem_mean': dem_mean_list,'dem_std':dem_std_list,
        'ndvi_mean':ndvi_mean_list,'ndvi_std':ndvi_std_list,'Percentage':percentage_list}
df_dem = pd.DataFrame(temp)
df_dem.to_csv('E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/Experiment1/difference.clip/dem/dem_LST_Error_Statistic.Night.csv')
#print(df_dem)
####------------------------------------
ax1 = fig.add_subplot(235)
bar1 = plt.bar(range(1,10),df_dem['Percentage'],color='blue',width=0.5,label='Percentage')
plt.xticks(range(1,10),df_dem['Range'],fontsize =12)
plt.xlabel('DEM Range')
ax1.tick_params(axis='y',labelsize=18,colors='blue') # y轴
ax1.set_ylim([0, 30])
ax1.text(0.5,27,'(e)', fontsize=18, ha='left',va='bottom', wrap=True)
#第二纵轴的设置和绘图
ax2=ax1.twinx()
line1, = plt.plot(range(1,10),df_dem['lst_error_mean'],color='black',marker='o',label='LST Error')
plt.axhline(y=1,ls=':',c='black')
plt.axhline(y=-1,ls=':',c='black')

ax1.yaxis.set_label_position("right")
#ax1.set_ylabel('Percentage(%)',color='blue')
ax1.yaxis.tick_right()
ax2.yaxis.set_label_position("left")
ax2.yaxis.tick_left()
ax2.set_ylabel('LST Error(K)')
ax2.set_yticks(range(-4,3))
ax2.set_ylim([-4, 3])
plt.legend((bar1,line1),('Percentage','LST Error'),loc='upper right',fontsize=12)

#### ==========================================================================
####---------------------------------------------------------------------------   
ndvi_raster = arcpy.Raster(ndvi_path)
ndvi_data = arcpy.RasterToNumPyArray(ndvi_raster)
mask_00to01 = ~((ndvi_data >=0.0) & (ndvi_data<=0.1))
mask_01to02 = ~((ndvi_data >0.1) & (ndvi_data<=0.2))
mask_02to03 = ~((ndvi_data >0.2) & (ndvi_data<=0.3))
mask_03to04 = ~((ndvi_data >0.3) & (ndvi_data<=0.4))
mask_04to05 = ~((ndvi_data >0.4) & (ndvi_data<=0.5))
mask_05to06 = ~((ndvi_data >0.5) & (ndvi_data<=0.6))
mask_06to07 = ~((ndvi_data >0.6) & (ndvi_data<=0.7))
mask_07to08 = ~((ndvi_data >0.7) & (ndvi_data<=0.8))
mask_08to09 = ~((ndvi_data >0.8) & (ndvi_data<=1.0))
mask_dict = {'0.0-0.1': mask_00to01,'0.1-0.2': mask_01to02,'0.2-0.3': mask_02to03,
                 '0.3-0.4': mask_03to04,'0.4-0.5': mask_04to05,'0.5-0.6': mask_05to06,
                 '0.6-0.7': mask_06to07,'0.7-0.8':mask_07to08,'0.8-0.9':mask_08to09}
availableData_list = [~mask_00to01,~mask_01to02,~mask_02to03,~mask_03to04,
                      ~mask_04to05,~mask_05to06,~mask_06to07,~mask_07to08,~mask_08to09]
ndvi_data_no = np.sum(ndvi_data>=0);

items = mask_dict.items()
items.sort() 
print 'NDVI' 
ndvi_range_list = []
lst_error_mean_list = []
lst_error_std_list = []
dem_mean_list = []
dem_std_list = []
ndvi_mean_list = []
ndvi_std_list = []
percentage_list = []
for (key,value) in items:
    ndvi_range_list.append(key)
    index = items.index((key,value))
    percentage_list.append(np.sum(availableData_list[index])/float(ndvi_data_no)*100)
    lst_error_mean,lst_error_std=group_by_varible(LST_error_Night_path,value)
    lst_error_mean_list.append(lst_error_mean)
    lst_error_std_list.append(lst_error_std)
    dem_mean,dem_std = group_by_varible(dem_path,value)
    dem_mean_list.append(dem_mean)
    dem_std_list.append(dem_std)
    ndvi_mean,ndvi_std = group_by_varible(ndvi_path,value)
    ndvi_mean_list.append(ndvi_mean)
    ndvi_std_list.append(ndvi_std)
    #print key,lst_error_mean,lst_error_std,dem_mean,dem_std,ndvi_mean,ndvi_std
temp = {'Range':ndvi_range_list,'lst_error_mean':lst_error_mean_list,
        'lst_error_std':lst_error_std_list,'dem_mean': dem_mean_list,'dem_std':dem_std_list,
        'ndvi_mean':ndvi_mean_list,'ndvi_std':ndvi_std_list,'Percentage':percentage_list}
df_ndvi = pd.DataFrame(temp)
df_ndvi.to_csv('E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/Experiment1/difference.clip/ndvi/ndvi_LST_Error_Statistic.Day.csv')
#print(df_ndvi)
####------------------------------------
ax1 = fig.add_subplot(236)
bar1 = plt.bar(range(1,10),df_ndvi['Percentage'],color='blue',width=0.5,label='Percentage')
plt.xticks(range(1,10),df_ndvi['Range'],fontsize =12)
plt.xlabel('NDVI Range')
ax1.tick_params(axis='y',labelsize=18,colors='blue') # y轴
ax1.set_ylim([0, 30])
ax1.text(0.5,27,'(f)', fontsize=18, ha='left',va='bottom', wrap=True)
#第二纵轴的设置和绘图
ax2=ax1.twinx()
line1, = plt.plot(range(1,10),df_ndvi['lst_error_mean'],color='black',marker='o',label='LST Error')
plt.axhline(y=1,ls=':',c='black')
plt.axhline(y=-1,ls=':',c='black')

ax1.yaxis.set_label_position("right")
ax1.set_ylabel('Percentage(%)',color='blue')
ax1.yaxis.tick_right()
ax2.yaxis.set_label_position("left")
ax2.yaxis.tick_left()
#ax2.set_ylabel('LST Error(K)')
ax2.set_yticks(range(-4,3))
ax2.set_ylim([-4, 3])
plt.legend((bar1,line1),('Percentage','LST Error'),loc='upper right',fontsize=12)

plt.savefig('E:/mywork/Paper/LST_Reconstruct_RS/output_figures/Figure 13. distribution histogram of the mean bias error Dem.ndvi.Day & Night.tif',dpi=300,bbox_inches = 'tight')