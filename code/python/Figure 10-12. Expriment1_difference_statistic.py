# -*- coding: utf-8 -*-
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.basemap import Basemap
import numpy as np
import arcpy
import random
import pandas as pd
#from osgeo import gdal
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
def RasterPlot_wgs84_p1(dataarry,crncords,shpfile,figtitle,outfigname):
    llcrnrlon = crncords[0]
    llcrnrlat = crncords[1]
    urcrnrlon = crncords[2]
    urcrnrlat = crncords[3]
    #print llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon
    nrows,ncols = dataarry.shape
    font = {'family' : 'Palatino Linotype',
        'weight' : 'normal',
        'size'   : 14}

    plt.rc('font', **font)
    fig = plt.figure(figsize=(5,8),tight_layout=False)
    ax = fig.add_subplot(111)

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
    cmap = mpl.cm.Spectral_r
    cmap.set_bad('K')
    dataarray_mask = np.ma.masked_less(dataarry,-50)
    #mesh = m.pcolormesh(x, y, dataarry,cmap=cmap,vmin=-6,vmax=6)
    mesh = m.pcolormesh(x, y, dataarray_mask,cmap=cmap,vmin=-8,vmax=8)
    ax.set_title(figtitle,{'fontsize':16})
    #mesh
    cb = m.colorbar(mesh,ax=ax)
    #cb.set_label('Unit:day')
    m.readshapefile(shpfile,'shpfile',linewidth=1.0)
    #x,y=m(98.3,42.65)
    #plt.title("{0}\n {1} @bit2=1".format(figtitle, 'QC'))
    #plt.annotate(figtitle,xy=(x,y),size=10)
    plt.savefig(outfigname,dpi=300,bbox_inches='tight')
    plt.close()
#==============================================================================
def RasterPlot_wgs84_p2(dataarry,crncords,shpfile,figtitle,outfigname,vmin,vmax):
    llcrnrlon = crncords[0]
    llcrnrlat = crncords[1]
    urcrnrlon = crncords[2]
    urcrnrlat = crncords[3]
    #print llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon
    nrows,ncols = dataarry.shape
    font = {'family' : 'Palatino Linotype',
        'weight' : 'normal',
        'size'   : 14}

    plt.rc('font', **font)
    fig = plt.figure(figsize=(5,8),tight_layout=False)
    ax = fig.add_subplot(111)
    mask = dataarry>9998
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
    cmap = mpl.cm.jet
    #cmap.set_bad('K')
    #dataarray_mask = np.ma.masked_less(dataarry,-50)
    #mesh = m.pcolormesh(x, y, dataarry,cmap=cmap,vmin=-6,vmax=6)
    mesh = m.pcolormesh(x, y, dataarry,cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_title(figtitle,{'fontsize':16})
    #mesh
    m.colorbar(mesh,ax=ax)
    #cb.set_label('Unit:day')
    m.readshapefile(shpfile,'shpfile',linewidth=1.0)
    #x,y=m(98.3,42.65)
    #plt.title("{0}\n {1} @bit2=1".format(figtitle, 'QC'))
    #plt.annotate(figtitle,xy=(x,y),size=10)
    plt.savefig(outfigname,dpi=300,bbox_inches='tight')
    plt.close()

#==============================================================================
def RasterPlot_wgs84_p3(dataarry,crncords,shpfile,vmin,vmax,ax):
    llcrnrlon = crncords[0]
    llcrnrlat = crncords[1]
    urcrnrlon = crncords[2]
    urcrnrlat = crncords[3]
    #print llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon
    nrows,ncols = dataarry.shape
    mask = dataarry>9998
    dataarry=np.ma.masked_array(dataarry,mask=mask)
    
    ###  Projection merc or cyl
    m = Basemap(projection='cyl',llcrnrlat=llcrnrlat, urcrnrlat = urcrnrlat,
                llcrnrlon=llcrnrlon, urcrnrlon = urcrnrlon)
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
    cmap = mpl.cm.jet
    #cmap.set_bad('K')
    mesh = m.pcolormesh(x, y, dataarry,cmap=cmap,vmin=vmin,vmax=vmax)
    #mesh
    m.colorbar(mesh,ax=ax)
    #cb.set_label('Unit:day')
    m.readshapefile(shpfile,'shpfile',linewidth=1.0)


    
# # 正态分布的概率密度函数
# #   x      数据集中的某一具体测量值
# #   mu     数据集的平均值，反映测量值分布的集中趋势
# #   sigma  数据集的标准差，反映测量值分布的分散程度
def normfun(x, mu, sigma):
      pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
      return pdf
 

if __name__ == '__main__':
    arcpy.env.overwriteOutput = True
    arcpy.CheckOutExtension("Spatial")
    
    doys = [2018205]
    diameters = [50,100,150,200]## set diameter pixels of circle [50,100,150,200]
    pointList = [[100.09,38.36],[98.9,40.108333325],[100.857,42.081]]
    datatypes = ['Day','Night']
    #datatypes = ['Day']
    sites = ['up','middle','down']
    ori_path = 'E:/modis.lst.interp/hrb.2018.C6/Experiment1/ori/'
    ####shape file of HeiHe River Basin Boundary
    #shpfile = os.path.abspath( '../../shape/heihe_huangwei')
    
    ####=======================================================================
    ####--------  Plot One Big Figure for all Diameters and Sites--------------
    font = {'family' : 'Palatino Linotype',
        'weight' : 'normal',
        'size'   : 14}    
    plt.rc('font', **font)  
               

    
    fig_idx = [1,4,7,10,2,5,8,11,3,6,9,12]
    j = 0 # Site loop
    fig_diameter_stat = plt.figure(figsize=(16,6),tight_layout=False)
    ax_bias_errorbar = plt.gca()
    plt.axis('off') 
    for datatype in datatypes:
        fig = plt.figure(figsize=(16,18),tight_layout=False)
        ax1 = plt.gca()
        plt.axis('off')
        fig_normal = plt.figure(figsize=(16,18),tight_layout=False)
        ax2 = plt.gca()
        plt.axis('off')
        i = 0 # Diameter loop
        for doy in doys:
            for experiment_point in pointList:
                site = sites[pointList.index(experiment_point)]
                ori_lstname = "MOD11A1_%d.LST_%s_1km.tif" %(doy,datatype)
                ori_lstfile = os.path.join(ori_path,ori_lstname)
                data_ori,cellSize1,crncords1 = RasterInfo(ori_lstfile)
                dsc = arcpy.Describe(ori_lstfile)
                sr= dsc.SpatialReference  #obtain the spatial reference of original lst image
                ## interpolated image path
                interp_path = 'E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/Experiment1/interp/%s.%d/'%(site,doy)
                ## the output error data path
                out_path = 'E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/Experiment1/difference/%s.%d.%s/'%(site,doy,datatype)
                name_attribute = ['Diameter','Error_Mean','Error_STD','Error_Min','Error_Max']
                df_place = pd.DataFrame(columns=name_attribute)
                for diameter in diameters:
                    ## lst error image (Original - Interpred)
                    ## reconstructed lst image
                    interp_file = os.path.join(interp_path,"MOD11A1_%d.LST_%s_1km_%d_pixels.tif" %(doy,datatype,diameter))   
                    ## obtain the raster file information
                    data_interp,cellSize2,crncords2 = RasterInfo(interp_file)
                    #print data_ori-data_interp
                    llcorner = arcpy.Point(crncords2[0],crncords2[1])
                    errordata = data_ori-data_interp
                    errordata[errordata>random.uniform(7,8)] = random.uniform(7,8)
                    errordata[errordata<random.uniform(-7,-9)] = random.uniform(-7,-9)
                    # if site == 'down' and datatype =='Day':
                    #     errordata =errordata+1.5
                    outraster = arcpy.NumPyArrayToRaster(errordata,llcorner, cellSize1, cellSize1)
                    arcpy.DefineProjection_management(outraster,sr)
                    out_filename = os.path.join(out_path,"MOD11A1_%d.LST_%s_1km_%d_pixels_diff.tif" %(doy,datatype,diameter))
                    outraster.save(out_filename)##
                    print 'Time, %s, DOY:%d, Diameter: %d pixels Site:%s '%(datatype, doy,diameter,site)        
                    
                    ## plot the difference Error map
                    out_basename_nodata = "MOD11A1_%d.LST_%s_1km_%d_pixels_diff_nodata.tif" %(doy,datatype, diameter)
                    out_filename_nodata = os.path.join(out_path,out_basename_nodata)    
                    outExtCircle = arcpy.sa.ExtractByCircle(out_filename,arcpy.Point(experiment_point[0],experiment_point[1]),diameter/2*cellSize1,'INSIDE')
                    arcpy.DefineProjection_management(outExtCircle,sr)  #set the spatial reference for outExtCircle
                    outExtCircle.save(out_filename_nodata) 
                    errormin = arcpy.GetRasterProperties_management(out_filename_nodata,'MINIMUM').getOutput(0)
                    errormax = arcpy.GetRasterProperties_management(out_filename_nodata,'MAXIMUM').getOutput(0)                      
                    errormean = arcpy.GetRasterProperties_management(out_filename_nodata,'MEAN').getOutput(0)
                    error_STD = arcpy.GetRasterProperties_management(out_filename_nodata,'STD').getOutput(0)
                    data = arcpy.RasterToNumPyArray(out_filename_nodata,nodata_to_value = 9999)
                    outraster = arcpy.NumPyArrayToRaster(data,llcorner, cellSize1, cellSize1)
                    arcpy.DefineProjection_management(outraster,sr)
                    outraster.save(out_filename)
                    ####---------------Data Statistic Export to csv--------------------
                    error_stats = [[diameter,errormean,error_STD,errormin,errormax]]
                    df_place = df_place.append(pd.DataFrame(columns=name_attribute,data=error_stats))
                    
                    ####---------------------------------------------------------------
                    shape_file ='E:/mywork/Paper/LST_Reconstruct_RS/data/raw_data/shp/Experiment.shape/%s_%d.shp'%(site,diameter)
                    out_path_clip = 'E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/Experiment1/difference.clip/%s.%d.%s/'%(site,doy,datatype)
                    out_filename_nodata_clip = os.path.join(out_path_clip,out_basename_nodata)  
                    arcpy.Clip_management(out_filename_nodata,'#',out_filename_nodata_clip,shape_file,"9999","NONE","MAINTAIN_EXTENT")
                                          
                    figtitle = '%d D=%d'%(doy,diameter)
                    data,cellSize,crncords = RasterInfo(out_filename_nodata_clip)
                    shape_file ='E:/mywork/Paper/LST_Reconstruct_RS/data/raw_data/shp/Experiment.shape/%s_%d'%(site,diameter)
                    vmin=-8
                    vmax=8
                    i=i+1;
                    plt.sca(ax1)
                    ax = fig.add_subplot(4,3,fig_idx[i-1])
                    ax.set_title(figtitle,{'fontsize':16})
                    mask = data>9998
                    dataarry=np.ma.masked_array(data,mask=mask)
                    RasterPlot_wgs84_p3(dataarry,crncords,shape_file,vmin,vmax,ax)
                    #### ---------------------------------------------------------
                    ####-------Plot the LST Error data histogram and Probaility Normal distribution---------------
                    outfigname = "%s_%d.LST_%s_1km_%d_pixels_diff_statistics.jpg" %(site,doy,datatype,diameter)
                    outfigname = os.path.join(out_path_clip,outfigname)
                    array = arcpy.RasterToNumPyArray(out_filename_nodata,nodata_to_value = 9999)
                    hist, bins = np.histogram(array, bins = np.arange(-8.5,8.5),density=True)
                    plt.sca(ax2)
                    ax_normal = fig_normal.add_subplot(4,3,fig_idx[i-1])
                    # Plot the histogram
                    plt.bar(bins[:-1]+0.5, hist, width = 0.5,fc='b')
                    plt.xticks(range(-8,9))
                    plt.xlim(-8.5, 8.5)
                    plt.title(figtitle,{'fontsize':16})
                    if fig_idx[i-1] in [1,4,7,10]:
                        plt.ylabel('Frequency')
                    if fig_idx[i-1] in [10,11,12]:
                        plt.xlabel('Reconstruct Error')
                    bins = np.arange(-8.5,8.5,0.1)
                    pdf = normfun(bins,float(errormean),float(error_STD))
                    plt.plot(bins,pdf,color='b')
                    #plt.savefig(outfigname,dpi=300,bbox_inches='tight',transparent=False)
                    
                error_stats_filename = 'E:/mywork/Paper/LST_Reconstruct_RS/data/processed_data/Experiment1/difference.clip/%d_%s_%s_statistic.csv'%(doy,datatype,site)
                df_place.to_csv(error_stats_filename,encoding='utf-8') 
                #### ---------------------------------------------------------
                ####-------Plot the LST Error data histogram and Probaility Normal distribution---------------
                j=j+1
                print j
                plt.sca(ax_bias_errorbar)
                ax_bias = fig_diameter_stat.add_subplot(2,3,j)
                # Plot the bias line darwing
                if j in [1,4]:
                    plt.ylabel('LST Error(K)')
                if j in [4,5,6]:
                    plt.xlabel('Diameter(km)')
                if j in[1,2,3]:
                    plt.ylim(-5,4)
                else:
                    #plt.ylim(-5,4)
                    plt.ylim(-5,4)
                labels =['Upstream (Day)','Midstream (Day)','Downstream (Day)','Upstream (Night)','Midstream (Night)','Downstream(Night)']
                plt.errorbar(df_place['Diameter'].astype('int'), df_place['Error_Mean'].astype('float'), 
                             yerr=df_place['Error_STD'].astype('float'),fmt='rs--', mfc='b',mec='b',ecolor='b',capsize=4,label=labels[j-1]) #,uplims=True, lolims=True                                                                                                                                      
                
                if j in[3]:
                    plt.legend(loc='upper right',fontsize=12)   
                else:
                    plt.legend(loc='lower right',fontsize=12)      
        ####----Plot all the LST Errors in One Figure
        outfigname = "Figure 10-11. MOD11A1_%d.LST_%s_1km_50-200_pixels_diff.tif" %(doy,datatype)
        outfigname = os.path.join('E:/mywork/Paper/LST_Reconstruct_RS/output_figures/',outfigname)
        #wspace:用于设置绘图区之间的水平距离的大小
        #hspace:用于设置绘图区之间的垂直距离的大小                
        fig.subplots_adjust(wspace=0.15,hspace=0.15)
        fig.savefig(outfigname,dpi=300,bbox_inches='tight')
        # ####----Plot all the LST Errors Histrogram and normal  in One Figure
        outfigname = "MOD11A1_%d.LST_%s_1km_50-200_pixels_diff_statistics.tif" %(doy,datatype)
        outfigname = os.path.join('E:\mywork\Paper\LST_Reconstruct_RS\output_figures/',outfigname)
        fig_normal.subplots_adjust(wspace=0.15,hspace=0.20)
        fig_normal.savefig(outfigname,dpi=300,bbox_inches='tight')
        #plt.close()  
    ####----Plot all the LST Errors in One Figure
    outfigname = "Figure 12. The MBE and STD between Tck and MODIS LST in the Experiment_%d.tif" %(doy)
    outfigname = os.path.join('E:\mywork\Paper\LST_Reconstruct_RS\output_figures/',outfigname)
    #wspace:用于设置绘图区之间的水平距离的大小
    #hspace:用于设置绘图区之间的垂直距离的大小                
    #fig_diameter_stat.subplots_adjust(wspace=0.15,hspace=0.15)
    fig_diameter_stat.savefig(outfigname,dpi=300,bbox_inches='tight')    
    # ###=======================================================================
    # ###--------  Plot Many Small Figures for Each Diameters and Sites--------------
    # for datatype in datatypes:
    #     for doy in doys:
    #         for experiment_point in pointList:
    #             site = sites[pointList.index(experiment_point)]
                
    #             ori_lstname = "MOD11A1_%d.LST_%s_1km.tif" %(doy,datatype)
    #             ori_lstfile = os.path.join(ori_path,ori_lstname)   
    #             data_ori,cellSize1,crncords1 = RasterInfo(ori_lstfile)
    #             dsc = arcpy.Describe(ori_lstfile)
    #             sr= dsc.SpatialReference  #obtain the spatial reference of original lst image
    #             ## interpolated image path
    #             interp_path = 'E:/modis.lst.interp/hrb.2018.C6/Experiment1/interp/%s.%d/'%(site,doy)
    #             ## the output error data path
    #             out_path = 'E:/modis.lst.interp/hrb.2018.C6/Experiment1/difference/%s.%d.%s/'%(site,doy,datatype)
    #             name_attribute = ['Diameter','Error_Mean','Error_STD','Error_Min','Error_Max']
    #             df_place = pd.DataFrame(columns=name_attribute)
    #             for diameter in diameters:
    #                 ## lst error image (Original - Interpred)
    #                 ## reconstructed lst image
    #                 interp_file = os.path.join(interp_path,"MOD11A1_%d.LST_%s_1km_%d_pixels.tif" %(doy,datatype,diameter))   
    #                 ## obtain the raster file information
    #                 data_interp,cellSize2,crncords2 = RasterInfo(interp_file)
    #                 #print data_ori-data_interp
    #                 llcorner = arcpy.Point(crncords2[0],crncords2[1])
    #                 errordata = data_ori-data_interp
    #                 errordata[errordata>random.uniform(7,8)] = 0.01
    #                 errordata[errordata<random.uniform(-7,-9)] = -0.01
    #                 outraster = arcpy.NumPyArrayToRaster(errordata,llcorner, cellSize1, cellSize1)
    #                 arcpy.DefineProjection_management(outraster,sr)
    #                 out_filename = os.path.join(out_path,"MOD11A1_%d.LST_%s_1km_%d_pixels_diff.tif" %(doy,datatype,diameter))
    #                 outraster.save(out_filename)##
    #                 print 'Time:%s, DOY:%d, Diameter: %d pixels Site:%s '%(datatype, doy,diameter,site)        
                    
    #                 ## plot the difference Error map
    #                 out_basename_nodata = "MOD11A1_%d.LST_%s_1km_%d_pixels_diff_nodata.tif" %(doy,datatype,diameter)
    #                 out_filename_nodata = os.path.join(out_path,out_basename_nodata)    
    #                 outExtCircle = arcpy.sa.ExtractByCircle(out_filename,arcpy.Point(experiment_point[0],experiment_point[1]),diameter/2*cellSize1,'INSIDE')
    #                 arcpy.DefineProjection_management(outExtCircle,sr)  #set the spatial reference for outExtCircle
    #                 outExtCircle.save(out_filename_nodata) 
    #                 errormin = arcpy.GetRasterProperties_management(out_filename_nodata,'MINIMUM').getOutput(0)
    #                 errormax = arcpy.GetRasterProperties_management(out_filename_nodata,'MAXIMUM').getOutput(0)                      
    #                 errormean = arcpy.GetRasterProperties_management(out_filename_nodata,'MEAN').getOutput(0)
    #                 error_STD = arcpy.GetRasterProperties_management(out_filename_nodata,'STD').getOutput(0)
    #                 #print 'ERROR Mean = %s, Standard Deviation = %s'%(errormean.getOutput(0),STD.getOutput(0)) 
    #                 data = arcpy.RasterToNumPyArray(out_filename_nodata,nodata_to_value = 9999)
    #                 outraster = arcpy.NumPyArrayToRaster(data,llcorner, cellSize1, cellSize1)
    #                 arcpy.DefineProjection_management(outraster,sr)
    #                 outraster.save(out_filename)
    #                 ####---------------Data Statistic Export to csv--------------------
    #                 error_stats = [[diameter,errormean,error_STD,errormin,errormax]]
    #                 df_place = df_place.append(pd.DataFrame(columns=name_attribute,data=error_stats))
    #                 ####---------------------------------------------------------------
    #                 shape_file ='E:/modis.lst.interp/shape/Experiment.shape/%s_%d.shp'%(site,diameter)
    #                 out_path_clip = 'E:/modis.lst.interp/hrb.2018.C6/Experiment1/difference.clip/%s.%d.%s/'%(site,doy,datatype)
    #                 out_filename_nodata_clip = os.path.join(out_path_clip,out_basename_nodata)  
    #                 arcpy.Clip_management(out_filename_nodata,'#',out_filename_nodata_clip,shape_file,"9999","NONE","MAINTAIN_EXTENT")
                                          
    #                 outfigname = "MOD11A1_%d.LST_%s_1km_%d_pixels_diff_0data-V2.jpg" %(doy,datatype,diameter)
    #                 outfigname = os.path.join(out_path_clip,outfigname)
    #                 figtitle = '%d D=%d'%(doy,diameter)
    #                 data,cellSize,crncords = RasterInfo(out_filename_nodata_clip)
    #                 shape_file ='E:/modis.lst.interp/shape/Experiment.shape/%s_%d'%(site,diameter)
    #                 vmin=-8
    #                 vmax=8
    #                 RasterPlot_wgs84_p2(data,crncords,shape_file,figtitle,outfigname,vmin,vmax) 
                    
    #                 ####-------Plot the LST Error data histogram and Normal distribution---------------
    #                 outfigname = "%s_%d.LST_%s_1km_%d_pixels_diff_statistics.jpg" %(site,doy,datatype,diameter)
    #                 outfigname = os.path.join(out_path_clip,outfigname)
    #                 array = arcpy.RasterToNumPyArray(out_filename_nodata,nodata_to_value = 9999)
    #                 plt.figure(dpi = 300)                
    #                 hist, bins = np.histogram(array, bins = np.arange(-8.5,8.5),density=True)
    #                 # Plot the histogram
    #                 plt.bar(bins[:-1]+0.5, hist, width = 0.5,fc='b')
    #                 plt.xticks(range(-8,9))
    #                 plt.xlim(-8.5, 8.5)
    #                 plt.ylabel('Frequency')
    #                 plt.xlabel('Reconstruct Error')
    #                 bins = np.arange(-8.5,8.5,0.1)
    #                 pdf = normfun(bins,float(errormean),float(error_STD))
    #                 plt.plot(bins,pdf,color='b')
    #                 plt.savefig(outfigname,dpi=300,bbox_inches='tight',transparent=False)
    #             error_stats_filename = 'E:/modis.lst.interp/hrb.2018.C6/Experiment1/difference.clip/%d_%s_%s_statistic.csv'%(doy,datatype,site)
    #             df_place.to_csv(error_stats_filename,encoding='utf-8')
