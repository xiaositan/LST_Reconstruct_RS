# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 10:02:51 2021

@author: Junlei Tan
"""

import numpy as np
import os
import arcpy
from datetime import datetime
import calendar
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
plt.rcParams['font.sans-serif'] = ['Palatino Linotype']  # 字体设置
plt.rcParams['xtick.direction'] = 'out' 
plt.rcParams['ytick.direction'] = 'out' 

# Check out any necessary licenses
arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True
def QA_statistics_annual_mean(filename):
    filebasename = os.path.basename(filename)
    year = filebasename[0:4]
    dataarr = arcpy.RasterToNumPyArray(filename)
    nrows,ncols = dataarr.shape
    averageDAY = np.average(dataarr)
    print year
    print averageDAY
    return averageDAY
def QA_statistics_bit10_eq_10_11(qc_abspath,wildcard,out_abspath,suffix):
    arcpy.env.workspace = qc_abspath
    rasters = arcpy.ListRasters(wildcard,'TIF')
    year = rasters[0][8:12]
    refraster = arcpy.Raster(os.path.join(qc_abspath,rasters[0]))
    refdata = arcpy.RasterToNumPyArray(refraster)
    lines,samples=refdata.shape
    lowerLeft = arcpy.Point(refraster.extent.XMin,refraster.extent.YMin)
    cellSize = refraster.meanCellWidth
#    mask = 0b10
    yearsum=np.zeros((lines,samples),dtype=np.uint16)
    #print yearsum.shape
    for raster in rasters:
        inraster = os.path.join(qc_abspath,raster)
        # Process: Raster Calculator
        arr = arcpy.RasterToNumPyArray(inraster)
        #print arr
        temp =  arr<<6
        desire =  temp>>6   
        desire[desire<2]=0
        desire[desire>1]=1
        yearsum = yearsum+desire
    newRaster = arcpy.NumPyArrayToRaster(yearsum,lowerLeft,cellSize)
    out_filename = os.path.join(out_abspath,year+suffix)
    print year+suffix+'Annual Average DAYS'
    print np.average(yearsum)
    if os.path.isfile(out_filename):
        os.remove(out_filename)
    newRaster.save(out_filename)
def QA_statistics_bit10_eq_01_00(qc_abspath,wildcard,out_abspath,suffix):
    arcpy.env.workspace = qc_abspath
    rasters = arcpy.ListRasters(wildcard,'TIF')
    year = rasters[0][8:12]
    refraster = arcpy.Raster(os.path.join(qc_abspath,rasters[0]))
    refdata = arcpy.RasterToNumPyArray(refraster)
    lines,samples=refdata.shape
    lowerLeft = arcpy.Point(refraster.extent.XMin,refraster.extent.YMin)
    cellSize = refraster.meanCellWidth
#    mask = 0b10
    yearsum=np.zeros((lines,samples),dtype=np.uint16)
    #print yearsum.shape
    for raster in rasters:
        inraster = os.path.join(qc_abspath,raster)
        # Process: Raster Calculator
        arr = arcpy.RasterToNumPyArray(inraster)
        #print arr
        temp =  arr<<6
        desire =  temp>>6   
        desire[desire<2]=1
        desire[desire>1]=0
        yearsum = yearsum+desire
    newRaster = arcpy.NumPyArrayToRaster(yearsum,lowerLeft,cellSize)
    out_filename = os.path.join(out_abspath,year+suffix)
    print year+suffix+'Annual Average DAYS'
    print np.average(yearsum)
    if os.path.isfile(out_filename):
        os.remove(out_filename)
    newRaster.save(out_filename)
def QA_statistics_bit10_eq_01_10_11(qc_abspath,wildcard,out_abspath,suffix):
    arcpy.env.workspace = qc_abspath
    rasters = arcpy.ListRasters(wildcard,'TIF')
    year = rasters[0][8:12]
    refraster = arcpy.Raster(os.path.join(qc_abspath,rasters[0]))
    refdata = arcpy.RasterToNumPyArray(refraster)
    lines,samples=refdata.shape
    lowerLeft = arcpy.Point(refraster.extent.XMin,refraster.extent.YMin)
    cellSize = refraster.meanCellWidth
#    mask = 0b10
    yearsum=np.zeros((lines,samples),dtype=np.uint16)
    #print yearsum.shape
    for raster in rasters:
        inraster = os.path.join(qc_abspath,raster)
        # Process: Raster Calculator
        arr = arcpy.RasterToNumPyArray(inraster)
        #print arr
        temp =  arr<<6
        desire =  temp>>6   
        desire[desire>0]=1
        yearsum = yearsum+desire
        #doy = raster[12:15]
        #print doy,np.average(desire)
    newRaster = arcpy.NumPyArrayToRaster(yearsum,lowerLeft,cellSize)
    out_filename = os.path.join(out_abspath,year+suffix)
    print year+suffix+'Annual Average DAYS'
    print np.nanmean(yearsum)
    if os.path.isfile(out_filename):
        os.remove(out_filename)
    newRaster.save(out_filename)
def QA_statistics_bit10_eq_01_10_11_daily(qc_abspath,wildcard):
    arcpy.env.workspace = qc_abspath
    rasters = arcpy.ListRasters(wildcard,'TIF')
    refraster = arcpy.Raster(os.path.join(qc_abspath,rasters[0]))
    refdata = arcpy.RasterToNumPyArray(refraster)
    lines,samples=refdata.shape
    #print yearsum.shape
    doylist = []
    daymeanlist = []
    for raster in rasters:
        inraster = os.path.join(qc_abspath,raster)
        doy = raster[8:15]
        doylist.append(doy)
        # Process: Raster Calculator
        dataarry = arcpy.RasterToNumPyArray(inraster)
        mask = [dataarry==65535]
        dataarry=np.ma.masked_array(dataarry,mask=mask)
        #print arr
        temp =  dataarry<<6
        desire =  temp>>6   
        desire[desire>0]=1
        desire=np.ma.masked_array(desire,mask=mask)
        #print doy,'mean ',np.mean(desire)
        daymeanlist.append(np.ma.mean(desire))
        #print doy,'count',np.ma.count(desire)
    dailymean_list = list(zip(doylist,daymeanlist))
    return dailymean_list

def QA_statistics_bit10_eq_01_10_11_month(qc_abspath,wildcard,out_abspath,suffix):
    # qc_abspath: QC file path
    # wildcard: filter word
    # out_abspath: output file path
    arcpy.env.workspace = qc_abspath
    rasters = arcpy.ListRasters(wildcard,'TIF')
    year = rasters[0][8:12]
    refraster = arcpy.Raster(os.path.join(qc_abspath,rasters[0]))
    refdata = arcpy.RasterToNumPyArray(refraster)
    lines,samples=refdata.shape
    print lines,samples
    lowerLeft = arcpy.Point(refraster.extent.XMin,refraster.extent.YMin)
    cellSize = refraster.meanCellWidth
    monthlist = []
    month_statistic_list = []
    for month in range(1,13):
        monthsum=np.zeros((lines,samples),dtype=np.int)
        for raster in rasters:
            yeardoy = raster[8:15]
            rasterdate = datetime.strptime(yeardoy,"%Y%j")
            m = rasterdate.month
            dayofmonth = calendar.monthrange(rasterdate.year,m)[1] 
            if (month == m):
                inraster = os.path.join(qc_abspath,raster)
                # Process: Raster Calculator
                dataarry = arcpy.RasterToNumPyArray(inraster)
                mask = [dataarry==65535]
                dataarry=np.ma.masked_array(dataarry,mask=mask)
                #print arr
                temp =  dataarry<<6
                desire =  temp>>6   
                desire[desire>0]=1
                desire=np.ma.masked_array(desire,mask=mask)
                #print type(desire)
                monthsum = monthsum+desire.data
                #doy = raster[12:15]
                #print doy,np.average(desire)
        #print type(monthsum)
        monthsum[mask]=65535
        monthsum=np.ma.masked_array(monthsum,mask=mask)
        newRaster = arcpy.NumPyArrayToRaster(monthsum.data,lowerLeft,cellSize)
        out_filename = os.path.join(out_abspath,year+format(month,'0>2d')+suffix)
        if os.path.isfile(out_filename):
            os.remove(out_filename)
        newRaster.save(out_filename)
        monthlist.append(year+format(month,'0>2d'))        
        month_statistic_list.append(np.ma.average(monthsum)/dayofmonth)#/dayofmonth
        #print year+format(month,'0>2d')+' '+str(np.average(monthsum)/dayofmonth*100)
    monthmean_list = list(zip(monthlist,month_statistic_list))
    return monthmean_list
def QA_statistics_bit10_eq_01_10_11_yearly(qc_abspath,wildcard,out_abspath,suffix):
    # qc_abspath: QC file path
    # wildcard: filter word
    # out_abspath: output file path
    arcpy.env.workspace = qc_abspath
    rasters = arcpy.ListRasters(wildcard,'TIF')
    year = rasters[0][8:12]
    refraster = arcpy.Raster(os.path.join(qc_abspath,rasters[0]))
    refdata = arcpy.RasterToNumPyArray(refraster)
    lines,samples=refdata.shape
    lowerLeft = arcpy.Point(refraster.extent.XMin,refraster.extent.YMin)
    cellSize = refraster.meanCellWidth
    datasum=np.zeros((lines,samples),dtype=np.int)
    for raster in rasters:
        # Calculate days number of the special year        
        year_days = 366 if calendar.isleap(int(year)) else 365
        inraster = os.path.join(qc_abspath,raster)
        # Process: Raster Calculator
        dataarry = arcpy.RasterToNumPyArray(inraster)
        mask = [dataarry==65535]
        dataarry=np.ma.masked_array(dataarry,mask=mask)
        #print arr
        temp =  dataarry<<6
        desire =  temp>>6   
        desire[desire>0]=1
        desire=np.ma.masked_array(desire,mask=mask)
        #print type(desire)
        datasum = datasum+desire.data
        #print type(monthsum)
    datasum[mask]=65535
    datasum=np.ma.masked_array(datasum,mask=mask)
    newRaster = arcpy.NumPyArrayToRaster(datasum.data,lowerLeft,cellSize)
    out_filename = os.path.join(out_abspath,year+suffix)
    if os.path.isfile(out_filename):
        os.remove(out_filename)
    newRaster.save(out_filename)
    return np.ma.average(datasum)/year_days
    
def QA_statistics_bit10_eq_01(qc_abspath,wildcard,out_abspath,suffix):
    arcpy.env.workspace = qc_abspath
    rasters = arcpy.ListRasters(wildcard,'TIF')
    year = rasters[0][8:12]
    refraster = arcpy.Raster(os.path.join(qc_abspath,rasters[0]))
    refdata = arcpy.RasterToNumPyArray(refraster)
    lines,samples=refdata.shape
    lowerLeft = arcpy.Point(refraster.extent.XMin,refraster.extent.YMin)
    cellSize = refraster.meanCellWidth
#    mask = 0b10
    yearsum=np.zeros((lines,samples),dtype=np.uint16)
    #print yearsum.shape
    for raster in rasters:
        inraster = os.path.join(qc_abspath,raster)
        # Process: Raster Calculator
        arr = arcpy.RasterToNumPyArray(inraster)
        #print arr
        temp =  arr<<6
        desire =  temp>>6   
        desire[desire!=1]=0
        yearsum = yearsum+desire
    newRaster = arcpy.NumPyArrayToRaster(yearsum,lowerLeft,cellSize)
    out_filename = os.path.join(out_abspath,year+suffix)
    print year+suffix+'Annual Average DAYS'
    print np.average(yearsum)
    if os.path.isfile(out_filename):
        os.remove(out_filename)
    newRaster.save(out_filename)
def QA_statistics_bit10_eq_10(qc_abspath,wildcard,out_abspath,suffix):
    arcpy.env.workspace = qc_abspath
    rasters = arcpy.ListRasters(wildcard,'TIF')
    year = rasters[0][8:12]
    refraster = arcpy.Raster(os.path.join(qc_abspath,rasters[0]))
    refdata = arcpy.RasterToNumPyArray(refraster)
    lines,samples=refdata.shape
    lowerLeft = arcpy.Point(refraster.extent.XMin,refraster.extent.YMin)
    cellSize = refraster.meanCellWidth
#    mask = 0b10
    yearsum=np.zeros((lines,samples),dtype=np.uint16)
    #print yearsum.shape
    for raster in rasters:
        inraster = os.path.join(qc_abspath,raster)
        # Process: Raster Calculator
        arr = arcpy.RasterToNumPyArray(inraster)
        #print arr
        temp =  arr<<6
        desire =  temp>>6
        desire[desire!=2]=0
        desire[desire==2]=1
        yearsum = yearsum+desire
    newRaster = arcpy.NumPyArrayToRaster(yearsum,lowerLeft,cellSize)
    out_filename = os.path.join(out_abspath,year+suffix)
    print year+suffix+'Annual Average DAYS'
    print np.average(yearsum)
    if os.path.isfile(out_filename):
        os.remove(out_filename)
    newRaster.save(out_filename)
def QA_statistics_bit10_eq_11(qc_abspath,wildcard,out_abspath,suffix):
    arcpy.env.workspace = qc_abspath
    rasters = arcpy.ListRasters(wildcard,'TIF')
    year = rasters[0][8:12]
    refraster = arcpy.Raster(os.path.join(qc_abspath,rasters[0]))
    refdata = arcpy.RasterToNumPyArray(refraster)
    lines,samples=refdata.shape
    lowerLeft = arcpy.Point(refraster.extent.XMin,refraster.extent.YMin)
    cellSize = refraster.meanCellWidth
#    mask = 0b10
    yearsum=np.zeros((lines,samples),dtype=np.uint16)
    #print yearsum.shape
    for raster in rasters:
        inraster = os.path.join(qc_abspath,raster)
        # Process: Raster Calculator
        arr = arcpy.RasterToNumPyArray(inraster)
        #print arr
        temp =  arr<<6
        desire =  temp>>6
        desire[desire!=3]=0
        desire[desire==3]=1
        yearsum = yearsum+desire
    newRaster = arcpy.NumPyArrayToRaster(yearsum,lowerLeft,cellSize)
    out_filename = os.path.join(out_abspath,year+suffix)
    print year+suffix+'Annual Average DAYS'
    print np.average(yearsum)
    if os.path.isfile(out_filename):
        os.remove(out_filename)
    newRaster.save(out_filename)
def QA_statistics_yearly_mean(qc_abspath,wildcard,out_abspath,suffix):
    arcpy.env.workspace = qc_abspath
    rasters = arcpy.ListRasters(wildcard,'tif')
    #print rasters
    year = '2013-2018'
    refraster = arcpy.Raster(os.path.join(qc_abspath,rasters[0]))
    refdata = arcpy.RasterToNumPyArray(refraster)
    lines,samples=refdata.shape
    lowerLeft = arcpy.Point(refraster.extent.XMin,refraster.extent.YMin)
    cellSize = refraster.meanCellWidth
    mask = [refdata==65535]
    #dataarry=np.ma.masked_array(dataarry,mask=mask)

#    mask = 0b10
    yearsum=np.zeros((lines,samples),dtype=np.double)
    #print yearsum.shape
    for raster in rasters:
        inraster = os.path.join(qc_abspath,raster)
        # Process: Raster Calculator
        yearsum = yearsum + np.ma.masked_array(arcpy.RasterToNumPyArray(inraster),mask=mask).data
    yearsum = yearsum/len(rasters)
    yearsum=np.ma.masked_array(yearsum,mask=mask)
    newRaster = arcpy.NumPyArrayToRaster(yearsum.data,lowerLeft,cellSize)
    out_filename = os.path.join(out_abspath,year+suffix)
    print year+suffix+'Annual Average DAYS'
    print np.ma.mean(yearsum)
    if os.path.isfile(out_filename):
        os.remove(out_filename)
    newRaster.save(out_filename)

######===========================================================================
######===========================================================================
######===========         Clip MOD11A1 Data by Heihe River Basin      ===========
###arcpy.env.nodata = "PROMOTION"    
#spatial_subset = '97.0 37.491666887 101.9999998 43.0'  #boundary coordinate
##### Boundary of Heihe River Basin
#shapedir = '../../shape/'
#clpshape_abspath = os.path.join(os.path.abspath(shapedir),'heihe_huangwei.shp')
#fillvalue = '65535'  #filled value when outside of boundary
#qc_modday_dir = '../../hrb.%d.C6/lst.MOD11A1/600.661/tif'
#wildcard = '*QC*.tif'
#out_dir = '../../hrb.%d.C6/lst.MOD11A1/600.661/clp'
#arcpy.env.overwriteOutput = True
##[2013,2014,2015,2016,2017,2018]
#for year in [2013,2014,2015,2016,2017,2018]:
#    qc_modday_abspath = os.path.abspath(qc_modday_dir%year)
#    out_abspath = os.path.abspath(out_dir%year)
#    arcpy.env.workspace = qc_modday_abspath
#    #get all rasters in the datadir
#    rasters = arcpy.ListRasters(wildcard,'tif')
#    #loop
#    for raster in rasters:
#        # Local variables...
#        Output_Raster = os.path.join(out_abspath,raster)
#        print Output_Raster
#        # Process: Clip...
#        # Bound: Huangweihui, Output PIXELS: 650*600
#        arcpy.Clip_management(raster,spatial_subset ,Output_Raster,clpshape_abspath, fillvalue,"ClippingGeometry","MAINTAIN_EXTENT")  
#        #arcpy.Clip_management(raster,# ,Output_Raster,clpshape_abspath, fillvalue,"ClippingGeometry","MAINTAIN_EXTENT")                    
####===========================================================================
####===========================================================================
####=== Daily Percentage of Missing Data Statistics Day and Night =========================
#
#qc_modday_dir = '../../hrb.%d.C6/lst.MOD11A1/600.661/clp'
#wildcard_day = '*.QC_Day.tif'
#statistic_file_day = '../../statistics/Nodata.SingleYear.Day.%d.csv'
#wildcard_night = '*.QC_Night.tif'
#statistic_file_night = '../../statistics/Nodata.SingleYear.Night.%d.csv'
#name=['Date','NoDataRate']
#for year in [2013,2014,2015,2016,2017,2018]:
#    ####-------------------     Day    ----------------------------------------
#    qc_modday_abspath = os.path.abspath(qc_modday_dir%year)
#    nodatarate_list = QA_statistics_bit10_eq_01_10_11_daily(qc_modday_abspath,wildcard_day)
#    nodatarate=pd.DataFrame(columns=name,data=nodatarate_list)#数据有二列，列名分别为Date,NoDataRate
#    nodatarate.to_csv(statistic_file_day%year,encoding='gbk')
#    #### matplotlib 画图常见参数
#    fig, axs = plt.subplots(2,1)
#    fig.suptitle('%d'%year,fontsize=12,y=0.96)
#    fig.text(0.08,0.7,'Percentage of Missing Data(%)',fontsize=12,rotation=90)
#    #nodatarate['Date'] = nodatarate['Date'].apply(lambda x:datetime.strptime(x,'%Y%j'))
#    axs[0].bar(nodatarate.index,nodatarate['NoDataRate']*100,facecolor='grey',edgecolor = 'grey',linewidth=0.03)
#    axs[0].set_title('Terra Day(10:30)',fontsize=12,color='black') #r: red
#    #axs[0].set_xlabel('DOY')
#    axs[0].set_xlim(0,364)
#    xlocator = [29,89,149,209,269,329]
#    x_ticklabels = ['30','90','150','210','270','330']
#    axs[0].set_xticks(xlocator) # set x ticks location
#    axs[0].set_xticklabels(x_ticklabels)# plot the Xticks 
#    #axs[0].set_ylabel('Percentage of Missing Data(%)')
#    ####-------------------     Night    ----------------------------------------
#    nodatarate_list = QA_statistics_bit10_eq_01_10_11_daily(qc_modday_abspath,wildcard_night)
#    nodatarate=pd.DataFrame(columns=name,data=nodatarate_list)#数据有二列，列名分别为Date,NoDataRate
#    nodatarate.to_csv(statistic_file_night%year,encoding='gbk')
#    #### matplotlib 画图常见参数
#    #nodatarate['Date'] = nodatarate['Date'].apply(lambda x:datetime.strptime(x,'%Y%j'))
#    axs[1].bar(nodatarate.index,nodatarate['NoDataRate']*100,facecolor='grey',edgecolor = 'grey',linewidth=0.03)
#    axs[1].set_title('Terra Night(22:30)',fontsize=12,color='black') #r: red
#    axs[1].set_xlabel('DOY',fontsize=12)
#    axs[1].set_xlim(0,364)
#    xlocator = [29,89,149,209,269,329]
#    x_ticklabels = ['30','90','150','210','270','330']
#    axs[1].set_xticks(xlocator) # set x ticks location
#    axs[1].set_xticklabels(x_ticklabels)# plot the Xticks 
#    #axs[1].set_ylabel('Percentage of Missing Data(%)')    
#    #### Save the Figure of Every Day's Day and Night Data Missing Percentage Every Year
#    outputjpg = '../../statistics/Nodata.SingleYear.Daily.%d.jpg'%year
#    fig.set_size_inches(15, 8)
#    fig.savefig(outputjpg,dpi=600,bbox_inches='tight')#,bbox_inches='tight'
#    plt.show()


#####===========================================================================
#####===========================================================================
#####===========         Month No Data Statistics      =========================
#qc_modday_dir = '../../hrb.%d.C6/lst.MOD11A1/600.661/clp'
#out_abspath = os.path.abspath('../../statistics/Nodata.Month')   
#if (os.path.exists(out_abspath)==False):
#    os.makedirs(out_abspath) 
# 
#name=['Date','NoDataRate']
#wildcard_day = '*.QC_Day.tif'
#statistic_file_day = '../../statistics/Nodata.Month/Nodata.SingleYear.Month.Day.%d.csv'
#suffix_day = '_MOD_QC_Day_bit10.eq.01.10.11.tif'
#
#wildcard_night = '*.QC_Night.tif'
#statistic_file_night = '../../statistics/Nodata.Month/Nodata.SingleYear.Month.Night.%d.csv'
#suffix_night = '_MOD_QC_Night_bit10.eq.01.10.11.tif'
#
#for year in [2013,2014,2015,2016,2017,2018]:
#    qc_modday_abspath = os.path.abspath(qc_modday_dir%year)
#    nodatarate_list = QA_statistics_bit10_eq_01_10_11_month(qc_modday_abspath,wildcard_day,out_abspath,suffix_day)
#    nodatarate=pd.DataFrame(columns=name,data=nodatarate_list)#数据有二列，列名分别为Date,NoDataRate
#    nodatarate.to_csv(statistic_file_day%year,encoding='gbk')
#    #### matplotlib 画图常见参数
#    fig, axs = plt.subplots(2,1)
#    fig.suptitle('%d'%year,fontsize=12,y=0.96)
#    fig.text(0.08,0.7,'Percentage of Missing Data(%)',fontsize=12,rotation=90)
#    #nodatarate['Date'] = nodatarate['Date'].apply(lambda x:datetime.strptime(x,'%Y%j'))
#    axs[0].plot(nodatarate['NoDataRate']*100,'-o',color='black',linewidth=0.5)
#    axs[0].set_title('Terra Day(10:30)',fontsize=12,color='black') #r: red
#    #axs[0].set_xlabel('DOY')
#    axs[0].set_xlim(-0.5,11.5)
#    xlocator = range(0,12)
#    x_ticklabels = range(1,13)
#    axs[0].set_xticks(xlocator) # set x ticks location
#    axs[0].set_xticklabels(x_ticklabels)# plot the Xticks 
#    axs[0].set_ylim(20,80)
#    #axs[0].set_ylabel('Percentage of Missing Data(%)')
#    
#    nodatarate_list = QA_statistics_bit10_eq_01_10_11_month(qc_modday_abspath,wildcard_night,out_abspath,suffix_night)
#    nodatarate=pd.DataFrame(columns=name,data=nodatarate_list)#数据有二列，列名分别为Date,NoDataRate
#    nodatarate.to_csv(statistic_file_night%year,encoding='gbk')
#    axs[1].plot(nodatarate['NoDataRate']*100,'-o',color='black',linewidth=0.5)
#    axs[1].set_title('Terra Night(22:30)',fontsize=12,color='black') #r: red
#    axs[1].set_xlabel('Month',fontsize=12)
#    axs[1].set_xlim(-0.5,11.5)
#    xlocator = range(0,12)
#    x_ticklabels = range(1,13)
#    axs[1].set_xticks(xlocator) # set x ticks location
#    axs[1].set_xticklabels(x_ticklabels)# plot the Xticks 
#    axs[1].set_ylim(10,70)
#    #axs[0].set_ylabel('Percentage of Missing Data(%)')    
#    
#    #### Save the Figure of Every Day's Day and Night Data Missing Percentage Every Year
#    outputjpg = '../../statistics/Nodata.Month/Nodata.SingleYear.Month.%d.jpg'%year
#    fig.set_size_inches(15, 8)
#    fig.savefig(outputjpg,dpi=600,bbox_inches='tight')#,bbox_inches='tight'
#    plt.show()    

####===========================================================================
####===========================================================================
####===========         Yearly No Data Statistics      =========================
qc_modday_dir = '../../hrb.%d.C6/lst.MOD11A1/600.661/clp'
out_abspath = os.path.abspath('../../statistics/Nodata.Yearly')   
if (os.path.exists(out_abspath)==False):
    os.makedirs(out_abspath) 
 
name=['Date','NoDataRate']
wildcard_day = '*QC_Day.tif'
statistic_file_day = '../../statistics/Nodata.Yearly/Nodata.SingleYearly.Day.csv'
suffix_day = '_MOD_QC_Day_bit10.eq.01.10.11.tif'

wildcard_night = '*QC_Night.tif'
statistic_file_night = '../../statistics/Nodata.Yearly/Nodata.SingleYearly.Night.csv'
suffix_night = '_MOD_QC_Night_bit10.eq.01.10.11.tif'

year_list =[2013,2014,2015,2016,2017,2018]
nodata_list = [None]*len(year_list)
for year in year_list:
    qc_modday_abspath = os.path.abspath(qc_modday_dir%year)
    nodata_list[year_list.index(year)] = QA_statistics_bit10_eq_01_10_11_yearly(qc_modday_abspath,wildcard_day,out_abspath,suffix_day)
yearlymean_list = list(zip(year_list,nodata_list))
nodatarate=pd.DataFrame(columns=name,data=yearlymean_list)#数据有二列，列名分别为Date,NoDataRate
nodatarate.to_csv(statistic_file_day,encoding='gbk')
#### matplotlib 画图常见参数
fig, axs = plt.subplots(2,1)
fig.suptitle('%d'%year,fontsize=12,y=0.96)
fig.text(0.08,0.7,'Percentage of Missing Data(%)',fontsize=12,rotation=90)
#nodatarate['Date'] = nodatarate['Date'].apply(lambda x:datetime.strptime(x,'%Y%j'))
axs[0].plot(nodatarate['NoDataRate']*100,'-o',color='black',linewidth=0.5)
axs[0].set_title('Terra Day(10:30)',fontsize=12,color='black') #r: red
#axs[0].set_xlabel('DOY')
axs[0].set_xlim(-0.5,5.5)
xlocator = range(0,6)
x_ticklabels = range(2013,2019)
axs[0].set_xticks(xlocator) # set x ticks location
axs[0].set_xticklabels(x_ticklabels)# plot the Xticks 
axs[0].set_ylim(40,80)
#axs[0].set_ylabel('Percentage of Missing Data(%)')

nodata_list = [None]*len(year_list)
for year in year_list:
    qc_modday_abspath = os.path.abspath(qc_modday_dir%year)
    nodata_list[year_list.index(year)] = QA_statistics_bit10_eq_01_10_11_yearly(qc_modday_abspath,wildcard_night,out_abspath,suffix_night)
yearlymean_list = list(zip(year_list,nodata_list))
nodatarate=pd.DataFrame(columns=name,data=yearlymean_list)#数据有二列，列名分别为Date,NoDataRate
nodatarate.to_csv(statistic_file_night,encoding='gbk')
axs[1].plot(nodatarate['NoDataRate']*100,'-o',color='black',linewidth=0.5)
axs[1].set_title('Terra Night(22:30)',fontsize=12,color='black') #r: red
axs[1].set_xlabel('Year',fontsize=12)
axs[1].set_xlim(-0.5,5.5)
xlocator = range(0,6)
x_ticklabels = range(2013,2019)
axs[1].set_xticks(xlocator) # set x ticks location
axs[1].set_xticklabels(x_ticklabels)# plot the Xticks 
axs[1].set_ylim(40,70)
axs[0].set_ylabel('Percentage of Missing Data(%)')    

#### Save the Figure of Every Day's Day and Night Data Missing Percentage Every Year
outputjpg = '../../statistics/Nodata.Month/Nodata.SingleYear.Month.%d.jpg'%year
fig.set_size_inches(15, 8)
fig.savefig(outputjpg,dpi=600,bbox_inches='tight')#,bbox_inches='tight'
plt.show()    
#####============================================================================
###===========================================================================
###=== Caculate Night Data From Single Year to MultiYears No Data Statistic===
###---------------------------------------------------------------------------
###------------------         Common Varibles               ------------------

####----------Day
qc_daily_abspath = os.path.abspath('../../statistics/Nodata.Yearly')
wildcard = '*_MOD_QC_Day_bit10.eq.01.10.11.tif'
out_dir = '../../statistics/Nodata.MultiYears'
suffix = '_MOD_QC_Day_bit10.eq.01.10.11.tif'
out_abspath = os.path.abspath(out_dir)
if (os.path.exists(out_abspath)==False):
    os.makedirs(out_abspath)  
qc_daily_dir =  '../../statistics/Nodata.SingleYear'
QA_statistics_yearly_mean(qc_daily_abspath,wildcard,out_abspath,suffix)

####----------Night
qc_daily_abspath = os.path.abspath('../../statistics/Nodata.Yearly')
wildcard = '*_MOD_QC_Night_bit10.eq.01.10.11.tif'
out_dir = '../../statistics/Nodata.MultiYears'
suffix = '_MOD_QC_Night_bit10.eq.01.10.11.tif'
out_abspath = os.path.abspath(out_dir)
if (os.path.exists(out_abspath)==False):
    os.makedirs(out_abspath)  
qc_daily_dir =  '../../statistics/Nodata.SingleYear'
QA_statistics_yearly_mean(qc_daily_abspath,wildcard,out_abspath,suffix)
    
####============================================================================
####================ Caculate From Year to Multi Years No Data Statistic========
#out_dir = '../../statistics/Nodata.MultiYears'
####----------------------------------------------------------------------------
#qc_modday_dir =  '../../statistics/Nodata.SingleYear'
#wildcard = '*_MOD_QC_Day_bit10.eq.01.10.11.tif'
#qc_modday_abspath = os.path.abspath(qc_modday_dir)
#out_abspath = os.path.abspath(out_dir)
#if (os.path.exists(out_abspath)==False):
#    os.makedirs(out_abspath)  
#suffix = '_MOD_QC_Day_bit10.eq.01.10.11.tif' 
#QA_statistics_yearly_mean(qc_modday_abspath,wildcard,out_abspath,suffix)



    
#qc_modday_dir = '../../hrb.%d.C6/lst.MOD11A1/600.661/tif'
#wildcard = '*.QC_Night.tif'
#out_abspath = os.path.abspath('../../statistics/Nodata.Month')   
#if (os.path.exists(out_abspath)==False):
#    os.makedirs(out_abspath) 
#suffix = '_MOD_QC_Night_bit10.eq.01.10.11.tif' 
#for year in [2013,2014,2015,2016,2017,2018]:
#    qc_modday_abspath = os.path.abspath(qc_modday_dir%year)
#    QA_statistics_bit10_eq_01_10_11_month(qc_modday_abspath,wildcard,out_abspath,suffix)

