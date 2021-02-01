# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:37:23 2020

@author: Administrator
"""
import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
import pandas as pd
import os
import scipy.stats
import csv
import datetime
# from compiler.ast import flatten
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
import collections
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from matplotlib.font_manager import FontProperties

def interp_validation_statistics(validation_file):
    df = pd.read_csv(validation_file,header=0,parse_dates={'DateTime':[0]})
    df = df[~df['LST_Insitu'].isnull()]
    df_interp = df[df.LST_Ori == 0.0]
    df_interp = df_interp[df_interp.LST_Interp != 0]
    # Interpolation with In-Situ data
    x = np.array(df_interp.LST_Insitu)
    y = np.array(df_interp.LST_Interp)
    slope_interp, intercept_interp, r_value_interp, p_value_interp, stderr_gradient = scipy.stats.linregress(x, y)
    rmse_interp = scipy.sqrt(scipy.mean((y-x)**2))
    bias_interp = scipy.mean(y-x)
    mae_interp = scipy.mean(abs(y-x))
    std_err_interp = np.std(y-x)
    data_number = len(df_interp.LST_Insitu)
    # image_text = 'Eq.: y=%6.2fx %+6.2f\nR-Square=%5.2f\nRMSE=%5.2f\nBIAS=%5.2f\nMAE=%5.2f\nN=%d'\
    #             %(slope_interp,intercept_interp,r_value_interp**2,rmse_interp,bias_interp,mae_interp,data_number)
    # statistics_name = ['Slope','Intercept', 'R-Square','p_value','std_err','RMSE', 'BIAS', 'MAE', 'DataNumber']
    # statistics_value = [slope_interp,intercept_interp,r_value_interp**2, p_value_interp, std_err_interp,rmse_interp,bias_interp,mae_interp,data_number]
    image_text = 'y=%5.2fx%+6.2f\nR$^2$=%4.2f\nN=%d'\
                %(slope_interp,intercept_interp,r_value_interp**2,data_number)
    statistics_name = ['Slope','Intercept', 'R-Square','p_value','std_err','RMSE', 'BIAS', 'MAE', 'DataNumber']
    statistics_value = [slope_interp,intercept_interp,float('%.2f' % r_value_interp**2), p_value_interp, 
                        float('%.2f' % std_err_interp),float('%.2f' % rmse_interp),float('%.2f' % bias_interp),float('%.2f' % mae_interp),data_number]
    return df_interp,image_text,statistics_name,statistics_value


def interp_validation_statistics_Month(validation_file,month):
    df = pd.read_csv(validation_file,header=0,parse_dates={'DateTime':[0]},infer_datetime_format=True)
    df = df[~df['LST_Insitu'].isnull()]
    df_interp = df[df.LST_Ori == 0.0]
    df_interp = df_interp[df_interp.LST_Interp != 0]
    df_interp['Month'] = pd.to_datetime(df_interp['DateTime']).map(lambda x: x.month)
    df_interp = df_interp[df_interp['Month'] == month]
    # Interpolation with In-Situ data
    x = np.array(df_interp.LST_Insitu)
    y = np.array(df_interp.LST_Interp)
    slope_interp, intercept_interp, r_value_interp, p_value_interp, stderr_gradient = scipy.stats.linregress(x, y)
    rmse_interp = scipy.sqrt(scipy.mean((y-x)**2))
    bias_interp = scipy.mean(y-x)
    mae_interp = scipy.mean(abs(y-x))
    std_err_interp = np.std(y-x)
    data_number = len(df_interp.LST_Insitu)
#    image_text = 'Eq.: y=%6.2fx %+6.2f\nR-Square=%5.2f\nRMSE=%5.2f\nBIAS=%5.2f\nMAE=%5.2f\nN=%d'\
#                %(slope_interp,intercept_interp,r_value_interp**2,rmse_interp,bias_interp,mae_interp,data_number)
    # image_text = 'R-Square=%5.2f\nRMSE=%5.2f\nBIAS=%5.2f\nMAE=%5.2f\nN=%d'\
    #             %(r_value_interp**2,rmse_interp,bias_interp,mae_interp,data_number)
    image_text = 'y=%5.2fx%+6.2f\nR$^2$=%4.2f\nN=%d'\
                %(slope_interp, intercept_interp, r_value_interp**2, data_number)
    statistics_name = ['Slope','Intercept', 'R-Square','p_value','std_err','RMSE', 'BIAS', 'MAE', 'DataNumber']
    statistics_value = [slope_interp,intercept_interp,float('%.2f' % r_value_interp**2), p_value_interp, 
                        float('%.2f' % std_err_interp),float('%.2f' % rmse_interp),float('%.2f' % bias_interp),float('%.2f' % mae_interp),data_number]
    return df_interp,image_text,statistics_name,statistics_value
def interp_file_corrected_by_month_statistic(interp_file,calibration_coefficient):
    df = pd.read_csv(interp_file, header=0, parse_dates={'DateTime': [0]}, infer_datetime_format=True)
    df = df[~df['LST_Insitu'].isnull()]
    df_interp = df[df.LST_Ori == 0.0]
    df_interp = df_interp[df_interp.LST_Interp != 0]
    df_interp['Month'] = pd.to_datetime(df_interp['DateTime']).map(lambda x:x.month)
    df_interp['coefficient'] = df_interp['Month'].map(lambda x: calibration_coefficient.iloc[x-1])
    df_interp['LST_Interp_Corrected'] = df_interp.apply(lambda r: (r['LST_Interp']-r['coefficient']), axis=1)
    
    #df_interp = df_interp[df_interp['Month']==month]
    #df_interp['LST_Interp_Corrected'] = df_interp['LST_Interp']-calibration_coefficient.iloc[month-1]
    # Interpolation with In-Situ data
    x = np.array(df_interp.LST_Insitu)
    y = np.array(df_interp.LST_Interp_Corrected)
    slope_interp, intercept_interp, r_value_interp, p_value_interp, stderr_gradient = scipy.stats.linregress(x, y)
    rmse_interp = scipy.sqrt(scipy.mean((y-x)**2))
    bias_interp = scipy.mean(y-x)
    mae_interp  = scipy.mean(abs(y-x))
    std_err_interp = np.std(y-x)
    data_number = len(df_interp.LST_Insitu)
    image_text = 'y=%5.2fx%+6.2f\nR$^2$=%4.2f\nN=%d'\
                %(slope_interp,intercept_interp,r_value_interp**2,data_number)    
    statistics_name = ['Slope', 'Intercept', 'R-Square', 'p_value', 'std_err', 'RMSE', 'BIAS', 'MAE', 'DataNumber']
    statistics_value = [slope_interp,intercept_interp, float('%.2f' % r_value_interp**2), p_value_interp,
                        float('%.2f' % std_err_interp), float('%.2f' % rmse_interp), float('%.2f' % bias_interp), float('%.2f' % mae_interp), data_number]
    return df_interp, image_text, statistics_name, statistics_value


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

# ##========================= To concat the multi-years data for every sites====
# datatypes = ['MOD11A1.Day','MOD11A1.Night']
# years = range(2013,2019)
# sites = ['DMZ','SDZ','HZZ','SDQ','JCHM','ARC','DSL','HHL']#
# outputfile = '../../validation/result/reconstruct.2013-2018.V2/%s_LST_AWS_%s.csv'
# for datatype in datatypes:
#     for site in sites:
#         filelist = list([])
#         for year in years:
#             datadir = '../../hrb.%d.C6/lst.interp/600.661.V2/combined/validation/%s/%d_LST_AWS_%s.csv'%(year,datatype,year,site)
#             filelist.append(os.path.abspath(datadir))
#         print filelist
#         df_concatfile = interp_files_concat(filelist)
#         sitefile = os.path.abspath(outputfile %(datatype,site))
#         df_concatfile.to_csv(sitefile,index=False)
# print 'File is OK'
# years = range(2014,2019)
# sites = ['YKZ']#
# outputfile = '../../validation/result/reconstruct.2013-2018.V2/%s_LST_AWS_%s.csv'
# for datatype in datatypes:
#     filelist = list([])
#     for site in sites:
#         for year in years:
#             datadir = '../../hrb.%d.C6/lst.interp/600.661.V2/combined/validation/%s/%d_LST_AWS_%s.csv'%(year,datatype,year,site)
#             filelist.append(os.path.abspath(datadir))
#         df_concatfile = interp_files_concat(filelist)
#         sitefile = os.path.abspath(outputfile %(datatype,site))
#         df_concatfile.to_csv(sitefile,index=False)
# print 'File is OK'
# years = range(2015,2019)
# sites = ['HMZ']#
# outputfile = '../../validation/result/reconstruct.2013-2018.V2/%s_LST_AWS_%s.csv'
# for datatype in datatypes:
#     filelist = list([])
#     for site in sites:
#         for year in years:
#             datadir = '../../hrb.%d.C6/lst.interp/600.661.V2/combined/validation/%s/%d_LST_AWS_%s.csv'%(year,datatype,year,site)
#             filelist.append(os.path.abspath(datadir))
#         df_concatfile = interp_files_concat(filelist)
#         sitefile = os.path.abspath(outputfile %(datatype,site))
#         df_concatfile.to_csv(sitefile,index=False)
# print 'File is OK'
# ###

# =============================================================================
# Fig 1. Daman, Arou Super Station,Gebi, Huazhaizi, Shenshawo, Shidi, Jichang Huangmo
# ============FOR Each MONTH IN YEAY==========================
datatypes = ['MOD11A1.Day', 'MOD11A1.Night']
# For 10 sites
input_abspath = os.path.abspath(u'E:/modis.lst.interp/validation/result/reconstruct.2013-2018.V2/')
# Target output absolute filepath for 10 sites
target_abspath = os.path.abspath(u'E:/modis.lst.interp/validation/result/reconstruct.2013-2018.V2.Cor/')
# output figures target absolute filepath
figure_target_abspath = os.path.abspath(u'E:/mywork/Paper/LST_Reconstruct_RS/output_figures')
fontproperties = 'Palatino Linotype'
fontsize_big = 10
fontsize_middle = 9
fontsize_small = 7
fontsize_legend = 6
font0 = FontProperties()

font_cb_tick_label = font0.copy()
font_cb_tick_label.set_size('10')
font_cb_tick_label.set_weight('bold')

font_cb_title = font0.copy()
font_cb_title.set_size('10')
font_cb_title.set_weight('bold')

for datatype in datatypes:
    statistics_list = []

####---------------------------------------------------------------------------
    if datatype =='MOD11A1.Day':
        xy_min = 240
        xy_max = 340
        text_pos_x = 242
        text_pos_y = 303
        sitename_text_pos_x = 242
        sitename_text_pos_y = 328
        majors = [250, 270, 290, 310, 330]
    else:
        xy_min = 230
        xy_max = 310
        text_pos_x = 232
        text_pos_y = 280
        sitename_text_pos_x = 232
        sitename_text_pos_y = 300
        majors = [240, 260, 280, 300]
    ylabel = 'Theoretical Clear-sky LST (K)' #Land Surface Temperature
    xlabel = 'In Situ Observed Land Surface Temperature (K)'
    point_lable = 'Temperature'
    point_size = 2
    linewidth = 0.5
    statistics_list = []
    out_filename = 'Figure 5-6. The Theoretical Clear-sky LST Validation_2013-2018.%s-Scatter.Density.tiff' % (datatype)
    output_figure_abspath = os.path.join(figure_target_abspath, out_filename)
    fig = plt.figure(figsize=cm2inch(20, 8))
    # #### fig.text(0.49,0.94,'2016',fontsize=20)
    # plt.suptitle('\n2013-2018-%s'%(datatype),fontsize=fontsize_big)
    # ###-----------------------------------------------------------------------
    # ###Dman Station;subplot1
    site_name = 'DMZ'
    read_filename = '%s_LST_AWS_%s.csv' % (datatype, site_name)
    validation_file = os.path.join(input_abspath, read_filename)
    df, image_text, statistics_name, statistics_value = interp_validation_statistics(validation_file)
    x = np.array(df.LST_Insitu)
    y = np.array(df.LST_Interp)
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    ax = fig.add_subplot(2, 5, 1, ylim=(xy_min, xy_max), xlim=(xy_min, xy_max))
    density = ax.scatter(x, y, label=point_lable, s=point_size, marker='o', c=z, linewidths=0.5, edgecolors='', cmap=plt.cm.jet)
    ax.plot(x, x * statistics_value[0] + statistics_value[1], 'r', label='Fitted Line', linewidth=linewidth)
    ax.plot([xy_min, xy_max], [xy_min, xy_max], c='k', linewidth=linewidth)
    ax.text(text_pos_x, text_pos_y, image_text, fontsize=fontsize_small, style='oblique', ha='left', va='bottom', wrap=True)
    ax.text(sitename_text_pos_x, sitename_text_pos_y, site_name, fontsize=fontsize_big, ha='left', va='bottom',
            wrap=True)
    #ax.set_xlabel(u'Insitu Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    # ax.set_ylabel(u'Reconstruct-Correct Temperature(K)', fontproperties=fontproperties, fontsize=fontsize_middle)
    # ax.set_title(site_name, fontproperties='Palatino Linotype', fontsize=fontsize_big)
    # ax.legend(loc=4, fontsize=fontsize_legend)
    ax.xaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    ax.yaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    ax.set_xticklabels([])
    plt.tick_params(labelsize=fontsize_middle)
    bounds = [np.min(z), (np.min(z)+np.max(z))/2.0, np.max(z)]
    cbar_ax = fig.add_axes([0.96, 0.048, 0.015, 0.9])
    cbar = fig.colorbar(density, cax=cbar_ax, orientation="vertical", spacing='uniform',
                      ticks=bounds, label='Density')  # orientation = 'horizontal'
    cbar.ax.set_yticklabels(['L', 'M', 'H'])
    # cb.ax.set_title('Density', fontproperties=font_cb_title, )
    # cb.ax.title.set_position([1.0, 0.5])
    # cb.ax.yaxis.label.set_font_properties(font_cb_tick_label)
    statistics_list.append([site_name, 'reconstruct_2018', datatype] + statistics_value)
    #####-----------------------------------------------------------------------
    ####Shi Di Station;
    site_name = 'SDZ'
    read_filename = '%s_LST_AWS_%s.csv' % (datatype, site_name)
    validation_file = os.path.join(input_abspath, read_filename)
    df, image_text, statistics_name, statistics_value = interp_validation_statistics(validation_file)
    x = np.array(df.LST_Insitu)
    y = np.array(df.LST_Interp)
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    ax = fig.add_subplot(2, 5, 2, ylim=(xy_min, xy_max), xlim=(xy_min, xy_max))
    ax.scatter(x, y, label=point_lable, s=point_size, marker='o', c=z, linewidths=0.5, edgecolors='', cmap=plt.cm.jet)
    ax.plot(x, x * statistics_value[0] + statistics_value[1], 'r', label='Fitted Line', linewidth=linewidth)
    ax.plot([xy_min, xy_max], [xy_min, xy_max], c='k', linewidth=linewidth)
    ax.text(text_pos_x, text_pos_y, image_text, fontsize=fontsize_small, style='oblique', ha='left', va='bottom', wrap=True)
    ax.text(sitename_text_pos_x, sitename_text_pos_y, site_name, fontsize=fontsize_big, ha='left',
            va='bottom', wrap=True)
    #ax.set_xlabel(u'Insitu Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    #ax.set_ylabel(u'Reconstruct Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    # ax.set_title(site_name,fontproperties='Palatino Linotype',fontsize=fontsize_big)
    # ax.legend(loc=4, fontsize=fontsize_legend)
    ax.xaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    ax.yaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tick_params(labelsize=fontsize_middle)
    statistics_list.append([site_name, 'reconstruct_2018', datatype] + statistics_value)
    ####-----------------------------------------------------------------------
    ####Hua Zhai Zi Station;
    site_name = 'HZZ'
    read_filename = '%s_LST_AWS_%s.csv'%(datatype, site_name)
    validation_file = os.path.join(input_abspath, read_filename)
    df, image_text, statistics_name, statistics_value = interp_validation_statistics(validation_file)
    x = np.array(df.LST_Insitu)
    y = np.array(df.LST_Interp)
    
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax = fig.add_subplot(2, 5, 3, ylim=(xy_min, xy_max), xlim=(xy_min, xy_max))
    ax.scatter(x, y, label= point_lable, s=point_size, marker='o', c=z, linewidths=0.5, edgecolors='', cmap=plt.cm.jet)
    ax.plot(x, x * statistics_value[0] + statistics_value[1], 'r', label='Fitted Line', linewidth=linewidth)
    ax.plot([xy_min, xy_max], [xy_min, xy_max], c='k', linewidth=linewidth)
    ax.text(text_pos_x, text_pos_y, image_text, fontsize=fontsize_small, style='oblique', ha='left',
            va='bottom', wrap=True)
    ax.text(sitename_text_pos_x, sitename_text_pos_y, site_name, fontsize=fontsize_big, ha='left',
            va='bottom', wrap=True)
    #ax.set_xlabel(u'Insitu Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    #ax.set_ylabel(u'Reconstruct Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    # ax.set_title(site_name,fontproperties=fontproperties,fontsize=fontsize_big)
    # ax.legend(loc=4, fontsize=fontsize_legend)
    ax.xaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    ax.yaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tick_params(labelsize=fontsize_middle)
    statistics_list.append([site_name, 'reconstruct_2018', datatype] + statistics_value)
    ####-----------------------------------------------------------------------
    ####Ji Chang Huang Mo Station;
    site_name = 'JCHM'
    read_filename = '%s_LST_AWS_%s.csv' % (datatype, site_name)
    validation_file = os.path.join(input_abspath, read_filename)
    df, image_text, statistics_name, statistics_value = interp_validation_statistics(validation_file)
    x = np.array(df.LST_Insitu)
    y = np.array(df.LST_Interp)
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax = fig.add_subplot(2, 5, 4, ylim=(xy_min, xy_max), xlim=(xy_min, xy_max))
    ax.scatter(x, y, label=point_lable, s=point_size, marker='o', c=z, linewidths=0.5, edgecolors='', cmap=plt.cm.jet)
    ax.plot(x, x * statistics_value[0] + statistics_value[1], 'r', label='Fitted Line', linewidth=linewidth)
    ax.plot([xy_min, xy_max], [xy_min, xy_max], c='k', linewidth=linewidth)
    ax.text(text_pos_x, text_pos_y, image_text, fontsize=fontsize_small, style='oblique', ha='left',
            va='bottom', wrap=True)
    ax.text(sitename_text_pos_x, sitename_text_pos_y, site_name, fontsize=fontsize_big, ha='left',
            va='bottom', wrap=True)
    # ax.set_xlabel(u'Insitu Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    # ax.set_ylabel(u'Reconstruct Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    # ax.set_title(site_name,fontproperties=fontproperties,fontsize=fontsize_big)
    # ax.legend(loc=4, fontsize=fontsize_legend)
    ax.xaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    ax.yaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tick_params(labelsize=fontsize_middle)
    statistics_list.append([site_name, 'reconstruct_2018', datatype] + statistics_value)

    ####-----------------------------------------------------------------------
    #### Huang Mo Station;
    site_name = 'HMZ'
    read_filename = '%s_LST_AWS_%s.csv'%(datatype, site_name)
    validation_file = os.path.join(input_abspath, read_filename)
    df, image_text, statistics_name, statistics_value = interp_validation_statistics(validation_file)
    x = np.array(df.LST_Insitu)
    y = np.array(df.LST_Interp)
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    ax = fig.add_subplot(2, 5, 5, ylim=(xy_min, xy_max), xlim=(xy_min, xy_max))
    ax.scatter(x, y, label= point_lable, s=point_size, marker='o', c=z, linewidths=0.5, edgecolors='', cmap=plt.cm.jet)
    ax.plot(x, x * statistics_value[0] + statistics_value[1], 'r', label='Fitted Line', linewidth=linewidth)
    ax.plot([xy_min, xy_max], [xy_min, xy_max], c='k', linewidth=linewidth)
    ax.text(text_pos_x, text_pos_y, image_text, fontsize=fontsize_small, style='oblique', ha='left',
            va='bottom', wrap=True)
    ax.text(sitename_text_pos_x, sitename_text_pos_y, site_name, fontsize=fontsize_big, ha='left',
            va='bottom', wrap=True)
    # ax.set_xlabel(u'Insitu Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    # ax.set_ylabel(u'Reconstruct Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    # ax.set_title(site_name,fontproperties=fontproperties,fontsize=fontsize_big)
    ax.legend(loc=4, fontsize=fontsize_legend)
    ax.xaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    ax.yaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tick_params(labelsize=fontsize_middle)
    statistics_list.append([site_name, 'reconstruct_2018', datatype] + statistics_value)
    
    ####-----------------------------------------------------------------------
    ####YaKou Station;
    site_name = 'YKZ'
    read_filename = '%s_LST_AWS_%s.csv' % (datatype, site_name)
    validation_file = os.path.join(input_abspath, read_filename)
    df, image_text, statistics_name, statistics_value = interp_validation_statistics(validation_file)
    x = np.array(df.LST_Insitu)
    y = np.array(df.LST_Interp)
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    ax = fig.add_subplot(2, 5, 6, ylim=(xy_min, xy_max), xlim=(xy_min, xy_max))
    ax.scatter(x, y, label=point_lable, s=point_size, marker='o', c=z, linewidths=0.5, edgecolors='', cmap=plt.cm.jet)
    ax.plot(x, x * statistics_value[0] + statistics_value[1], 'r', label='Fitted Line', linewidth=linewidth)
    ax.plot([xy_min, xy_max], [xy_min, xy_max], c='k', linewidth=linewidth)
    ax.text(text_pos_x, text_pos_y, image_text, fontsize=fontsize_small, style='oblique', ha='left',
            va='bottom', wrap=True)
    ax.text(sitename_text_pos_x, sitename_text_pos_y, site_name, fontsize=fontsize_big, ha='left',
            va='bottom', wrap=True)
    # ax.set_xlabel(u'Insitu Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    ax.set_ylabel(ylabel, fontproperties=fontproperties, fontsize=fontsize_big, position=(0.0, 1))
    # ax.set_title(site_name,fontproperties=fontproperties,fontsize=fontsize_big)
    # ax.legend(loc=4, fontsize=fontsize_legend)
    ax.xaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    ax.yaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    plt.tick_params(labelsize=fontsize_middle)
    statistics_list.append([site_name, 'reconstruct_2018', datatype] + statistics_value)
    
    ####-----------------------------------------------------------------------
    ####A'Rou Chaoji Station;
    site_name = 'ARC'
    read_filename = '%s_LST_AWS_%s.csv' % (datatype, site_name)
    validation_file = os.path.join(input_abspath, read_filename)
    df, image_text, statistics_name, statistics_value = interp_validation_statistics(validation_file)
    x = np.array(df.LST_Insitu)
    y = np.array(df.LST_Interp)
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    ax = fig.add_subplot(2, 5, 7, ylim=(xy_min, xy_max), xlim=(xy_min, xy_max))
    ax.scatter(x, y, label=point_lable, s=point_size, marker='o', c=z, linewidths=0.5, edgecolors='', cmap=plt.cm.jet)
    ax.plot(x, x * statistics_value[0] + statistics_value[1], 'r', label='Fitted Line', linewidth=linewidth)
    ax.plot([xy_min, xy_max], [xy_min, xy_max], c='k', linewidth=linewidth)
    ax.text(text_pos_x, text_pos_y, image_text, fontsize=fontsize_small, style='oblique', ha='left',
            va='bottom', wrap=True)
    ax.text(sitename_text_pos_x, sitename_text_pos_y, site_name, fontsize=fontsize_big, ha='left',
            va='bottom', wrap=True)
    # ax.set_xlabel(u'Insitu Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    # ax.set_ylabel(u'Reconstruct Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    # ax.set_title(site_name,fontproperties=fontproperties,fontsize=fontsize_big)
    # ax.legend(loc=4, fontsize=fontsize_legend)
    ax.xaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    ax.yaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    # ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tick_params(labelsize=fontsize_middle)
    statistics_list.append([site_name, 'reconstruct_2018', datatype] + statistics_value)
    
    ####-----------------------------------------------------------------------
    ####Da Sha Long Station;
    site_name = 'DSL'
    read_filename = '%s_LST_AWS_%s.csv' % (datatype, site_name)
    validation_file = os.path.join(input_abspath, read_filename)
    df, image_text, statistics_name, statistics_value = interp_validation_statistics(validation_file)
    x = np.array(df.LST_Insitu)
    y = np.array(df.LST_Interp)
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax = fig.add_subplot(2, 5, 8, ylim=(xy_min, xy_max), xlim=(xy_min, xy_max))
    ax.scatter(x, y, label='Recon-Corr', s=point_size, marker='o', c=z,linewidths=0.5, edgecolors='', cmap=plt.cm.jet)
    ax.plot(x, x * statistics_value[0] + statistics_value[1], 'r', label='Fitted Line', linewidth=linewidth)
    ax.plot([xy_min, xy_max], [xy_min, xy_max], c='k', linewidth=linewidth)
    ax.text(text_pos_x, text_pos_y, image_text, fontsize=fontsize_small, style='oblique', ha='left',
            va='bottom', wrap=True)
    ax.text(sitename_text_pos_x, sitename_text_pos_y, site_name, fontsize=fontsize_big, ha='left',
            va='bottom', wrap=True)
    ax.set_xlabel(xlabel, fontproperties=fontproperties, fontsize=fontsize_big)
    # ax.set_ylabel(u'Reconstruct Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    # ax.set_title(site_name,fontproperties=fontproperties,fontsize=fontsize_big)
    # ax.legend(loc=4, fontsize=fontsize_legend)
    ax.xaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    ax.yaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    # ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tick_params(labelsize=fontsize_middle)
    statistics_list.append([site_name, 'reconstruct_2018', datatype] + statistics_value)
    # ###-----------------------------------------------------------------------
    # ###Hun He Lin Station;
    site_name = 'HHL'
    read_filename = '%s_LST_AWS_%s.csv' % (datatype, site_name)
    validation_file = os.path.join(input_abspath, read_filename)
    df, image_text, statistics_name, statistics_value = interp_validation_statistics(validation_file)
    x = np.array(df.LST_Insitu)
    y = np.array(df.LST_Interp)
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax = fig.add_subplot(2, 5, 9, ylim=(xy_min, xy_max), xlim=(xy_min, xy_max))
    ax.scatter(x, y, label=point_lable, s=point_size, marker='o', c=z, linewidths=0.5, edgecolors='', cmap=plt.cm.jet)
    ax.plot(x, x * statistics_value[0] + statistics_value[1], 'r', label='Fitted Line', linewidth=linewidth)
    ax.plot([xy_min, xy_max], [xy_min, xy_max], c='k', linewidth=linewidth)
    ax.text(text_pos_x, text_pos_y, image_text, fontsize=fontsize_small, style='oblique', ha='left',
            va='bottom', wrap=True)
    ax.text(sitename_text_pos_x, sitename_text_pos_y, site_name, fontsize=fontsize_big, ha='left',
            va='bottom', wrap=True)
    # ax.set_xlabel(u'Insitu Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    # ax.set_ylabel(u'Reconstruct Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    # ax.set_title(site_name,fontproperties=fontproperties,fontsize=fontsize_big)
    # ax.legend(loc=4, fontsize=fontsize_legend)
    ax.xaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    ax.yaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    # ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tick_params(labelsize=fontsize_middle)
    statistics_list.append([site_name, 'reconstruct_2018', datatype] + statistics_value)
    ####-----------------------------------------------------------------------
    ####SiDaoQiao Station;
    site_name = 'SDQ'
    read_filename = '%s_LST_AWS_%s.csv'%(datatype, site_name)
    validation_file = os.path.join(input_abspath, read_filename)
    df, image_text, statistics_name, statistics_value = interp_validation_statistics(validation_file)
    x = np.array(df.LST_Insitu)
    y = np.array(df.LST_Interp)
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    ax = fig.add_subplot(2, 5, 10, ylim=(xy_min, xy_max), xlim=(xy_min, xy_max))
    ax.scatter(x, y, label=point_lable, s=point_size, marker='o', c=z, linewidths=0.5, edgecolors='', cmap=plt.cm.jet)
    ax.plot(x, x * statistics_value[0] + statistics_value[1], 'r', label='Fitted Line', linewidth=linewidth)
    ax.plot([xy_min, xy_max], [xy_min, xy_max], c='k', linewidth=linewidth)
    ax.text(text_pos_x, text_pos_y, image_text, fontsize=fontsize_small, style='oblique', ha='left',
            va='bottom', wrap=True)
    ax.text(sitename_text_pos_x, sitename_text_pos_y, site_name, fontsize=fontsize_big, ha='left',
            va='bottom', wrap=True)
    # ax.set_xlabel(u'Insitu Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    # ax.set_ylabel(u'Reconstruct Temperature(K)',fontproperties=fontproperties,fontsize=fontsize_middle)
    # ax.set_title(site_name,fontproperties=fontproperties,fontsize=fontsize_big)
    ax.legend(loc=4, fontsize=fontsize_legend)
    ax.xaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    ax.yaxis.set_major_locator(mplt.ticker.FixedLocator(majors))
    # ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tick_params(labelsize=fontsize_middle)
    statistics_list.append([site_name, 'reconstruct_2018', datatype] + statistics_value)

    # ###----------------------------- Save Figure -----------------------------
    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95, wspace=0, hspace=0)
    plt.savefig(output_figure_abspath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close() 
          
#     ####------------- create statistics and export to excele file -------------
#     statistics_file = 'MODIS_Reconstruct_statistics_2013-2018.%s_Corrected-Scatter.Density.csv'%(datatype)
#     statistics_file = os.path.join(target_abspath,statistics_file)
#     with open(statistics_file,'wb') as csvfile:
#         spamwriter = csv.writer(csvfile, dialect = 'excel')
#         spamwriter.writerow(['sitename','datakind','datatype']+statistics_name)
#         spamwriter.writerows(statistics_list)
#
#     statistics_file = 'MODIS_Reconstruct_statistics_2013-2018.%s_UnCorrected-Scatter.Density.csv'%(datatype)
#     statistics_file = os.path.join(target_abspath,statistics_file)
#     with open(statistics_file,'wb') as csvfile:
#         spamwriter = csv.writer(csvfile, dialect = 'excel')
#         spamwriter.writerow(['sitename','datakind','datatype']+statistics_name)
#         spamwriter.writerows(statistics_list_Uncorrected)
# #####--------------------------    Figure Plot End   --------------------------
# #####==========================================================================

