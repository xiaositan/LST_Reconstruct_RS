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
from compiler.ast import flatten
def interp_validation_statistics_Month(validation_file,month):
    df = pd.read_csv(validation_file,header=0,parse_dates={'DateTime':[0]},infer_datetime_format=True)
    df = df[~df['LST_Insitu'].isnull()]
    df_interp = df[df.LST_Ori == 0.0]
    df_interp = df_interp[df_interp.LST_Interp != 0]
    df_interp['Month'] = pd.to_datetime(df_interp['DateTime']).map(lambda x:x.month)
    df_interp = df_interp[df_interp['Month']==month]
    # Interpolation with In-Situ data
    x = np.array(df_interp.LST_Insitu)
    y = np.array(df_interp.LST_Interp)
    slope_interp, intercept_interp, r_value_interp, p_value_interp, stderr_gradient = scipy.stats.linregress(x, y)
    rmse_interp = scipy.sqrt(scipy.mean((y-x)**2))
    bias_interp = scipy.mean(y-x)
    mae_interp  = scipy.mean(abs(y-x))
    std_err_interp = np.std(y-x)
    data_number = len(df_interp.LST_Insitu)
#    image_text = 'Eq.: y=%6.2fx %+6.2f\nR-Square=%5.2f\nRMSE=%5.2f\nBIAS=%5.2f\nMAE=%5.2f\nN=%d'\
#                %(slope_interp,intercept_interp,r_value_interp**2,rmse_interp,bias_interp,mae_interp,data_number)
    # image_text = 'R-Square=%5.2f\nRMSE=%5.2f\nBIAS=%5.2f\nMAE=%5.2f\nN=%d'\
    #             %(r_value_interp**2,rmse_interp,bias_interp,mae_interp,data_number)
    image_text = 'y=%6.2fx%+6.2f\nR$^2$=%5.2f\nN=%d'\
                %(slope_interp,intercept_interp,r_value_interp**2,data_number)    
    statistics_name = ['Slope','Intercept', 'R-Square','p_value','std_err','RMSE', 'BIAS', 'MAE', 'DataNumber']
    statistics_value = [slope_interp,intercept_interp,r_value_interp**2, p_value_interp, std_err_interp,rmse_interp,bias_interp,mae_interp,data_number]
    return df_interp,image_text,statistics_name,statistics_value

#=============================================================================
# Fig 1. Daman, Arou Super Station,Gebi, Huazhaizi, Shenshawo, Shidi, Jichang Huangmo
### ============FOR Each MONTH IN YEAY==========================
datatypes = ['MOD11A1.Day','MOD11A1.Night']
datadir = u'../../data/processed_data/result/Tck_reconstructed.2013-2018/'
output_path = u'../../data/processed_data/result/Tck_reconstructed.2013-2018.month/'
fontproperties = 'Palatino Linotype'
fontsize_big = 22
fontsize_middle = 18
fontsize_small = 16
fontsize_legend = 14
for datatype in datatypes:
    df_site_R2 = pd.DataFrame()
    df_site_BIAS = pd.DataFrame()
    df_site_RMSE = pd.DataFrame()
    month_list = list([])
    DMZ_r2 = list([])
    SDZ_r2 = list([])
    HZZ_r2 = list([])
    SDQ_r2 = list([])
    YKZ_r2 = list([])
    JCHM_r2 = list([])
    ARC_r2 = list([])
    DSL_r2 = list([])
    HHL_r2 = list([])
    HMZ_r2 = list([])
    
    DMZ_BIAS = list([])
    SDZ_BIAS = list([])
    HZZ_BIAS = list([])
    SDQ_BIAS = list([])
    YKZ_BIAS = list([])
    JCHM_BIAS = list([])
    ARC_BIAS = list([])
    DSL_BIAS = list([])
    HHL_BIAS = list([])
    HMZ_BIAS = list([]) 

    DMZ_RMSE = list([])
    SDZ_RMSE = list([])
    HZZ_RMSE = list([])
    SDQ_RMSE = list([])
    YKZ_RMSE = list([])
    JCHM_RMSE = list([])
    ARC_RMSE = list([])
    DSL_RMSE = list([])
    HHL_RMSE = list([])
    HMZ_RMSE = list([]) 
    
    for month in range(1, 13):
        month_list.append(month)
        statistics_list = []
        indir_abspath = os.path.abspath(datadir)
        target_abspath = os.path.abspath(output_path)
        out_filename = 'MODIS_Reconstruct_LST_Validation_2013-2018_%d.%s.jpg'%(month,datatype)
        outfile_abspath = os.path.join(target_abspath,out_filename)
        titletext = '2013-2018-month-%d-%s'%(month,datatype)
        fig = plt.figure(figsize=[27,13.0])
        plt.suptitle(titletext, fontsize=20)
        ####-----------------------------------------------------------------------
        ####Dman Station
        site_name = 'DMZ'
        read_filename = '%s_LST_AWS_%s.csv' % (datatype, site_name)
        validation_file = os.path.join(indir_abspath,read_filename)
        df_modis,image_text_modis,statistics_name_modis,statistics_value_modis = interp_validation_statistics_Month(validation_file,month)
        x = np.array(df_modis.LST_Insitu)
        y = np.array(df_modis.LST_Interp)
        ax = fig.add_subplot(251,ylim=(240,340),xlim=(240,340))  
        ax.plot([240,340],[240,340],c='k')
        ax.scatter(df_modis.LST_Insitu,df_modis.LST_Interp,label='Reconstruct',s=15,c='b',linewidths=0.5,edgecolors='w')
        ax.plot(x,x * statistics_value_modis[0] +statistics_value_modis[1],'r',label='Rec_Fit')
        ax.text(242,315,image_text_modis, fontsize=12, style='oblique', ha='left',va='bottom', wrap=True)
        #ax.set_xlabel(u'Insitu Temperature (K)',fontproperties=fontproperties,fontsize=18)
        ax.set_ylabel(u'Reconstruct Temperature (K)',fontproperties=fontproperties,fontsize=18)
        ax.set_title(site_name,fontproperties=fontproperties,fontsize=20)
        ax.legend(loc=4)
        ax.xaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        ax.yaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        plt.tick_params(labelsize=16)
        statistics_list.append([site_name,'reconstruct_month-%d' %month, datatype]+statistics_value_modis)
        DMZ_r2.append([statistics_value_modis[5]])
        DMZ_BIAS.append([statistics_value_modis[6]])
        ####-----------------------------------------------------------------------
        ####Dman Station
        site_name = 'SDZ'
        read_filename = '%s_LST_AWS_%s.csv'%(datatype,site_name)
        validation_file = os.path.join(indir_abspath,read_filename)
        df_modis,image_text_modis,statistics_name_modis,statistics_value_modis = interp_validation_statistics_Month(validation_file,month)
        x = np.array(df_modis.LST_Insitu)
        y = np.array(df_modis.LST_Interp)
        ax = fig.add_subplot(252,ylim=(240,340),xlim=(240,340))  
        ax.plot([240,340],[240,340],c='k')
        ax.scatter(df_modis.LST_Insitu,df_modis.LST_Interp,label='Reconstruct', s=15,c='b',linewidths=0.5,edgecolors='w')
        ax.plot(x,x * statistics_value_modis[0] +statistics_value_modis[1],'r', label='Rec_Fit')
        ax.text(242,315,image_text_modis, fontsize=12, style='oblique', ha='left',va='bottom', wrap=True)
        #ax.set_xlabel(u'Insitu Temperature (K)',fontproperties=fontproperties,fontsize=18)
        #ax.set_ylabel(u'Reconstruct Temperature (K)',fontproperties=fontproperties,fontsize=18)
        ax.set_title(site_name,fontproperties=fontproperties,fontsize=20)
        ax.legend(loc=4)
        ax.xaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        ax.yaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        plt.tick_params(labelsize=16)
        statistics_list.append([site_name,'reconstruct_month-%s' %datatype]+statistics_value_modis)
        SDZ_r2.append([statistics_value_modis[5]])
        SDZ_BIAS.append([statistics_value_modis[6]])
        ####-----------------------------------------------------------------------
        ####Dman Station
        site_name = 'HZZ'
        read_filename = '%s_LST_AWS_%s.csv'%(datatype,site_name)
        validation_file = os.path.join(indir_abspath,read_filename)
        df_modis,image_text_modis,statistics_name_modis,statistics_value_modis = interp_validation_statistics_Month(validation_file,month)
        x = np.array(df_modis.LST_Insitu)
        y = np.array(df_modis.LST_Interp)
        ax = fig.add_subplot(253,ylim=(240,340),xlim=(240,340))  
        ax.plot([240,340],[240,340],c='k')
        ax.scatter(df_modis.LST_Insitu,df_modis.LST_Interp,label='Reconstruct',s=15,c='b',linewidths=0.5,edgecolors='w')
        ax.plot(x,x * statistics_value_modis[0] +statistics_value_modis[1],'r',label='Rec_Fit')
        ax.text(242,315,image_text_modis, fontsize=12, style='oblique', ha='left',va='bottom', wrap=True)
        #ax.set_xlabel(u'Insitu Temperature (K)',fontproperties=fontproperties,fontsize=18)
        #ax.set_ylabel(u'Reconstruct Temperature (K)',fontproperties=fontproperties,fontsize=18)
        ax.set_title(site_name,fontproperties=fontproperties,fontsize=20)
        ax.legend(loc=4)
        ax.xaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        ax.yaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        plt.tick_params(labelsize=16)
        statistics_list.append([site_name,'reconstruct_month-%d' %month, datatype]+statistics_value_modis)
        HZZ_r2.append([statistics_value_modis[5]])
        HZZ_BIAS.append([statistics_value_modis[6]])
        ####-----------------------------------------------------------------------
        ####----Ji Chang Huang Mo------
        site_name = 'JCHM'
        read_filename = '%s_LST_AWS_%s.csv'%(datatype,site_name)
        validation_file = os.path.join(indir_abspath,read_filename)
        df_modis,image_text_modis,statistics_name_modis,statistics_value_modis = interp_validation_statistics_Month(validation_file,month)
        x = np.array(df_modis.LST_Insitu)
        y = np.array(df_modis.LST_Interp)
        ax = fig.add_subplot(254,ylim=(240,340),xlim=(240,340))  
        ax.plot([240,340],[240,340],c='k')
        ax.scatter(df_modis.LST_Insitu,df_modis.LST_Interp,label='Reconstruct',s=15,c='b',linewidths=0.5,edgecolors='w')
        ax.plot(x,x * statistics_value_modis[0] +statistics_value_modis[1],'r',label='Rec_Fit')
        ax.text(242,315,image_text_modis, fontsize=12, style='oblique', ha='left',va='bottom', wrap=True)
        #ax.set_xlabel(u'Insitu Temperature (K)',fontproperties=fontproperties,fontsize=18)
        #ax.set_ylabel(u'Reconstruct Temperature (K)',fontproperties=fontproperties,fontsize=18)
        ax.set_title(site_name,fontproperties=fontproperties,fontsize=20)
        ax.legend(loc=4)
        ax.xaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        ax.yaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        plt.tick_params(labelsize=16)
        statistics_list.append([site_name,'reconstruct_month-%d' %month, datatype]+statistics_value_modis)
        JCHM_r2.append([statistics_value_modis[5]])
        JCHM_BIAS.append([statistics_value_modis[6]])
        ####-----------------------------------------------------------------------
        ####Dman Station
        site_name = 'HMZ'
        read_filename = '%s_LST_AWS_%s.csv'%(datatype,site_name)
        validation_file = os.path.join(indir_abspath,read_filename)
        df_modis,image_text_modis,statistics_name_modis,statistics_value_modis = interp_validation_statistics_Month(validation_file,month)
        x = np.array(df_modis.LST_Insitu)
        y = np.array(df_modis.LST_Interp)
        ax = fig.add_subplot(2,5,5,ylim=(240,340),xlim=(240,340))  
        ax.plot([240,340],[240,340],c='k')
        ax.scatter(df_modis.LST_Insitu,df_modis.LST_Interp,label='Reconstruct',s=15,c='b',linewidths=0.5,edgecolors='w')
        ax.plot(x,x * statistics_value_modis[0] +statistics_value_modis[1],'r',label='Rec_Fit')
        ax.text(242,315,image_text_modis, fontsize=12, style='oblique', ha='left',va='bottom', wrap=True)
        #ax.set_xlabel(u'Insitu Temperature (K)',fontproperties=fontproperties,fontsize=18)
        #ax.set_ylabel(u'Reconstruct Temperature (K)',fontproperties=fontproperties,fontsize=18)
        ax.set_title(site_name,fontproperties=fontproperties,fontsize=20)
        ax.legend(loc=4)
        ax.xaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        ax.yaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        plt.tick_params(labelsize=16)
        statistics_list.append([site_name,'reconstruct_month-%d' %month, datatype]+statistics_value_modis)
        HMZ_r2.append([statistics_value_modis[5]])
        HMZ_BIAS.append([statistics_value_modis[6]])
        
        ####-----------------------------------------------------------------------
        site_name = 'YKZ'
        read_filename = '%s_LST_AWS_%s.csv'%(datatype,site_name)
        validation_file = os.path.join(indir_abspath,read_filename)
        df_modis,image_text_modis,statistics_name_modis,statistics_value_modis = interp_validation_statistics_Month(validation_file,month)
        x = np.array(df_modis.LST_Insitu)
        y = np.array(df_modis.LST_Interp)
        ax = fig.add_subplot(256,ylim=(240,340),xlim=(240,340))  
        ax.plot([240,340],[240,340],c='k')
        ax.scatter(df_modis.LST_Insitu,df_modis.LST_Interp,label='Reconstruct',s=15,c='b',linewidths=0.5,edgecolors='w')
        ax.plot(x,x * statistics_value_modis[0] +statistics_value_modis[1],'r',label='Rec_Fit')
        ax.text(242,315,image_text_modis, fontsize=12, style='oblique', ha='left',va='bottom', wrap=True)
        ax.set_xlabel(u'Insitu Temperature (K)',fontproperties=fontproperties,fontsize=18)
        ax.set_ylabel(u'Reconstructed Temperature (K)',fontproperties=fontproperties,fontsize=18)
        ax.set_title(site_name,fontproperties=fontproperties,fontsize=20)
        ax.legend(loc=4)
        ax.xaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        ax.yaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        plt.tick_params(labelsize=16)
        statistics_list.append([site_name,'reconstruct_month-%d' %month, datatype]+statistics_value_modis)
        YKZ_r2.append([statistics_value_modis[5]])  
        YKZ_BIAS.append([statistics_value_modis[6]])
        ####-----------------------------------------------------------------------
        ####Dman Station
        site_name = 'ARC'
        read_filename = '%s_LST_AWS_%s.csv'%(datatype,site_name)
        validation_file = os.path.join(indir_abspath,read_filename)
        df_modis,image_text_modis,statistics_name_modis,statistics_value_modis = interp_validation_statistics_Month(validation_file,month)
        x = np.array(df_modis.LST_Insitu)
        y = np.array(df_modis.LST_Interp)
        ax = fig.add_subplot(2,5,7,ylim=(240,340),xlim=(240,340))  
        ax.plot([240,340],[240,340],c='k')
        ax.scatter(df_modis.LST_Insitu,df_modis.LST_Interp,label='Reconstruct',s=15,c='b',linewidths=0.5,edgecolors='w')
        ax.plot(x,x * statistics_value_modis[0] +statistics_value_modis[1],'r',label='Rec_Fit')
        ax.text(242,315,image_text_modis, fontsize=12, style='oblique', ha='left',va='bottom', wrap=True)
        ax.set_xlabel(u'Insitu Temperature (K)',fontproperties=fontproperties,fontsize=18)
        #ax.set_ylabel(u'Reconstruct Temperature (K)',fontproperties=fontproperties,fontsize=18)
        ax.set_title(site_name,fontproperties=fontproperties,fontsize=20)
        ax.legend(loc=4)
        ax.xaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        ax.yaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        plt.tick_params(labelsize=16)
        statistics_list.append([site_name,'reconstruct_month-%d' %month, datatype]+statistics_value_modis) 
        ARC_r2.append([statistics_value_modis[5]])
        ARC_BIAS.append([statistics_value_modis[6]])
        ####-----------------------------------------------------------------------
        ####Dman Station
        site_name = 'DSL'
        read_filename = '%s_LST_AWS_%s.csv'%(datatype,site_name)
        validation_file = os.path.join(indir_abspath,read_filename)
        df_modis,image_text_modis,statistics_name_modis,statistics_value_modis = interp_validation_statistics_Month(validation_file,month)
        x = np.array(df_modis.LST_Insitu)
        y = np.array(df_modis.LST_Interp)
        ax = fig.add_subplot(258,ylim=(240,340),xlim=(240,340))  
        ax.plot([240,340],[240,340],c='k')
        ax.scatter(df_modis.LST_Insitu,df_modis.LST_Interp,label='Reconstruct',s=15,c='b',linewidths=0.5,edgecolors='w')
        ax.plot(x,x * statistics_value_modis[0] +statistics_value_modis[1],'r',label='Rec_Fit')
        ax.text(242,315,image_text_modis, fontsize=12, style='oblique', ha='left',va='bottom', wrap=True)
        ax.set_xlabel(u'Insitu Temperature (K)',fontproperties=fontproperties,fontsize=18)
        #ax.set_ylabel(u'Reconstruct Temperature (K)',fontproperties=fontproperties,fontsize=18)
        ax.set_title(site_name,fontproperties=fontproperties,fontsize=20)
        ax.legend(loc=4)
        ax.xaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        ax.yaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        plt.tick_params(labelsize=16)
        statistics_list.append([site_name,'reconstruct_month-%d' %month, datatype]+statistics_value_modis)
        DSL_r2.append([statistics_value_modis[5]])
        DSL_BIAS.append([statistics_value_modis[6]])
        ####-----------------------------------------------------------------------
        ####Dman Station
        site_name = 'HHL'
        read_filename = '%s_LST_AWS_%s.csv'%(datatype,site_name)
        validation_file = os.path.join(indir_abspath,read_filename)
        df_modis,image_text_modis,statistics_name_modis,statistics_value_modis = interp_validation_statistics_Month(validation_file,month)
        x = np.array(df_modis.LST_Insitu)
        y = np.array(df_modis.LST_Interp)
        ax = fig.add_subplot(2,5,9,ylim=(240,340),xlim=(240,340))  
        ax.plot([240,340],[240,340],c='k')
        ax.scatter(df_modis.LST_Insitu,df_modis.LST_Interp,label='Reconstruct',s=15,c='b',linewidths=0.5,edgecolors='w')
        ax.plot(x,x * statistics_value_modis[0] +statistics_value_modis[1],'r',label='Rec_Fit')
        ax.text(242,315,image_text_modis, fontsize=12, style='oblique', ha='left',va='bottom', wrap=True)
        ax.set_xlabel(u'Insitu Temperature (K)',fontproperties=fontproperties,fontsize=18)
        #ax.set_ylabel(u'Reconstruct Temperature (K)',fontproperties=fontproperties,fontsize=18)
        ax.set_title(site_name,fontproperties=fontproperties,fontsize=20)
        ax.legend(loc=4)
        ax.xaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        ax.yaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        plt.tick_params(labelsize=16)
        statistics_list.append([site_name,'reconstruct_month-%d' %month, datatype]+statistics_value_modis) 
        HHL_r2.append([statistics_value_modis[5]])
        HHL_BIAS.append([statistics_value_modis[6]])
        ####-----------------------------------------------------------------------
        ####-----------------------------------------------------------------------
        ####Dman Station
        site_name = 'SDQ'
        read_filename = '%s_LST_AWS_%s.csv'%(datatype,site_name)
        validation_file = os.path.join(indir_abspath,read_filename)
        df_modis,image_text_modis,statistics_name_modis,statistics_value_modis = interp_validation_statistics_Month(validation_file,month)
        x = np.array(df_modis.LST_Insitu)
        y = np.array(df_modis.LST_Interp)
        ax = fig.add_subplot(2,5,10,ylim=(240,340),xlim=(240,340))  
        ax.plot([240,340],[240,340],c='k')
        ax.scatter(df_modis.LST_Insitu,df_modis.LST_Interp,label='Reconstruct',s=15,c='b',linewidths=0.5,edgecolors='w')
        ax.plot(x,x * statistics_value_modis[0] +statistics_value_modis[1],'r',label='Rec_Fit')
        ax.text(242,315,image_text_modis, fontsize=12, style='oblique', ha='left',va='bottom', wrap=True)
        #ax.set_xlabel(u'Insitu Temperature (K)',fontproperties=fontproperties,fontsize=18)
        #ax.set_ylabel(u'Reconstruct Temperature (K)',fontproperties=fontproperties,fontsize=18)
        ax.set_title(site_name,fontproperties=fontproperties,fontsize=20)
        ax.legend(loc=4)
        ax.xaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        ax.yaxis.set_major_locator(mplt.ticker.MultipleLocator(20))
        plt.tick_params(labelsize=16)
        statistics_list.append([site_name,'reconstruct_month-%d' %month, datatype]+statistics_value_modis)
        SDQ_r2.append([statistics_value_modis[5]])
        SDQ_BIAS.append([statistics_value_modis[6]])
        # ##===== create statistics===========================
        # plt.savefig(outfile_abspath,dpi=400,bbox_inches='tight')
        # plt.close()
        #
        # statistics_file = 'MODIS_Reconstruct_statistics_month.%d.%s.csv'%(month,datatype)
        # statistics_file = os.path.join(target_abspath,statistics_file)
        # with open(statistics_file,'wb') as csvfile:
        #     spamwriter = csv.writer(csvfile, dialect = 'excel')
        #     spamwriter.writerow(['sitename','datakind','datatype']+statistics_name_modis)
        #     spamwriter.writerows(statistics_list)
    ####=======================================================================
    df_site_BIAS['Month'] = month_list
    df_site_BIAS['DMZ'] = flatten(DMZ_BIAS)
    df_site_BIAS['SDZ'] = flatten(SDZ_BIAS)
    df_site_BIAS['HZZ'] = flatten(HZZ_BIAS)
    df_site_BIAS['SDQ'] = flatten(SDQ_BIAS)
    df_site_BIAS['YKZ'] = flatten(YKZ_BIAS)
    df_site_BIAS['JCHM'] = flatten(JCHM_BIAS)
    df_site_BIAS['ARC'] = flatten(ARC_BIAS)
    df_site_BIAS['DSL'] = flatten(DSL_BIAS)
    df_site_BIAS['HHL'] = flatten(HHL_BIAS)
    df_site_BIAS['HMZ'] = flatten(HMZ_BIAS)
    BIAS_statics_file = os.path.join(target_abspath,'BIAS_Reconstructed_statistics_%s.csv'% (datatype))
    df_site_BIAS.to_csv(BIAS_statics_file)
    font = {'family' : 'Palatino Linotype',
        'weight' : 'normal',
        'size'   : 18}    
    plt.rc('font', **font)  

    if datatype =='MOD11A1.Day':
        y_min = -3
        y_max = 13
        text_pos_x = 242
        text_pos_y = 320
    else:
        y_min = -10
        y_max = 2
        text_pos_x = 232
        text_pos_y = 295
    ####--------        BIAS Plot 4 small figures      --------
    if datatype =='MOD11A1.Day':
        text_pos_x = 0.5
        text_pos_y = 11.6
    else:
        text_pos_x = 0.5
        text_pos_y = 1.0
    fig = plt.figure(figsize=[19,5])
    ax = fig.add_subplot(141,ylim=(y_min,y_max))
    plt.plot(df_site_BIAS['Month'],df_site_BIAS['ARC'],label='ARC') 
    plt.plot(df_site_BIAS['Month'],df_site_BIAS['SDZ'],label='SDZ')
    plt.plot(df_site_BIAS['Month'],df_site_BIAS['DMZ'],label='DMZ')
       
    plt.legend(loc='lower center',fontsize=11,ncol=5)
    ax.set_xlim(0,13) 
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax.set_xlabel('Month',fontsize=18)
    ax.set_ylabel('Mean Bias Error (K)',color='black',fontsize=18)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    #ax.set_title('Arou Soil pf 2016')
    if datatype =='MOD11A1.Day':
        image_text = '(a)'
    else:
        image_text = '(e)'
    ax.text(text_pos_x,text_pos_y,image_text, fontsize=fontsize_middle, ha='left',va='bottom', wrap=True)
    for label in ax.get_xticklabels():  
        label.set_color('black') 
        label.set_fontsize(18) 
    for label in ax.get_yticklabels(): 
        label.set_color('black')
        label.set_fontsize(18) 
        
    ax = fig.add_subplot(142,ylim=(y_min,y_max))
    plt.plot(df_site_BIAS['Month'],df_site_BIAS['DSL'],label='DSL')
    plt.plot(df_site_BIAS['Month'],df_site_BIAS['YKZ'],label='YKZ')
    
    plt.legend(loc='lower center',fontsize=11,ncol=5)
    ax.set_xlim(0,13)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax.set_xlabel('Month',fontsize=18)
    #ax.set_ylabel('Mean Bias Error (K)',color='black',fontsize=18)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    #ax.set_title('Arou Soil pf 2016')
    if datatype =='MOD11A1.Day':
        image_text = '(b)'
    else:
        image_text = '(f)'
    ax.text(text_pos_x,text_pos_y,image_text, fontsize=fontsize_middle, ha='left',va='bottom', wrap=True)
    for label in ax.get_xticklabels():  
        label.set_color('black') 
        label.set_fontsize(18) 
    for label in ax.get_yticklabels(): 
        label.set_color('black')
        label.set_fontsize(18) 
        
    ax = fig.add_subplot(143,ylim=(y_min,y_max))   
    plt.plot(df_site_BIAS['Month'],df_site_BIAS['SDQ'],label='SDQ')
    plt.plot(df_site_BIAS['Month'],df_site_BIAS['HHL'],label='HHL')
    
    plt.legend(loc='lower center',fontsize=11,ncol=5)
    ax.set_xlim(0,13)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax.set_xlabel('Month',fontsize=18)
    #ax.set_ylabel('Mean Bias Error (K)',color='black',fontsize=18)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    if datatype =='MOD11A1.Day':
        image_text = '(c)'
    else:
        image_text = '(g)'
    ax.text(text_pos_x,text_pos_y,image_text, fontsize=fontsize_middle, ha='left',va='bottom', wrap=True)#, style='oblique'
    for label in ax.get_xticklabels():  
        label.set_color('black') 
        label.set_fontsize(18) 
    for label in ax.get_yticklabels(): 
        label.set_color('black')
        label.set_fontsize(18) 

    ax = fig.add_subplot(144,ylim=(y_min,y_max))
    plt.plot(df_site_BIAS['Month'],df_site_BIAS['HZZ'],label='HZZ')
    plt.plot(df_site_BIAS['Month'],df_site_BIAS['JCHM'],label='JCHM')
    plt.plot(df_site_BIAS['Month'],df_site_BIAS['HMZ'],label='HMZ')   
    
    plt.legend(loc='lower center',fontsize=11,ncol=5)
    ax.set_xlim(0, 13)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax.set_xlabel('Month',fontsize=18)
    #ax.set_ylabel('Mean Bias Error (K)',color='black',fontsize=18)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    #ax.set_title('Arou Soil pf 2016')
    if datatype =='MOD11A1.Day':
        image_text = '(d)'
    else:
        image_text = '(h)'
    ax.text(text_pos_x,text_pos_y,image_text, fontsize=fontsize_middle, ha='left',va='bottom', wrap=True)
    for label in ax.get_xticklabels():  
        label.set_color('black') 
        label.set_fontsize(18) 
    for label in ax.get_yticklabels(): 
        label.set_color('black')
        label.set_fontsize(18) 
        
    output_fig = u'Figure 6. Daytime and nighttime monthly mean bias errors.%s.tif' % datatype
    output_fig = os.path.join('../../output_figures', output_fig)
    fig.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.show()