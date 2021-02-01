from __future__ import (absolute_import, division, print_function)

"""
example showing how to plot data from a land surface temperature file 
and an ESRI shape file using  gdal (http://pypi.python.org/pypi/GDAL).
"""
import os
import numpy as np
from osgeo import gdal, ogr
import shapefile
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.basemap import Basemap, cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
import pandas as pd

def getClpPathPatch_FromShapeFile(shpfilename, record_no, record_value, axis, basemap):
    # obtain one shp geometry from shpFile that has many polygons by the data value
    # from wgs84 coordinate latitude and longitude to m.projection
    # obtain the clip PathPatch
    try:
        sf = shapefile.Reader(shpfilename)
        for shape_rec in sf.shapeRecords():
            if shape_rec.record[record_no] == record_value:
                vertices = []
                codes = []
                pts = shape_rec.shape.points
                prt = list(shape_rec.shape.parts) + [len(pts)]
                for i in range(len(prt) - 1):
                    for j in range(prt[i], prt[i + 1]):
                        # obtain the reproject coordinate
                        vertices.append(basemap(pts[j][0], pts[j][1]))
                    codes += [Path.MOVETO]
                    codes += [Path.LINETO] * (prt[i + 1] - prt[i] - 2)
                    codes += [Path.CLOSEPOLY]
                path = Path(vertices, codes)
        # Clip
        if path is not None:
            patch = PathPatch(path, transform=axis.transData)
            print('patch', patch)
        else:
            patch = None
            print('There is No PathPatch, Maybe No this shape record')
        return patch
    except Exception as err:
        print('There is a error in this part code! Please recheck it!!')
        return err


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Palatino Linotype"
    font0 = FontProperties()

    font_sub_title = font0.copy()
    font_sub_title.set_size('10')
    font_sub_title.set_weight('bold')

    font_tick_label = font0.copy()
    font_tick_label.set_size('10')
    #font_tick_label.set_weight('bold')

    font_cb_title = font0.copy()
    font_cb_title.set_size('10')
    font_cb_title.set_weight('bold')

    font_cb_tick_label = font0.copy()
    font_cb_tick_label.set_size('10')
    font_cb_tick_label.set_weight('bold')

    x_sub_title = 97.0  # sub title position x
    y_sub_title = 42.75  # sub title position y

    llcrnrlon = 96.5     # low left corner longitude coordinate
    urcrnrlon = 102.5  # up right corner longitude coordinate
    llcrnrlat = 37.6   # low left corner latitude coordinate
    urcrnrlat = 43.2   # up right corner latitude coordinate

    # --------------Read tiff file 2013-2018 day and night
    # read MOD11A1 data gaps statistics in 2013-2018
    gd = gdal.Open('../../data/processed_data/MOD11A1.Data_Gap_Statistics/Nodata.MultiYears/2013-2018_MOD_QC_Day_bit10.eq.01.10.11.tif')
    gd_band = gd.GetRasterBand(1)  # LST Change Rate
    print(gd_band)
    array_2013_2018_day = gd_band.ReadAsArray()
    mask = array_2013_2018_day > 366
    array_2013_2018_day = np.ma.masked_array(array_2013_2018_day, mask=mask)
    array_2013_2018_day = array_2013_2018_day[::-1, :]
    print('array_2013_2018_day_Max', np.nanmax(array_2013_2018_day))
    print('array_2013_2018_day_Min', np.nanmin(array_2013_2018_day))

    gd = gdal.Open('../../data/processed_data/MOD11A1.Data_Gap_Statistics/Nodata.MultiYears/2013-2018_MOD_QC_Night_bit10.eq.01.10.11.tif')
    gd_band = gd.GetRasterBand(1)  # LST Change Rate
    print(gd_band)
    array_2013_2018_night = gd_band.ReadAsArray()
    mask = array_2013_2018_night > 366
    array_2013_2018_night = np.ma.masked_array(array_2013_2018_night, mask=mask)
    array_2013_2018_night = array_2013_2018_night[::-1, :]
    print('array_2013_2018_night_Max', np.nanmax(array_2013_2018_night))
    print('array_2013_2018_night_Min', np.nanmin(array_2013_2018_night))

    # get lat/lon coordinates from file.
    coords = gd.GetGeoTransform()
    nlons = array_2013_2018_night.shape[1]  # dimension of longitude
    nlats = array_2013_2018_night.shape[0]  # dimension of latitude
    delon = coords[1]  # resolution of longitude
    delat = coords[5]  # resolution of latitude
    lons = coords[0] + delon * np.arange(nlons)
    lats = coords[3] + delat * np.arange(nlats)[::-1]  # reverse lats

    clip_shapefile = "../../data/raw_data/shp/heihe_huangwei"
    # Add shpfile to figure
    plot_shapefile = os.path.abspath('../../data/raw_data/shp/HeiheFluxsites/heihe_up_middle_down_stream')
    # setup basemap instance.
    # -------------------------------------------------  Plot  Figure   ---------------------------------------------
    # setup figure.
    fig = plt.figure(figsize=cm2inch(19, 7))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 0.3, 1.6])
    cmap = mpl.cm.jet
    bounds = [125, 150, 175, 200, 225, 250, 275, 300, 325, 350]
    # setup basemap instance.
    ax1 = fig.add_subplot(gs[0, 0])
    # setup basemap instance.
    m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
                projection='merc', suppress_ticks=False)
    x, y = m(*np.meshgrid(lons, lats))  # inverse=True can be used to obtain latitude and longitude value
    cbar = m.pcolormesh(x, y, array_2013_2018_day, cmap=cmap, norm=mpl.colors.Normalize(vmin=120, vmax=360))  #cm.GMT_haxby_r
    # mask the figure by shape file
    clip_patch = getClpPathPatch_FromShapeFile(clip_shapefile, 5, 'Heihe', ax1, m)
    if clip_patch is not None:
        for collection in ax1.collections:
            collection.set_clip_on(True)
            collection.set_clip_path(clip_patch)
    # Add shpfile to figure
    m.readshapefile(plot_shapefile, 'shpfile', linewidth=0.6, color='#444444',)
    # draw meridians and parallels.
    m.drawparallels(np.arange(37, 44, 1), labels=[1, 0, 0, 0], linewidth=0.0, xoffset=0.04*abs(m.xmax-m.xmin),
                    zorder=-2, fmt="%d", fontproperties=font_tick_label)
    m.drawmeridians(np.arange(95, 105, 2), labels=[0, 0, 0, 1], linewidth=0.0, yoffset=0.04*abs(m.ymax-m.ymin),
                    zorder=-2, fmt="%d", fontproperties=font_tick_label)
    # This code is to remove the line of longitude and latitude
    lon_ticks = np.arange(np.ceil(llcrnrlon), np.ceil(urcrnrlon), 1)
    lat_ticks = np.arange(np.ceil(llcrnrlat), np.ceil(urcrnrlat), 1)
    # convert from degree to map projection
    lon_ticks_proj, _ = m(lon_ticks, np.zeros(len(lon_ticks)))
    _, lat_ticks_proj = m(np.zeros(len(lat_ticks)), lat_ticks)
    # manually add ticks
    ax1.set_xticks(lon_ticks_proj)
    ax1.set_yticks(lat_ticks_proj)
    ax1.tick_params(axis='both', which='major')
    # add ticks to the opposite side as well
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    # remove the tick labels
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])

    # Add some text needed in figure
    x, y = m(x_sub_title, y_sub_title)
    ax1.text(x, y, '(a) Day', fontproperties=font_sub_title, transform=ax1.transData)
    cbar_ax = fig.add_axes([0.08, 0.12, 0.42, 0.02])
    cb = fig.colorbar(cbar, cax=cbar_ax, orientation="horizontal", spacing='uniform',
                      ticks=bounds)  # orientation = 'vertical'
    cb.ax.set_title('day', fontproperties=font_cb_title)
    cb.ax.title.set_position([1.05, -1.0])
    cb.ax.yaxis.label.set_font_properties(font_cb_tick_label)
    # -------------------------------------------------    SubFigure 2    ---------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    # setup basemap instance.
    m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
                projection='merc', suppress_ticks=False)
    x, y = m(*np.meshgrid(lons, lats))  # inverse=True can be used to obtain latitude and longitude value
    cbar = m.pcolormesh(x, y, array_2013_2018_night, cmap=cmap, norm=mpl.colors.Normalize(vmin=120, vmax=360))  #cm.GMT_haxby_r
    # mask the figure by shape file
    clip_patch = getClpPathPatch_FromShapeFile(clip_shapefile, 5, 'Heihe', ax2, m)
    if clip_patch is not None:
        for collection in ax2.collections:
            collection.set_clip_on(True)
            collection.set_clip_path(clip_patch)
            print(collection)

    # Add shpfile to figure
    m.readshapefile(plot_shapefile, 'shpfile', linewidth=0.6, color='#444444',)
    # draw meridians and parallels.
    m.drawparallels(np.arange(37, 44, 1), labels=[0, 0, 0, 0], linewidth=0.0, xoffset=0.04*abs(m.xmax-m.xmin),
                    zorder=-2, fmt="%d", fontproperties=font_tick_label)
    m.drawmeridians(np.arange(95, 105, 2), labels=[0, 0, 0, 1], linewidth=0.0, yoffset=0.04*abs(m.ymax-m.ymin),
                    zorder=-2, fmt="%d", fontproperties=font_tick_label)

    # This code is to remove the line of longitude and latitude
    lon_ticks = np.arange(np.ceil(llcrnrlon), np.ceil(urcrnrlon), 1)
    lat_ticks = np.arange(np.ceil(llcrnrlat), np.ceil(urcrnrlat), 1)
    # convert from degree to map projection
    lon_ticks_proj, _ = m(lon_ticks, np.zeros(len(lon_ticks)))
    _, lat_ticks_proj = m(np.zeros(len(lat_ticks)), lat_ticks)
    # manually add ticks
    ax2.set_xticks(lon_ticks_proj)
    ax2.set_yticks(lat_ticks_proj)
    ax2.tick_params(axis='both', which='major') #both
    # add ticks to the opposite side as well
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    # remove the tick labels
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])

    # Add some text needed in figure
    x, y = m(x_sub_title, y_sub_title)
    ax2.text(x, y, '(b) Night', fontproperties=font_sub_title, transform=ax2.transData)
    # # -------------------------------------------------    SubFigure 3    ---------------------------------------------
    # read Heihe River Basin Land Surface Temperature ###############
    # Whole Heihe river basin
    infile = "../../data/processed_data/MOD11A1.Data_Gap_Statistics/Nodata.MultiYears_Month.csv"
    usecols = ['Month', 'Daytime_Mean', 'Daytime_Std', 'Nighttime_Mean', 'Nighttime_Std']
    df_lst_gap = pd.read_csv(infile, usecols=usecols)
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.set_ylim([20, 80])
    ax3.set_yticks([30, 40, 50, 60, 70, 80])
    ax3.set_yticklabels([30, 40, 50, 60, 70, 80], fontproperties=font_tick_label)
    ax3.set_ylabel('Data gap percent (%)', fontproperties=font_tick_label)

    ax3.set_xlim([0, 13])
    ax3.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax3.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], fontproperties=font_tick_label)
    ax3.set_xlabel('Month', fontproperties=font_sub_title)

    ax3.plot(df_lst_gap['Month'], df_lst_gap['Daytime_Mean'], 'r-o', Markersize=4, linewidth=1, label=r'Day')
    ax3.errorbar(df_lst_gap['Month'], df_lst_gap['Daytime_Mean'], yerr=df_lst_gap['Daytime_Std'],linestyle='None',
                 capsize=3, ecolor='r')
    ax3.plot(df_lst_gap['Month'], df_lst_gap['Nighttime_Mean'], 'b-^', Markersize=4, linewidth=1, label=r'Night')  # ARC
    ax3.errorbar(df_lst_gap['Month'], df_lst_gap['Nighttime_Mean'], yerr=df_lst_gap['Nighttime_Std'], linestyle='None',
                 capsize=3, ecolor='b')
    plt.legend(frameon=False, bbox_to_anchor=(0.65, 0.70), fontsize=10)
    # Add some text needed in figure
    x, y = m(x_sub_title, y_sub_title)
    ax3.text(0.5, 75, '(c) ', fontproperties=font_sub_title)

    # Export Figure
    fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.00, hspace=0.0)
    plt.savefig('../../output_figures/Figure 2. MOD11A1 Data Gaps Statistics.tif', dpi=300, bbox_inches='tight')
    plt.show()