# -*- coding: utf-8 -*-
# using SOAPpy to download MODIS Data
# Tan Junlei 
from __future__ import division
import string
from SOAPpy import SOAPProxy
import os,glob,subprocess
import arcpy
import numpy as np
from scipy.spatial import cKDTree as KDTree
class Invdisttree:
    """ inverse-distance-weighted interpolation using KDTree:
invdisttree = Invdisttree( X, z )  -- data points, values
interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
    interpolates z from the 3 points nearest each query point q;
    For example, interpol[ a query point q ]
    finds the 3 data points nearest q, at distances d1 d2 d3
    and returns the IDW average of the values z1 z2 z3
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
        = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

    q may be one point, or a batch of points.
    eps: approximate nearest, dist <= (1 + eps) * true nearest
    p: use 1 / distance**p
    weights: optional multipliers for 1 / distance**p, of the same shape as q
    stat: accumulate wsum, wn for average weights

How many nearest neighbors should one take ?
a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
    |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
    I find that runtimes don't increase much at all with nnear -- ymmv.

p=1, p=2 ?
    p=2 weights nearer points more, farther points less.
    In 2d, the circles around query points have areas ~ distance**2,
    so p=2 is inverse-area weighting. For example,
        (z1/area1 + z2/area2 + z3/area3)
        / (1/area1 + 1/area2 + 1/area3)
        = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
    Similarly, in 3d, p=3 is inverse-volume weighting.

Scaling:
    if different X coordinates measure different things, Euclidean distance
    can be way off.  For example, if X0 is in the range 0 to 1
    but X1 0 to 1000, the X1 distances will swamp X0;
    rescale the data, i.e. make X0.std() ~= X1.std() .

A nice property of IDW is that it's scale-free around query points:
if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
the IDW average
    (z1/d1 + z2/d2 + z3/d3)
    / (1/d1 + 1/d2 + 1/d3)
is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
is exceedingly sensitive to distance and to h.

    """
# anykernel( dj / av dj ) is also scale-free
# error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__( self, X, z, leafsize=10, stat=0 ):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree( X, leafsize=leafsize )  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None;

    def __call__( self, q, nnear=6, eps=0, p=1, weights=None ):
            # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query( q, k=nnear, eps=eps )
        interpol = np.zeros( (len(self.distances),) + np.shape(self.z[0]) )
        jinterpol = 0
        for dist, ix in zip( self.distances, self.ix ):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot( w, self.z[ix] )
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]

def MODIS_hrb_fileurls_coords(outputdir_abs,product,collection,startyear,stopyear):
    ''' MOD05_L2,
    '''
    url = "http://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices?wsdl"
    client = SOAPProxy(url)
    north=42.916666645
    south=37.5
    west=97.0
    east=101.99999998
    for d in range(startyear,stopyear+1):
        startdate = "%d-01-01" %d
        stopdate = "%d-12-31" %d
    # HRB
        file_ids = client.searchForFiles(product=product,collection=collection, start=startdate,stop=stopdate,north=north,
                                         south=south,west=west,east=east,coordsOrTiles='coords',dayNightBoth='DN')
        file_num = len(file_ids)
        filename = product + ('_C%d_%d_DN_%d.txt'% (collection,d,file_num))
        print filename
        f = open(outputdir_abs+filename,'w')
        file_ids_str = string.join(file_ids,',')
        fileurls = client.getFileUrls(fileIds=file_ids_str)
        for element in fileurls:
            f.writelines(element+'\n')
        f.close()
        print(d,len(file_ids),'OK')
def MODIS_hrb_fileurls_tiles(outputdir_abs,product,collection,startyear,endyear):
    ''' MOD11A1，MOD11B1 MOD13A2'''
    url = "http://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices?wsdl"
    client = SOAPProxy(url)
    for d in range(startyear,endyear+1):
        startdate = "%d-01-01" %d
        stopdate = "%d-12-31" %d
    # QTP
    # D Ok; N ,No results; B, No results; DN, 1848 OK; NB, No results; DNB, 1848,OK
        file_ids = client.searchForFiles(product=product,collection=collection, start=startdate,stop=stopdate,
                                         north=4,south=4,west=24,east=25,coordsOrTiles='tiles',dayNightBoth='DN');
        file_ids =file_ids + client.searchForFiles(product=product,collection=collection, start=startdate,stop=stopdate,
                                                   north=5,south=5,west=25,east=26,coordsOrTiles='tiles',dayNightBoth='DN')
        file_num = len(file_ids)
        filename = product + '_C%d_%d_DN_%d.txt'% (collection,d,file_num)
        f = open(outputdir_abs+filename,'w')
        file_ids_str = string.join(file_ids,',')
        fileurls = client.getFileUrls(fileIds=file_ids_str)
        for element in fileurls:
            f.writelines(element+'\n')
        f.close()
        print(d,len(file_ids),'OK')
def MODIS_fileurls_common(outputdir_abs,product,collection,startyear,stopyear,north,south,west,east,coordsOrTiles,dayNightBoth):
    '''
    Example 1:
    product = 'MOD05_L2',collection = 6,startyear = 2001,stopyear = 2013,north=42.916666645,south=37.5
    west=97.0,east=101.99999998,coordsOrTiles='coords',dayNightBoth='DN'
    '''    
    url = "http://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices?wsdl"
    client = SOAPProxy(url)
    for d in range(startyear,stopyear+1):
        startdate = "%d-01-01" %d
        stopdate = "%d-12-31" %d
    # HRB
        file_ids = client.searchForFiles(product=product,collection=collection, start=startdate,stop=stopdate,
                                         north=north,south=south,west=west,east=east,coordsOrTiles='coords',dayNightBoth='DN')
        file_num = len(file_ids)
        filename = product + ('_C%d_%d_DN_%d.txt'% (collection,d,file_num))
        print filename
        f = open(outputdir_abs+filename,'w')
        file_ids_str = string.join(file_ids,',')
        fileurls = client.getFileUrls(fileIds=file_ids_str)
        for element in fileurls:
            f.writelines(element+'\n')
        f.close()
        print(d,len(file_ids),'OK')
class MRT_PRM_FILE:
    def __init__(self,product_name,abspath, SPECTRAL_SUBSET, SPATIAL_SUBSET_UL_CORNER,
                SPATIAL_SUBSET_LR_CORNER,DATUM,OUTPUT_PIXEL_SIZE):
        self.product_name = product_name
        self.abspath = abspath
        self.SPECTRAL_SUBSET = SPECTRAL_SUBSET
        self.SPATIAL_SUBSET_UL_CORNER = SPATIAL_SUBSET_UL_CORNER
        self.SPATIAL_SUBSET_LR_CORNER = SPATIAL_SUBSET_LR_CORNER
        self.DATUM = DATUM
        self.OUTPUT_PIXEL_SIZE = OUTPUT_PIXEL_SIZE
    def Create(self):
        mrt_prmfilename = os.path.join(self.abspath,self.product_name+'.prm')
        mrtparfile = open(mrt_prmfilename,"wt")
        mrtparfile.write("INPUT_FILENAME = temp\n")
        line = "SPECTRAL_SUBSET = ( %s )\n" % self.SPECTRAL_SUBSET
        mrtparfile.write(line)
        mrtparfile.write("SPATIAL_SUBSET_TYPE = INPUT_LAT_LONG\n")
        line = "SPATIAL_SUBSET_UL_CORNER = ( %s )\n" % self.SPATIAL_SUBSET_UL_CORNER
        mrtparfile.write(line)
        line = "SPATIAL_SUBSET_LR_CORNER = ( %s )\n" % self.SPATIAL_SUBSET_LR_CORNER
        mrtparfile.write(line)
        mrtparfile.write("OUTPUT_FILENAME = temp\n")
        mrtparfile.write("RESAMPLING_TYPE = NEAREST_NEIGHBOR\n")
        mrtparfile.write("OUTPUT_PROJECTION_TYPE = GEO\n")
        mrtparfile.write("OUTPUT_PROJECTION_PARAMETERS = ( \n")
        mrtparfile.write(" 0.0 0.0 0.0\n 0.0 0.0 0.0\n 0.0 0.0 0.0\n 0.0 0.0 0.0\n 0.0 0.0 0.0 )\n")
        line = "DATUM = %s\n" % self.DATUM
        mrtparfile.write(line)
        line = "OUTPUT_PIXEL_SIZE = %r\n" % self.OUTPUT_PIXEL_SIZE
        mrtparfile.write(line)
        mrtparfile.close()
        #print 'The file %s.prm is created in %s' % (self.product_name,self.abspath)
class MODIS_Process:
    '''Represents any school member.'''
    def __init__(self,product,startday,endday,spectal_subset,spatial_subset,fillvalue,data_abspath,tif_abspath,
                 temp_abspath,clpshape_abspath,target_abspath):
        self.product = product
        self.startday = startday
        self.endday = endday
        self.spectral_sbuset = spectal_subset
        self.spatial_subset = spatial_subset
        self.fillvalue = fillvalue
        self.data_abspath = data_abspath
        self.tif_abspath = tif_abspath
        self.temp_abspath = temp_abspath
        self.clpshape_abspath = clpshape_abspath
        self.target_abspath = target_abspath
    def mosaic(self):
        mrt_prmfilename = os.path.join(self.data_abspath,self.product+'.prm')
        mrtbatchfilename = os.path.join(self.data_abspath,"mrtmosaic_resample.bat")
        #print mrtbatchfilename
        mrtbatchfile = open(mrtbatchfilename,"wt")
        tempfilename = os.path.join(self.temp_abspath,"MOD_LST.HDF")
        if self.product in ['MOD13A2']:
            step = 16
        else:
            step = 1
        if self.product in ['MYD13A2']:
            step = 16
            self.startday = self.startday+8
        for d  in range(self.startday,self.endday+1,step):
            #print d
            # Create the list files used by MRT Tools
            inhdf = "%s.A%07d*.hdf" % (self.product,d)
            outlist = "list%07d.txt" % d
            outtif = "%s_%07d.tif" % (self.product,d)
            # absolute path
            filenames = glob.glob(os.path.join(self.data_abspath,inhdf))
            file_outlist = open(os.path.join(self.data_abspath,outlist),"wt")
            outlistname = os.path.join(self.data_abspath,outlist)
            outtifname = os.path.join(self.tif_abspath,outtif)
            # listfile by day
            for inhdffile in filenames:
                #filename = os.path.basename(file)
                file_outlist.write(inhdffile)
                file_outlist.write("\n")
            file_outlist.close()
            #MRTMosaic Resample the file
            mrtbatchfile.write("MRTMOSAIC -i "+outlistname)
            mrtbatchfile.write(' -o '+tempfilename)
            line = ' -s " %s "' % self.spectral_sbuset
            mrtbatchfile.write(line)
            mrtbatchfile.write("\n")
            mrtbatchfile.write("RESAMPLE -p "+mrt_prmfilename)
            mrtbatchfile.write(" -i "+tempfilename)
            mrtbatchfile.write(" -o "+outtifname)
            mrtbatchfile.write("\n")
                # Delete exist mosaiced tif files and cliped tif files
            if os.path.isfile(outtifname):
                os.remove(outtifname)
        mrtbatchfile.close()
        # run batch file  
        handle = subprocess.Popen(mrtbatchfilename, shell=True)
        handle.wait()
        line = "%s: %d-%d is Mosaiced by MRT" % (self.product,self.startday,self.endday)
        print line
    def del_tempfile(self,dir,postfix):
        if os.path.isdir(dir):
            for file in os.listdir(dir):
                self.del_tempfile(dir+"/"+file,postfix)
        else:
            if os.path.splitext(dir)[1] == postfix:
                os.remove(dir)
        hdftempfile = os.path.join(self.temp_abspath,"MOD_LST.HDF")
        if os.path.isfile(hdftempfile):
            os.remove(hdftempfile)
            
    def clip(self):
        # Create the WorkSpace,used by ListRasters
        arcpy.env.workspace = self.tif_abspath
        arcpy.env.overwriteOutput = True
        #get all rasters in the datadir
        rasters = arcpy.ListRasters()
        #loop
        for raster in rasters:
            # Local variables...
            Output_Raster = os.path.join(self.target_abspath,raster)
            # Process: Clip...
            # Bound: Huangweihui, Output PIXELS: 650*600
            arcpy.Clip_management(raster,self.spatial_subset ,Output_Raster,self.clpshape_abspath, 
                                  self.fillvalue,"ClippingGeometry","MAINTAIN_EXTENT")                      
        line = "%s: %d-%d is Cliped by ArcGIS" % (self.product,self.startday,self.endday)
        print line
def MOD11A1B1_ViewTimeAngl_Interp(in_dvt_abspath):
    arcpy.env.overwriteOutput = True
    # Check out any necessary licenses
    arcpy.CheckOutExtension("spatial")
    out_pre = os.path.splitext(os.path.basename(in_dvt_abspath))
    out_viewtime = out_pre[0]+'_interp.tif'
    output_dvt_abspath = os.path.join(in_dvt_abspath,out_viewtime)
    # Process: Raster Calculator
    mask = arcpy.sa.SetNull(in_dvt_abspath,in_dvt_abspath,"VALUE = 255")
    # Process: Nibble
    out_interp = arcpy.sa.Nibble(in_dvt_abspath, mask,"DATA_ONLY")
    out_interp.save(output_dvt_abspath)
def MOD11A1B1_Emissivity_Interp(in_emissivity_abspath):
    arcpy.env.overwriteOutput = True
    # Check out any necessary licenses
    arcpy.CheckOutExtension("spatial")
    out_pre = os.path.splitext(os.path.basename(in_emissivity_abspath))
    out_emissivity = out_pre[0]+'_interp.tif'
    output_emi_abspath = os.path.join(in_emissivity_abspath,out_emissivity)
    # Process: Raster Calculator
    mask = arcpy.sa.SetNull(in_emissivity_abspath,in_emissivity_abspath,"VALUE = 0")
    # Process: Nibble
    out_interp = arcpy.sa.Nibble(in_emissivity_abspath, mask,"DATA_ONLY")
    out_interp.save(output_emi_abspath)
    
def EmissivityBroadBand(cord_lonlat,emissivity_29,emissivity_31,emissivity_32,
                        day_viewtime,day_viewangl,night_viewtime,night_viewangl,QC_Emis):
    
    SampleValue = arcpy.GetCellValue_management(emissivity_29, cord_lonlat)    
    e29 = int(SampleValue.getOutput(0))*0.002+0.49
    
    SampleValue = arcpy.GetCellValue_management(emissivity_31, cord_lonlat)    
    e31 = int(SampleValue.getOutput(0))*0.002+0.49
    
    SampleValue = arcpy.GetCellValue_management(emissivity_32, cord_lonlat)
    e32 = int(SampleValue.getOutput(0))*0.002+0.49
    
    SampleValue = arcpy.GetCellValue_management(day_viewtime, cord_lonlat)    
    day_time = int(SampleValue.getOutput(0))*0.2

    SampleValue = arcpy.GetCellValue_management(day_viewangl, cord_lonlat)    
    day_angle = int(SampleValue.getOutput(0))*1-65

    SampleValue = arcpy.GetCellValue_management(night_viewtime, cord_lonlat)    
    night_time = int(SampleValue.getOutput(0))*0.2

    SampleValue = arcpy.GetCellValue_management(night_viewangl, cord_lonlat)    
    night_angle = int(SampleValue.getOutput(0))*1-65    

    SampleValue = arcpy.GetCellValue_management(QC_Emis, cord_lonlat)    
    QC_e = int(SampleValue.getOutput(0))  
    
    e_broadband = 0.2122*e29 + 0.3859*e31 + 0.4029*e32
    line1 = ' %8.6f %8.6f %8.6f %10.6f' % (e29,e31,e32,e_broadband)
    line2 = ' %8.2f %9d %10.2f %11d %4d' % (day_time,day_angle,night_time,night_angle,QC_e)
    result = emissivity_29[8:15] + line1 + line2  
    return result
    
def EmissivityBroadBand2(cord_lonlat,datadir,emissivity_29,emissivity_31,emissivity_32,
                        day_viewtime,day_viewangl,night_viewtime,night_viewangl,QC_Emis):
    
    RasterAbspath = os.path.abspath(os.path.join(datadir,emissivity_29))
    SampleValue = arcpy.GetCellValue_management(RasterAbspath, cord_lonlat)    
    e29 = int(SampleValue.getOutput(0))*0.002+0.49
    
    RasterAbspath = os.path.abspath(os.path.join(datadir,emissivity_31))
    SampleValue = arcpy.GetCellValue_management(RasterAbspath, cord_lonlat)    
    e31 = int(SampleValue.getOutput(0))*0.002+0.49
    
    RasterAbspath = os.path.abspath(os.path.join(datadir,emissivity_32))
    SampleValue = arcpy.GetCellValue_management(RasterAbspath, cord_lonlat)
    e32 = int(SampleValue.getOutput(0))*0.002+0.49
    
    RasterAbspath = os.path.abspath(os.path.join(datadir,day_viewtime))
    SampleValue = arcpy.GetCellValue_management(RasterAbspath, cord_lonlat)    
    day_time = int(SampleValue.getOutput(0))*0.2

    RasterAbspath = os.path.abspath(os.path.join(datadir,day_viewangl))
    SampleValue = arcpy.GetCellValue_management(RasterAbspath, cord_lonlat)    
    day_angle = int(SampleValue.getOutput(0))*1-65

    RasterAbspath = os.path.abspath(os.path.join(datadir,night_viewtime))
    SampleValue = arcpy.GetCellValue_management(RasterAbspath, cord_lonlat)    
    night_time = int(SampleValue.getOutput(0))*0.2

    RasterAbspath = os.path.abspath(os.path.join(datadir,night_viewangl))
    SampleValue = arcpy.GetCellValue_management(RasterAbspath, cord_lonlat)    
    night_angle = int(SampleValue.getOutput(0))*1-65    

    RasterAbspath = os.path.abspath(os.path.join(datadir,QC_Emis))
    SampleValue = arcpy.GetCellValue_management(RasterAbspath, cord_lonlat)    
    QC_e = int(SampleValue.getOutput(0))  
    
    e_broadband = 0.2122*e29 + 0.3859*e31 + 0.4029*e32
    line1 = '%10.5f,%7.3f,%7.3f,%7.3f,' % (e_broadband,e29,e31,e32)
    line2 = '%8.2f,%9d,%10.2f,%11d,%4d' % (day_time,day_angle,night_time,night_angle,QC_e)
#    result = emissivity_29[8:15] + line1 + line2    
    result = line1 + line2 
    return result
def LST_Interp_sample(cord_lonlat,day_viewtime_file,lst_interp_file,
                      lst_ori_file,day_viewangl_file,QC_day_file):
    
    SampleValue = arcpy.GetCellValue_management(day_viewtime_file, cord_lonlat)    
    day_time = int(SampleValue.getOutput(0))*0.1
    hour_diff = (120-float(cord_lonlat[:cord_lonlat.find(' ')]))*4/60
    if day_time<25.5:
        day_time = day_time+hour_diff
    SampleValue = arcpy.GetCellValue_management(lst_interp_file, cord_lonlat)    
    lst_interp = float(SampleValue.getOutput(0))
    
    SampleValue = arcpy.GetCellValue_management(lst_ori_file, cord_lonlat)    
    #lst_ori = int(SampleValue.getOutput(0))*0.02-273.15
    lst_ori = int(SampleValue.getOutput(0))*0.02

    SampleValue = arcpy.GetCellValue_management(day_viewangl_file, cord_lonlat)    
    day_angle = int(SampleValue.getOutput(0))*1-65  

    SampleValue = arcpy.GetCellValue_management(QC_day_file, cord_lonlat)    
    QC_lst = int(SampleValue.getOutput(0))  
    
    line1 = ',%5.2f,%7.3f,%8.3f,%8d,%6d' % (day_time,lst_interp,lst_ori,day_angle,QC_lst)
    filebasename = os.path.basename(lst_interp_file)
    result = filebasename[8:15] + line1   
    return result