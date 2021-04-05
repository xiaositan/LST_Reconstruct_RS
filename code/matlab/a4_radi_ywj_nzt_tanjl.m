%%  
dem_file='E:/modis.lst.interp/dem/hrb.600.661/hrb.dem.tif'; % the dem file should be in projected meter unit
[dem, RR, bbox]=geotiffread(dem_file);
info = geotiffinfo(dem_file);
tags = info.GeoTIFFTags.GeoKeyDirectoryTag;
dem=double(dem);
%ywj code
%dem(dem==-32768)=NaN;
%tanjl code 20160802
dem(dem==32767)=NaN;

%prepare the latitude file
[long, lat]=pix2map(RR, 1:size(dem,1), ones(1, size(dem,1)));
lat = lat';
%disp(lat);

%%%========================================================================
%%%=========== MOD Radiation===============================================
% lst time info file name pattern
year = 2012;
lst_time_file_pattern = sprintf('E:/modis.lst.interp/hrb.%d.C6/lst.MOD11A1/600.661/tif/MOD11A1_%d%s',year,year,'%03d.Day_view_time_interp.tif');
% output file name pattern
output_file_pattern = 'radiation_%03d.tif';
% the folder that outputs will be place
output_folder = sprintf('E:/modis.lst.interp/hrb.%d.C6/radiation/600.661/MOD11A1',year);
% ==================
if ~exist(output_folder)
    mkdir(output_folder);
end

%size(lat);

cs=829.2061142; %cell size, 1000 meter
r=0.2;
%n = 0.5;              % timestep of calculation over sunshine hours: 1=hourly, 0.5=30min, 2=2hours etc
tau_a = 366;     %length of the year in days
S0 = 1367;          % solar constant W m^-2   default 1367

dr= 0.0174532925;   % degree to radians conversion factor
%%
% Original code
% =============
% calculate slope and aspect (deg) using GRADIENT function
% [fx,fy] = gradient(dem,cs,cs); % uses simple, unweighted gradient of immediate neighbours
% [asp,grad]=cart2pol(fy,fx); % convert to carthesian coordinates
% grad=atan(grad); %steepest slope
% asp=asp.*-1+pi; % convert asp 0 facing south
% slop=grad;
%===============
%%tjl add============
slope_file='E:/modis.lst.interp/dem/hrb.600.661/hrb.slope.kang.tif'; % the dem file should be in projected meter unit
[slop, R1, bbox]=geotiffread(slope_file);
% info = geotiffinfo(dem_file);
% tags = info.GeoTIFFTags.GeoKeyDirectoryTag;
slop=double(slop);
slop(slop==32767)=NaN;
aspect_file='E:/modis.lst.interp/dem/hrb.600.661/hrb.aspect.tif'; % the dem file should be in projected meter unit
[aspect, R2, bbox]=geotiffread(aspect_file);
% info = geotiffinfo(dem_file);
% tags = info.GeoTIFFTags.GeoKeyDirectoryTag;
asp=double(aspect);
asp(aspect==32767)=NaN;
%%==================
% size(dem,2)返回矩阵的列数
% 生成矩阵，大小Size(lat)*dem的列数
%meshgrid用于从数组a和b产生网格。生成的网格矩阵A和B大小是相同的。它也可以是更高维的。这里的大小指的是，size()函数的大小，size()函数返回的是一个向量， 那么size(A) = size(B).
%[A,B]=Meshgrid(a,b)生成size(b)Xsize(a)大小的矩阵A和B。它相当于a从一行重复增加到size(b)行，把b转置成一列再重复增加到size(a)列。
[dummy,L]=meshgrid(1:size(dem,2),lat);  % grid latitude
clear dummy;
%size(L);
L=L*dr;                     % convert to radians
fcirc = 360*dr; % 360 degrees in radians

%% some setup calculations
srad=0;
sinL=sin(L);
cosL=cos(L);
tanL=tan(L);
sinSlop=sin(slop);
cosSlop=cos(slop);
cosSlop2=cosSlop.*cosSlop;
sinSlop2=sinSlop.*sinSlop;
sinAsp=sin(asp);
cosAsp=cos(asp);
term1 = ( sinL.*cosSlop - cosL.*sinSlop.*cosAsp);
term2 = ( cosL.*cosSlop + sinL.*sinSlop.*cosAsp);
term3 = sinSlop.*sinAsp;

modis_t1=0; % init offset time against sunrise

%% loop over year
tau_a = yeardays(year);
for d = 1:tau_a;
    %display(['Calculating melt for day ',num2str(d)])
    % clear sky solar radiation
    I0 = S0 * (1 + 0.0344*cos(fcirc*d/tau_a)); % extraterr rad per day
    % sun declination dS
    dS = 23.45 * dr* sin(fcirc * ( (284+d)/tau_a ) ); %in radians, correct/verified
    % angle at sunrise/sunset t = 1:It; % sun hour
    hsr = real(acos(-tanL*tan(dS)));  % angle at sunrise
    % this only works for latitudes up to 66.5 deg N! Workaround:
    % hsr(hsr<-1)=acos(-1); hsr(hsr>1)=acos(1);
    It=round(12*(1+mean(hsr(:))/pi)-12*(1-mean(hsr(:))/pi)); % calc daylength
    
    t2=round(12*(1+mean(hsr(:))/pi)); %sunset time
    t1=round(12*(1-mean(hsr(:))/pi)); %sunrise time
    
    %read in pass-over time 
    % 
    lst_time_file = sprintf(lst_time_file_pattern,d);
    [lst_tm, RR, bbox]=geotiffread(lst_time_file);
    lst_tm=double(lst_tm);
    lst_tm(lst_tm==0)=NaN;
    %lst_tm(lst_tm==255)=NaN;
    % add by tanjl============
    lst_tm(lst_tm==255)=NaN;
    %   %======================
    lst_tm = lst_tm * 0.1; 
    % 0.1 is a scale factor
    % calc offset time against sunrise
    modis_t1 = lst_tm -t1; 
    %   
    modis_t1(modis_t1<0)=0;

  
  %% calculate radiation at the given time slop
  t=modis_t1; 
        % if accounting for shading should be included, calc hillshade here
        % hourangle of sun hs
        hs=hsr-(pi*t/It);               % hs(t)
        %solar angle and azimuth alpha =
        %asin(sinL*sin(dS)+cosL*cos(dS)*cos(hs));% solar altitude angle
        sinAlpha = sinL.*sin(dS)+cosL.*cos(dS).*cos(hs);
        %alpha_s = asin(cos(dS)*sin(hs)/cos(alpha)); % solar azimuth angle
        % correction  using atmospheric transmissivity taub_b
        M=sqrt(1229+((614.*sinAlpha)).^2)-614.*sinAlpha; % Air mass ratio
        tau_b = 0.56 * (exp(-0.65*M) + exp(-0.095*M));
        tau_d = 0.271-0.294*tau_b; % radiation diffusion coefficient for diffuse insolation
        tau_r = 0.271+0.706*tau_b; % reflectance transmitivity
        % correct for local incident angle
        cos_i = (sin(dS).*term1) + (cos(dS).*cos(hs).*term2) + (cos(dS).*term3.*sin(hs));
        Is = I0 * tau_b; % potential incoming shortwave radiation at surface normal (equator)
        % R = potential clear sky solar radiation W m2
        R = Is .* cos_i;
        R(R<0)=0;  % kick out negative values
        Id = I0 .* tau_d .* cosSlop2./ 2.*sinAlpha; %diffuse radiation;
        Ir = I0 .* r .* tau_r .* sinSlop2./ 2.* sinAlpha; % reflectance
        R= R + Id + Ir;
        R(R<0)=0; 
                    
%    end % end of sun hours in day loop
     %test
     fprintf('Day %d @(320,300): %f\n',d, R(320,300));
     %prepare the output file name
     output_filen=sprintf(output_file_pattern,d);
     output_filen = fullfile(output_folder,output_filen);
     % print to geotiff
     geotiffwrite(output_filen,R, RR,'GeoKeyDirectoryTag',tags);
end   % end of days in year loop


% % %==========================================================================
% %================MYD Radiation=============================================
% year = 2018;
% lst_time_file_pattern = sprintf('../../hrb.%d.C6/lst.MYD11A1/600.661/tif/MYD11A1_%d%s',year,year,'%03d.Day_view_time_interp.tif');
% % output file name pattern
% output_file_pattern = 'radiation_%03d.tif';
% % the folder that outputs will be place
% output_folder = sprintf('../../hrb.%d.C6/radiation/600.661/MYD11A1',year);
% % ==================
% if ~exist(output_folder)
%     mkdir(output_folder);
% end
% 
% %size(lat);
% 
% cs=829.2061142; %cell size, 1000 meter
% r=0.2;
% %n = 0.5;              % timestep of calculation over sunshine hours: 1=hourly, 0.5=30min, 2=2hours etc
% tau_a = 366;     %length of the year in days
% S0 = 1367;          % solar constant W m^-2   default 1367
% 
% dr= 0.0174532925;   % degree to radians conversion factor
% %%
% % Original code
% % =============
% % calculate slope and aspect (deg) using GRADIENT function
% % [fx,fy] = gradient(dem,cs,cs); % uses simple, unweighted gradient of immediate neighbours
% % [asp,grad]=cart2pol(fy,fx); % convert to carthesian coordinates
% % grad=atan(grad); %steepest slope
% % asp=asp.*-1+pi; % convert asp 0 facing south
% % slop=grad;
% %===============
% %%tjl add============
% slope_file='../../dem/hrb.600.661/hrb.slope.tif'; % the dem file should be in projected meter unit
% [slop, R1, bbox]=geotiffread(slope_file);
% % info = geotiffinfo(dem_file);
% % tags = info.GeoTIFFTags.GeoKeyDirectoryTag;
% slop=double(slop);
% slop(slop==32767)=NaN;
% aspect_file='../../dem/hrb.600.661/hrb.aspect.tif'; % the dem file should be in projected meter unit
% [aspect, R2, bbox]=geotiffread(aspect_file);
% % info = geotiffinfo(dem_file);
% % tags = info.GeoTIFFTags.GeoKeyDirectoryTag;
% asp=double(aspect);
% asp(aspect==32767)=NaN;
% %%==================
% % size(dem,2)返回矩阵的列数
% % 生成矩阵，大小Size(lat)*dem的列数
% %meshgrid用于从数组a和b产生网格。生成的网格矩阵A和B大小是相同的。它也可以是更高维的。这里的大小指的是，size()函数的大小，size()函数返回的是一个向量， 那么size(A) = size(B).
% %[A,B]=Meshgrid(a,b)生成size(b)Xsize(a)大小的矩阵A和B。它相当于a从一行重复增加到size(b)行，把b转置成一列再重复增加到size(a)列。
% [dummy,L]=meshgrid(1:size(dem,2),lat);  % grid latitude
% clear dummy;
% %size(L);
% L=L*dr;                     % convert to radians
% fcirc = 360*dr; % 360 degrees in radians
% 
% %% some setup calculations
% srad=0;
% sinL=sin(L);
% cosL=cos(L);
% tanL=tan(L);
% sinSlop=sin(slop);
% cosSlop=cos(slop);
% cosSlop2=cosSlop.*cosSlop;
% sinSlop2=sinSlop.*sinSlop;
% sinAsp=sin(asp);
% cosAsp=cos(asp);
% term1 = ( sinL.*cosSlop - cosL.*sinSlop.*cosAsp);
% term2 = ( cosL.*cosSlop + sinL.*sinSlop.*cosAsp);
% term3 = sinSlop.*sinAsp;
% 
% modis_t1=0; % init offset time against sunrise
% 
% %% loop over year
% tau_a = yeardays(year);
% for d = 1:tau_a;
%     %display(['Calculating melt for day ',num2str(d)])
%     % clear sky solar radiation
%     I0 = S0 * (1 + 0.0344*cos(fcirc*d/tau_a)); % extraterr rad per day
%     % sun declination dS
%     dS = 23.45 * dr* sin(fcirc * ( (284+d)/tau_a ) ); %in radians, correct/verified
%     % angle at sunrise/sunset t = 1:It; % sun hour
%     hsr = real(acos(-tanL*tan(dS)));  % angle at sunrise
%     % this only works for latitudes up to 66.5 deg N! Workaround:
%     % hsr(hsr<-1)=acos(-1); hsr(hsr>1)=acos(1);
%     It=round(12*(1+mean(hsr(:))/pi)-12*(1-mean(hsr(:))/pi)); % calc daylength
%     
%     t2=round(12*(1+mean(hsr(:))/pi)); %sunset time
%     t1=round(12*(1-mean(hsr(:))/pi)); %sunrise time
%     
%     %read in pass-over time 
%     % 
%     lst_time_file = sprintf(lst_time_file_pattern,d);
%     [lst_tm, RR, bbox]=geotiffread(lst_time_file);
%     lst_tm=double(lst_tm);
%     lst_tm(lst_tm==0)=NaN;
%     %lst_tm(lst_tm==255)=NaN;
%     % add by tanjl============
%     lst_tm(lst_tm==255)=NaN;
%     %   %======================
%     lst_tm = lst_tm * 0.1; 
%     % 0.1 is a scale factor
%     % calc offset time against sunrise
%     modis_t1 = lst_tm -t1; 
%     %   
%     modis_t1(modis_t1<0)=0;
% 
%   
%   %% calculate radiation at the given time slop
%   t=modis_t1; 
%         % if accounting for shading should be included, calc hillshade here
%         % hourangle of sun hs
%         hs=hsr-(pi*t/It);               % hs(t)
%         %solar angle and azimuth alpha =
%         %asin(sinL*sin(dS)+cosL*cos(dS)*cos(hs));% solar altitude angle
%         sinAlpha = sinL.*sin(dS)+cosL.*cos(dS).*cos(hs);
%         %alpha_s = asin(cos(dS)*sin(hs)/cos(alpha)); % solar azimuth angle
%         % correction  using atmospheric transmissivity taub_b
%         M=sqrt(1229+((614.*sinAlpha)).^2)-614.*sinAlpha; % Air mass ratio
%         tau_b = 0.56 * (exp(-0.65*M) + exp(-0.095*M));
%         tau_d = 0.271-0.294*tau_b; % radiation diffusion coefficient for diffuse insolation
%         tau_r = 0.271+0.706*tau_b; % reflectance transmitivity
%         % correct for local incident angle
%         cos_i = (sin(dS).*term1) + (cos(dS).*cos(hs).*term2) + (cos(dS).*term3.*sin(hs));
%         Is = I0 * tau_b; % potential incoming shortwave radiation at surface normal (equator)
%         % R = potential clear sky solar radiation W m2
%         R = Is .* cos_i;
%         R(R<0)=0;  % kick out negative values
%         Id = I0 .* tau_d .* cosSlop2./ 2.*sinAlpha; %diffuse radiation;
%         Ir = I0 .* r .* tau_r .* sinSlop2./ 2.* sinAlpha; % reflectance
%         R= R + Id + Ir;
%         R(R<0)=0; 
%                     
% %    end % end of sun hours in day loop
%      %test
%      fprintf('Day %d @(320,300): %f\n',d, R(320,300));
%      %prepare the output file name
%      output_filen=sprintf(output_file_pattern,d);
%      output_filen = fullfile(output_folder,output_filen);
%      % print to geotiff
%      geotiffwrite(output_filen,R, RR,'GeoKeyDirectoryTag',tags);
% end   % end of days in year loop




