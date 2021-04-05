function a5_interp_lst3_day_ndvi
%open dem, slope, and aspect
disp('reading dem, slope, aspect');
dem_file = 'E:/modis.lst.interp/dem/hrb.600.661/hrb.dem.tif';
slope_file = 'E:/modis.lst.interp/dem/hrb.600.661/hrb.slope.tif';
aspect_file = 'E:/modis.lst.interp/dem/hrb.600.661/hrb.aspect.tif';
%==========================================================================
[dem,R,bbox]=geotiffread(dem_file); %single
[slope,R,bbox]=geotiffread(slope_file); %single
[aspect,R,bbox]=geotiffread(aspect_file); %single
dem=double(dem); slope=double(slope); aspect=double(aspect); %double
% original
dem(dem==-32768)=NaN;
slope(slope==-32768)=NaN;
aspect(aspect==-32768)=NaN;
bnd = dem==-32768;  
year = 2015;
%=============================================================================
%%%%%=========================MOD==========================================
lst_dir = sprintf('E:/modis.lst.interp/hrb.%d.C6/lst.flg.processed/MOD11A1/600.661',year); %dir of lst files to be interpolated
ref_dir = sprintf('E:/modis.lst.interp/hrb.%d.C6/lst.ref.interp/near.600.661/MOD11A1',year); %dir of reference lst files
output_dir = sprintf('E:/modis.lst.interp/hrb.%d.C6/lst.interp/600.661/combined/MOD11A1',year); %output dir4
ref_statistic = sprintf('E:/modis.lst.interp/hrb.%d.C6/lst.ref.interp/near.600.661/MOD_Invalid_statistic.txt',year);
%create output dir
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
%radiation files
radiation_dir= sprintf('E:/modis.lst.interp/hrb.%d.C6/radiation/600.661/MOD11A1',year);
radiation_fl_pattern = 'radiation_%s.tif'; % %s字符格式的一个标准化输出
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%============================
% add by tanjl
% ndvi files
ndvi_dir = sprintf('E:/modis.lst.interp/hrb.%d.C6/ndvi.MOD13A2/600.661/tif',year);
ndvi_fl_pattern = sprintf('MOD13A2_%d%s',year,'%s.1_km_16_days_NDVI.tif');                                                                                            
%get all lst files ready for interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lst_data_pat = sprintf('MOD11A1_%d%s',year,'*.LST_Day_1km.tif');% *代表代表任意字符
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% list files that match the string lst_data_pat
lst_data_fns = dir(fullfile(lst_dir, lst_data_pat)); 

%len = length(lst_data_fns);
%for i=1:length(lst_data_fns)
% 2011169:176 %194 289 
for i=318:318 %length(lst_data_fns) % 2015 163:170
	interp_image_fn = lst_data_fns(i).name;
    %disp(interp_image_fn);
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	namex1=interp_image_fn(13:15); % 13:15, string of interplotion file DOY(day of year)
    %namex1=interp_image_fn(21:23);
	% namex2=interp_image_fn(20:23)
    %Formate output radiation file name and fullname 
	radiation_fn = sprintf(radiation_fl_pattern, namex1);
	radiation_file=fullfile(radiation_dir,radiation_fn);    
	[radiation,R,bbox]=geotiffread(radiation_file);
	radiation=double(radiation);
	radiation(bnd)=NaN;
     %because radiation tif file already with NaN, no need for next line
	 %radiation(radiation==0)=NaN;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%======================================================
    %====add by tanjl
    if str2num(namex1)>=346
        namex2 = '353';
    elseif str2num(namex1)<=9
        namex2 = '001';
    elseif (str2num(namex1)<=15 && str2num(namex1)>=10)
        namex2 = '017';
    else
        if mod(str2num(namex1),16)<=9
            namex2 = sprintf('%03d',fix(str2num(namex1)/16)*16+1);
        else 
            namex2 =  sprintf('%03d',fix(str2num(namex1)/16)*16+17);
        end
    end 
    %disp(namex2);
	ndvi_fn = sprintf(ndvi_fl_pattern, namex2);
    disp(ndvi_fn);
	ndvi_file=fullfile(ndvi_dir,ndvi_fn);    
	[ndvi,R,bbox]=geotiffread(ndvi_file);
    ndvi=double(ndvi);
	ndvi(ndvi==65535)=NaN;
    %disp(ndvi);
    %========================================================
	decompose2_func(interp_image_fn, lst_dir, ref_dir, ref_statistic,output_dir, dem,...
        slope, aspect,radiation,ndvi);
end


% %=============================================================================
% %%%%%=========================MYD==========================================
% %=============================================================================
% lst_dir = sprintf('../../hrb.%d.C6/lst.flg.processed/MYD11A1/600.661',year); %dir of lst files to be interpolated
% ref_dir = sprintf('../../hrb.%d.C6/lst.ref.interp/near.600.661/MYD11A1',year); %dir of reference lst files
% output_dir = sprintf('../../hrb.%d.C6/lst.interp/600.661/combined/MYD11A1',year); %output dir4
% ref_statistic = sprintf('E:/modis.lst.interp/hrb.%d.C6/lst.ref.interp/near.600.661/MYD_Invalid_statistic.txt',year);
% %create output dir
% if ~exist(output_dir, 'dir')
%     mkdir(output_dir);
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%
% %radiation files
% radiation_dir= sprintf('../../hrb.%d.C6/radiation/600.661/MYD11A1',year);
% radiation_fl_pattern = 'radiation_%s.tif'; % %s字符格式的一个标准化输出
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %============================
% % add by tanjl
% % ndvi files
% ndvi_dir = sprintf('../../hrb.%d.C6/ndvi.MYD13A2/600.661/tif',year);
% ndvi_fl_pattern = sprintf('MYD13A2_%d%s',year,'%s.1_km_16_days_NDVI.tif');                                                                                             
% %get all lst files ready for interpolation
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lst_data_pat = sprintf('MYD11A1_%d%s',year,'*.LST_Day_1km.tif');% *代表代表任意字符
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % list files that match the string lst_data_pat
% lst_data_fns = dir(fullfile(lst_dir, lst_data_pat)); 
% %len = length(lst_data_fns);
% %for i=1:length(lst_data_fns)
% %194 289 340
% for i=1:365
% 	interp_image_fn = lst_data_fns(i).name;
% 	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	namex1=interp_image_fn(13:15); % 13:15, string of interplotion file DOY(day of year)
%     %namex1=interp_image_fn(21:23);
% 	% namex2=interp_image_fn(20:23)
%     %Formate output radiation file name and fullname 
% 	radiation_fn = sprintf(radiation_fl_pattern, namex1);
% 	radiation_file=fullfile(radiation_dir,radiation_fn);    
% 	[radiation,R,bbox]=geotiffread(radiation_file);
% 	radiation=double(radiation);
% 	radiation(bnd)=NaN;
%      %because radiation tif file already with NaN, no need for next line
% 	 %radiation(radiation==0)=NaN;
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	%======================================================
%     %====add by tanjl
%     if str2num(namex1)>=354
%         namex2 = '361';
%     elseif str2num(namex1)<=17
%         namex2 = '009';
%     else
%         if mod(str2num(namex1),16)>=2
%             namex2 = sprintf('%03d',fix(str2num(namex1)/16)*16+9);
%         else 
%             namex2 =  sprintf('%03d',fix(str2num(namex1)/16)*16-7);
%         end 
%     end
%     %disp(namex2);
% 	ndvi_fn = sprintf(ndvi_fl_pattern, namex2);
%     disp(ndvi_fn);
% 	ndvi_file=fullfile(ndvi_dir,ndvi_fn);    
% 	[ndvi,R,bbox]=geotiffread(ndvi_file);
% 	ndvi=double(ndvi);
% 	ndvi(ndvi==65535)=NaN;    
%     %========================================================
% 	decompose2_func(interp_image_fn, lst_dir, ref_dir, ref_statistic,output_dir, dem, ...
%         slope, aspect,radiation,ndvi);
% end

function decompose2_func(interp_image_fn, lst_dir,ref_dir, ref_statistic, output_dir, ...
    dem, slope, aspect, radiation,ndvi )
%interp_image_fn: the filename of the image to be interpolated
%lst_dir: the dir of lst files for interpolation
%ref_dir: the dir of ref files
%output_dir: output dir
%dem: dem matrix
%slope: slope matrix
%aspect: aspect matrix
%radiation: radiation matrix
%ndvi: ndvi matrix
%%%%%%%%%%%%%
% tic:  starts a stopwatch timer to measure performance. 
% The function records the internal time at execution of the tic command. 
% Display the elapsed time with the toc function.
tic
%interp_image_fn = 'MOD11A1_2005003.LST_Day_1km.tif';
disp(interp_image_fn);

%locate and open the reference image
[ref_fn]=a3_locate_ref(interp_image_fn,ref_statistic);
fprintf('..ref image: %s\n',ref_fn);%formatted output
%open reference image
[ref_lst, R, bbox]=geotiffread(fullfile(ref_dir, ref_fn)); %unsigned integer
ref_lst=double(ref_lst);
ref_lst(ref_lst==65535)=NaN;
%compute real LST 
%ref_lst = ref_lst * 0.02 - 273.15;
ref_lst = ref_lst * 0.02;
%open lst image to be interpolated
%bnd: bound of lst file
disp('..load lst image');
[lst, R, bbox]=geotiffread(fullfile(lst_dir, interp_image_fn)); %unsigned integer
bnd = lst==65535;
%bnd = lst==-32768;
%interp_pnts: to be interpolation
interp_pnts = lst ==0;
%============= add by tanjl
no_pnts = length(lst(~bnd));
%==========================
%lst = double(lst) * 0.02 - 273.15; %compute real LST
lst = double(lst) * 0.02;
lst(bnd | interp_pnts)=NaN; % change lst matrix of bnd or interp_pnts to NaN

%%%%%%%%%%%%%%%%%%%%%%%%
%make sure no NaN within the boundary in radiation
radi_mn = mean(mean(radiation(~isnan(radiation))));
radiation(~bnd & isnan(radiation)) = radi_mn;
%make sure NaN for pixels outside the boundary in radiation
radiation(bnd) = NaN;
%%%%%%%%%%%%%%%%%%%%%%%%
%valid pnts in ref image and others
ref_lst_other = ref_lst(~isnan(ref_lst) & ~interp_pnts);
dem_other = dem(~isnan(dem) & ~interp_pnts);
slope_other = slope(~isnan(slope) & ~interp_pnts);
aspect_other = aspect(~isnan(aspect) & ~interp_pnts);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
radiation_other = radiation(~isnan(radiation) & ~interp_pnts);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=========================================================
%=====add by tanjl
ndvi(bnd) = NaN;
ndvi_other = ndvi(~isnan(ndvi) & ~interp_pnts);
%=========================================================
%valid pnts in lst image: lst_other(interp_pnts/bnd)
lst_other = lst(~isnan(ref_lst) & ~interp_pnts);% from matrix to sequence
%randomly 10% used 
validpnts_count = length(ref_lst_other); % no. of all valid pnts
%若20%<QC,imax=validpnts_count;
%imax = int32(validpnts_count /10); % Convert to 32-bit signed integer(nearest)
%imax = int32(validpnts_count /5);
% add by tanjl=========================
% if validpnts_count==0
% 
if validpnts_count < int32(no_pnts*0.3)
    imax = validpnts_count;
else
    imax = int32(no_pnts*0.3);
end
%======================================
if imax > 0
    % create a imax-by-1 vector of random integers between 1 and validpnts_count. add by tanjl.
    ss = randi(validpnts_count, imax,1);
    ref_lst_other = ref_lst_other(ss,1);% choose imax points randomly from 1 to validpnts_count
    dem_other=dem_other(ss,1);
    slope_other=slope_other(ss,1);
    aspect_other=aspect_other(ss,1);
    lst_other=lst_other(ss,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    radiation_other=radiation_other(ss,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %=========================================================
    %=====add by tanjl
    ndvi_other = ndvi_other(ss,1);
    %=========================================================
    
    %find:Find indices and values of nonzero element
    %pnts to be interpolated,rs:rows,cs:columns
    [rs, cs]=find(interp_pnts);
    interp_count = length(rs);
    % interp_count = 500;
    
    Rs = zeros(interp_count, 1);%0 matrix interp_count-by-1
    interp_vals = zeros(interp_count, 1);
    rmse_ywj = zeros(interp_count, 1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %CPU no.
    labs = 4;
    %lsf = parcluster('scheduler', 'configuration','lsfconfig');
    if verLessThan('matlab', '8.3')
        %lsf = findResource('scheduler', 'configuration','lsfconfig');
        lsf = findResource('scheduler', 'configuration','local');
        pjob = createParallelJob(lsf);
    else
        lsf = parcluster();
        pjob = createCommunicatingJob(lsf,'Type','spmd');
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    createTask(pjob, @interp_priv, 3, {interp_count,  ...
        rs, cs, ref_lst, dem, slope, aspect, radiation, ndvi,lst_other, ref_lst_other, dem_other,...
        slope_other, aspect_other,radiation_other,ndvi_other});
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if verLessThan('matlab', '8.3')
        set(pjob, 'MaximumNumberOfWorkers', labs);
    else
        pjob.NumWorkersRange=[1,labs];
    end
    
    %alltasks = get(pjob, 'Tasks');
    %set(alltasks, 'CaptureCommandWindowOutput', true)
    
    disp('..submit job');
    submit(pjob);
    
    if verLessThan('matlab', '8.3')
        waitForState(pjob, 'finished');
        %outputmessages = get(alltasks, 'CommandWindowOutput')
        outputs = getAllOutputArguments(pjob);
    else
        wait(pjob);
        outputs = fetchOutputs(pjob);
    end
    % {{Rs}, {Intp}}
    %{{Rs}, {Intp}}
    for i=1:size(outputs,1)
        Rs = Rs + outputs{i,1};
        interp_vals = interp_vals+ outputs{i,2};
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        rmse_ywj = rmse_ywj+ outputs{i,3};
    end
    if verLessThan('matlab', '8.3')
        destroy(pjob);
    else
        delete(pjob);
    end
    clear pjob;
    %processing time
    fprintf('--elapsed: %fs\n',toc);
    
    %save the interpolate points value to lst file
    lst(interp_pnts) = interp_vals;
    disp('..lst saved');
    option.NaN=-32768;
    %geotiffwrite(fullfile(output_dir, interp_image_fn), bbox, lst, 32, option);
    %=======tanjl add
    lst(bnd) = -32768;
    geotiffwrite(fullfile(output_dir, interp_image_fn), lst, R);
    %=================
    %save Rs: sqrt(R2 statistic) to corr....txt
    disp('..save corr');
    fid=fopen(fullfile(output_dir, ['corr.' interp_image_fn '.txt']), 'w');
    fprintf(fid, '%f \n',Rs');
    fclose(fid);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   rmse_ywj: Root Mean Squre Error(lst_other vs interp_val_cal calucate by ref_lst_other with B )
    fid=fopen(fullfile(output_dir, ['rmse.' interp_image_fn '.txt']), 'w');
    fprintf(fid, '%f \n', rmse_ywj');
    fclose(fid);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %save interpolated points locations(latitude/longitude)
    disp('..save interpolated points locations');
    fid=fopen(fullfile(output_dir, ['interp_pnts.' interp_image_fn '.txt']), 'w');
    [x, y]=pix2map(R, rs, cs);
    fprintf(fid, '%f %f\t\n',[x y]');
    fclose(fid);
else
    lst(bnd) = -32768;
    lst(~bnd) = 0;
    geotiffwrite(fullfile(output_dir, interp_image_fn), lst, R);
    %==========================================
    %save Rs: sqrt(R2 statistic) to corr....txt
    disp('..save corr');
    fid=fopen(fullfile(output_dir, ['corr.' interp_image_fn '.txt']), 'w');
    fprintf(fid, 'There is no data in this region \n');
    fclose(fid);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   rmse_ywj: Root Mean Squre Error(lst_other vs interp_val_cal calucate by ref_lst_other with B )
    fid=fopen(fullfile(output_dir, ['rmse.' interp_image_fn '.txt']), 'w');
    fprintf(fid, 'There is no data in this region\n');
    fclose(fid);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %save interpolated points locations(latitude/longitude)
    disp('..save interpolated points locations');
    fid=fopen(fullfile(output_dir, ['interp_pnts.' interp_image_fn '.txt']), 'w');
    fprintf(fid, 'There is no data in this region\n');
    fclose(fid);
end

function [Rs, interp_vals, rmse_ywj]=interp_priv(interp_count, ...
    rs, cs, ref_lst, dem, slope, aspect,radiation, ndvi,lst_other, ref_lst_other, dem_other,...
    slope_other, aspect_other, radiation_other,ndvi_other)
    %   Rs: sqrt(R2 statistic) 
    %   interp_vals: interpolation points values matrix
    %   rmse_ywj: Root Mean Squre Error(lst_other vs interp_val_cal calucate by ref_lst_other with B )
    %   rs,cs: interpolation pnts rows and columns 
    rmse_ywj = zeros(interp_count, 1);
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Rs = zeros(interp_count, 1);
    interp_vals = zeros(interp_count, 1);
    %   parallelization according by cpu no. labs and labindex
    part = int32(interp_count/numlabs);
    if part ~= interp_count/numlabs; part=part +1; end;
    
    no_start = (labindex-1) * part +1;
    no_end = labindex * part;
    if no_end > interp_count; no_end = interp_count; end;
    
    %  disp('..start iteration');
    for i=no_start:no_end
      
        %each pnt for interpolation
        r=rs(i);
        c=cs(i);
        ref_lst_rc = ref_lst(r,c);
        dem_rc = dem(r,c);
        slope_rc = slope(r,c);
        aspect_rc = aspect(r,c);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        radiation_rc = radiation(r,c);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=========================================================
%=====add by tanjl
        ndvi_rc = ndvi(r,c);
%=========================================================
        %   min and max in ref image and others
        ref_lst_other_max = max(ref_lst_other);
        ref_lst_other_min = min(ref_lst_other);
        dem_other_max = max(dem_other);
        dem_other_min = min(dem_other);
        slope_other_max = max(slope_other);
        slope_other_min = min(slope_other);
        aspect_other_max = max(aspect_other);
        aspect_other_min = min(aspect_other);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        radiation_other_max = max(radiation_other);
        radiation_other_min = min(radiation_other);
%       NSSR radiation code
        if radiation_other_min<10 ; radiation_other_min=10;end
        disp(radiation(100,100))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%		
%=========================================================
%=====add by tanjl
        ndvi_other_max = max(ndvi_other);
        disp(ndvi_other_max);
        ndvi_other_min = min(ndvi_other);
        if ndvi_other_min<0 ; radiation_other_min=0;end
%=========================================================	
        %   normalize of other points
        ref_lst_other1 = (ref_lst_other- ref_lst_other_min ) ./ (ref_lst_other_max -ref_lst_other_min );
        dem_other1 = ( dem_other- dem_other_min) ./ (dem_other_max - dem_other_min);
        slope_other1 = ( slope_other - slope_other_min) ./ (slope_other_max - slope_other_min);
        aspect_other1 = ( aspect_other -aspect_other_min ) ./ (aspect_other_max - aspect_other_min);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%		
		radiation_other1 = ( radiation_other -radiation_other_min ) ./ (radiation_other_max - radiation_other_min);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=========================================================
%=====add by tanjl
        ndvi_other1 = ( ndvi_other -ndvi_other_min ) ./ (ndvi_other_max - ndvi_other_min);
%=========================================================
        %   normalize of interpolation point
        ref_lst_rc1 = (  ref_lst_rc- ref_lst_other_min) ./ (ref_lst_other_max - ref_lst_other_min);
        dem_rc1 = (  dem_rc - dem_other_min) ./ (dem_other_max -dem_other_min );
        slope_rc1 = (  slope_rc - slope_other_min) ./ (slope_other_max -slope_other_min );
        aspect_rc1 = (  aspect_rc -aspect_other_min) ./ (aspect_other_max - aspect_other_min);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        radiation_rc1 = (  radiation_rc -radiation_other_min) ./ (radiation_other_max - radiation_other_min);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%		
%=========================================================
%=====add by tanjl
        if ndvi_rc >= 0;
            ndvi_rc1 = ( ndvi_rc -ndvi_other_min ) ./ (ndvi_other_max - ndvi_other_min);
            %=========================================================
             %shortest distance
            dist = (ref_lst_other1 - ref_lst_rc1 ).^2 + (dem_other1 - dem_rc1 ).^2 ...
                + (slope_other1 - slope_rc1 ).^2 + (aspect_other1 - aspect_rc1 ).^2 + (radiation_other1 - radiation_rc1 ).^2 ...
                + (ndvi_other1 - ndvi_rc1 ).^2;
        else
            %%% ndvi<0 water body, didn't consider well
            dist = (ref_lst_other1 - ref_lst_rc1 ).^2 + (dem_other1 - dem_rc1 ).^2 ...
                + (slope_other1 - slope_rc1 ).^2 + (aspect_other1 - aspect_rc1 ).^2 + (radiation_other1 - radiation_rc1 ).^2;
        end;
		 
        dist = sqrt(dist);
        %   corresponding lst
        %	lst_other, in same order
        %   [dist indices] = sort(dist);
        %   interest_portion = dist<0.5; 
        interest_portion = dist<0.3; 
        %fprintf('....pixel: %ld, min dist: %f, max dist: %f, <0.5#: %d\n', ...
        %i, min(dist), max(dist), length(find(interest_portion)));
        %indices10 = indices(1: int32(length(indices)/10));

        % only a small portion with smallest distances used for correlation
        %	R = corr([ref_lst_other(interest_portion) lst_other(interest_portion)]);
        %	Rs(i)=R(1,2);
                %% Added by TanJL, 2019.12.21
%          lst_other_filtered_step1 = lst_other(interest_portion);
%          ref_lst_other_filtered_step1 = ref_lst_other(interest_portion);
%          lst_mean = mean(lst_other_filtered_step1) ;
%          lst_sigma = std(lst_other_filtered_step1) ;
%          lst_data_filter1 = abs(lst_other_filtered_step1 - lst_mean)<= 3*lst_sigma;
%          ref_lst_other_filtered_step1 = ref_lst_other(interest_portion);
%          ref_lst_mean = mean(ref_lst_other_filtered_step1) ;
%          ref_lst_sigma = std(ref_lst_other_filtered_step1) ;
%          lst_data_filter2 = abs(ref_lst_other_filtered_step1 - ref_lst_mean)<= 3*ref_lst_sigma;
%          lst_data_filter = lst_data_filter1& lst_data_filter2;
         
         lst_other_ref_difference = lst_other(interest_portion)-ref_lst_other(interest_portion);
         lst_median = median(lst_other_ref_difference);
         lst_S = 1.4826*median(abs(lst_other_ref_difference-lst_median));
         lst_data_filter = abs(lst_other_ref_difference-lst_median)<3*lst_S;
      try
      	[B,dummy,dummy1, dummy2, stats]=regress(lst_other(lst_data_filter), ...
            [ref_lst_other(lst_data_filter), ones(length(find(lst_data_filter)), 1)]); 
      	Rs(i)=sqrt(stats(1,1));
      catch
          %disp('....error occurred, in regression');
          B=[1 0];
          Rs(i)=0.7;
      end
      % disp('..interpolated');
      interp_val =  B(2) + B(1) * ref_lst_rc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
%%interp_val_cal是用相似像元集合ref_lst_other计算出来的模拟值，然后计算模拟值interp_val_cal和真实值的rmse
       interp_val_cal =  B(2) + B(1) * ref_lst_other(lst_data_filter);
	   len_ywj=length(ref_lst_other(lst_data_filter));
	   rmse_ywj(i)=sqrt(sum((lst_other(lst_data_filter)-interp_val_cal).^2)/len_ywj);      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%		

%       try
%       	[B,dummy,dummy1, dummy2, stats]=regress(lst_other(interest_portion), ...
%             [ref_lst_other(interest_portion), ones(length(find(interest_portion)), 1)]); 
%       	Rs(i)=sqrt(stats(1,1));
%       catch
%           %disp('....error occurred, in regression');
%           B=[1 0];
%           Rs(i)=0.7;
%       end
%       % disp('..interpolated');
%       interp_val =  B(2) + B(1) * ref_lst_rc;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
% %%interp_val_cal是用相似像元集合ref_lst_other计算出来的模拟值，然后计算模拟值interp_val_cal和真实值的rmse
%        interp_val_cal =  B(2) + B(1) * ref_lst_other(interest_portion);
% 	   len_ywj=length(ref_lst_other(interest_portion));
% 	   rmse_ywj(i)=sqrt(sum((lst_other(interest_portion)-interp_val_cal).^2)/len_ywj);      
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%		
	  
        %Rs(i) = R(1,2);
       % fprintf('....a: %f, b: %f, value: %f, ref_val: %f\n', B(1), B(2), interp_val, ref_lst_rc);
       % fprintf('......corr: %f\n', Rs(i));
       if (interp_val >330 || interp_val<240 || rmse_ywj(i)>6)
           interp_val =0;
           interp_vals(i)=interp_val;
       else
           interp_vals(i)=interp_val;
       end

         %if(mod(i,100)==1) 
         %   elapse=toc;
         %  fprintf('--i: %d, elapsed: %fs\n',i, elapse);
         %end
      %  fprintf('....%d done\n',i);
    end
    
function [ref_fn]=a3_locate_ref(input_image,ref_statistic)
%locate reference image
%input_image: the file name of image whose reference image is to be
%determined
%ref_statistic: the file abs path of image invalid points number statistic
%file
%ref_fn: the file name of reference image
%input_image = 'MOD11A1_2012362.LST_Day_1km.tif'
%ref_statistic = 'E:/modis.lst.interp/hrb.2012.C6/lst.ref.interp/near/MOD_Invalid_statistic.txt';
[filenames,invalid_num,invalid_rate]=textread(ref_statistic,'%s %d %f','headerlines',1);
%disp(size(filenames));
ref_fn = input_image;
i = 0;
iflag = '-';
%% while ref_fn == input file name and invalid_rate <= 0.05 then 
%       ref_fn = input file name
% while invalid_rate > 0.05 find the nearest date file at invalid_rate <= 0.05
while (sum(strcmp(filenames,ref_fn))==0 || invalid_rate(strcmp(filenames,ref_fn))>0.05)
    day = input_image(13:15);
    day = str2num(day);
    i = i+1;
    %disp(i);
    if strcmp('+',iflag)
        iflag = '-';
    else
        iflag = '+';
    end
    day = day + str2num(strcat(iflag,num2str(fix(i/2)+1)));
    day = sprintf('%03d',day);
    ref_fn = strcat(input_image(1:12),day,input_image(16:end));
%disp('ref_fn')
%disp(ref_fn)
end
