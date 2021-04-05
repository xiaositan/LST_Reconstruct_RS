function a3_interp_ref
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   MOD    %%%%%%%%%%%%%%%%%%%%%%%%%
%%%% MODIS Nodata
%%%% year = 2018;
%%%% output_dir = sprintf('../../hrb.%d.C6/lst.ref.interp/near.600.661/MOD11A1',year);
%%%% input_dir = sprintf('../../hrb.%d.C6/lst.MOD11A1/600.661/tif',year);
%%%% statoutput_dir = sprintf('../../hrb.%d.C6/lst.ref.interp/near.600.661',year);
%% 
year = 2015;
input_dir = sprintf('E:/modis.lst.interp/hrb.%d.C6/lst.flg.processed/MOD11A1/600.661',year);
output_dir = sprintf('E:/modis.lst.interp/hrb.%d.C6/lst.ref.interp/near.600.661/MOD11A1',year);
statoutput_dir = sprintf('E:/modis.lst.interp/hrb.%d.C6/lst.ref.interp/near.600.661',year);

stat_fn = 'MOD_Invalid_statistic.txt';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
lst_data_pat = 'M*.LST*.tif';
lst_data_fns = dir(fullfile(input_dir, lst_data_pat));
%disp(lst_data_fns);
% fid = fopen(fullfile(statoutput_dir,stat_fn), 'w');
% fprintf(fid, '%s %s %s\r\n','Filename', 'Invalid_number', 'Invalid_rate');
% for i = 1 : length(lst_data_fns)
for i = 325 : 340 %629 : 642 %325 : 340
   %disp(i);
   interp_image_fn = lst_data_fns(i).name;
   [interpnts,rate_interp]=interp_ref(interp_image_fn,input_dir,output_dir);
   %fprintf(fid,'%s %d %f\r\n',interp_image_fn, interpnts, rate_interp);
end   
%fclose(fid);

% % %%%%%%%%%%   MYD    %%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%year = 2018;
% %%%%output_dir = sprintf('../../hrb.%d.C6/lst.ref.interp/near.600.661/MYD11A1',year);
% %%%%input_dir = sprintf('../../hrb.%d.C6/lst.MYD11A1/600.661/tif',year);
% %%%%statoutput_dir = sprintf('../../hrb.%d.C6/lst.ref.interp/near.600.661',year);
% %%%%year = 2016;
% output_dir = sprintf('../../hrb.%d.C6/lst.ref.interp/near.600.661.V4/MYD11A1',year);
% input_dir = sprintf('../../hrb.%d.C6/lst.flg.processed/MYD11A1/600.661.V4',year);
% statoutput_dir = sprintf('../../hrb.%d.C6/lst.ref.interp/near.600.661.V4',year);
% stat_fn = 'MYD_Invalid_statistic.txt';
% if ~exist(output_dir, 'dir')
%     mkdir(output_dir);
% end
% lst_data_pat = 'M*.LST*.tif';
% lst_data_fns = dir(fullfile(input_dir, lst_data_pat));
% %disp(lst_data_fns);
% fid = fopen(fullfile(statoutput_dir,stat_fn), 'w');
% fprintf(fid, '%s %s %s\r\n','Filename', 'Invalid_number', 'Invalid_rate');
% for i = 1 : length(lst_data_fns)
%    %disp(i);
%    interp_image_fn = lst_data_fns(i).name;
%    [interpnts,rate_interp]=interp_ref(interp_image_fn,input_dir,output_dir);
%    fprintf(fid, '%s %d %f\r\n',interp_image_fn, interpnts, rate_interp);
% end   
% fclose(fid);
 end
function [interpnts,rate_interp]=interp_ref(input_image,input_dir,output_dir)
%locate combined reference image
%input_dir: the dir where is LST image to be combined and interpolate
%input_image: the file name of image whose reference image is to be
%determined
%ref_num: the no of reference image
%ref_fn: the file name of reference image
disp(input_image);
near_fns = cell(1,7);% near reference images for combined% MOD or MYD
satellite_type = input_image(1:3); 
year = str2num(input_image(9:12));% year
day = str2num(input_image(13:15));% day of year
ydays = yeardays(year);% days in one year
if strfind(input_image,'Day')
    time = 'Day';
elseif strfind(input_image,'Night')
    time = 'Night';
else
    disp('Do Not Know Day or Night Data');
end
if day < 4
    near_num = day+3;
    for i = 1:near_num
        near_fns{i} = sprintf('%3s11A1_%d%03d.LST_%s_1km.tif',satellite_type,year,day-1+i,time);
    end
elseif day > ydays-3
    near_num = ydays-day+4;
    for i = 1:near_num
        near_fns{i} = sprintf('%3s11A1_%d%03d.LST_%s_1km.tif',satellite_type,year,day-4+i,time);
    end
else
    near_num = 7;
    for i = 1:near_num
        near_fns{i} = sprintf('%3s11A1_%d%03d.LST_%s_1km.tif',satellite_type,year,day-4+i,time);
    end
end 
[temp_dat,R,bbox]=geotiffread(fullfile(input_dir, input_image));
[rows,columns] = size(temp_dat);
near_data = zeros(rows,columns,near_num);
for i = 1:near_num
    [tif_dat,R,bbox]=geotiffread(fullfile(input_dir, near_fns{i}));
    tif_dat = double(tif_dat);
    tif_dat(tif_dat==0)=NaN;
    tif_dat(tif_dat==65536)=NaN;
    tif_dat(tif_dat==65535)=NaN;
    near_data(:,:,i) = tif_dat;
end    
combineddata = nanmean(near_data,3);
combineddata(isnan(combineddata)) = 0;
combineddata = uint16(combineddata);
%find good quality points
validpnts = combineddata ~= 0; 
if sum(sum(validpnts))~=0;  
    %points to be interpolated
    interppnts = combineddata==0;
    interpnts = sum(sum(interppnts));
    rate_interp = interpnts/rows/columns;
    if rate_interp<0.06   
        [r, c] = find(validpnts);
        [xc, yc] = pix2map(R, r, c);
        vc = double(combineddata) * 0.02 - 273.15;
        vc = vc(validpnts);
        %only random 10% used
        validpnts_count = length(xc);
        imax = int32(validpnts_count /10);
        ss = randi(validpnts_count, imax,1);
        xc =xc(ss,1);
        yc = yc(ss,1);
        vc = vc(ss,1);
        
        [r, c]= find(interppnts);
        [xi, yi]=pix2map(R, r,c);
        
        %interpolate
        disp('..interpolating');
        vi = gIDW(xc,yc,vc,xi,yi,-2,'n',20);
        %
        vi = ( vi + 273.15 ) / 0.02;
        vi = uint16(vi);
        combineddata(interppnts)=vi;
        %save to disk
        disp('..output');
        geotiffwrite(fullfile(output_dir, input_image), combineddata, R);
    else
        disp('Not interpolating because of invalid number >6%');
        geotiffwrite(fullfile(output_dir, input_image), combineddata, R);
    end
else
    disp('All Data is 0');
    combineddata = zeros(rows,columns);
    geotiffwrite(fullfile(output_dir, input_image), combineddata, R)
    interpnts = 0;
    rate_interp = 1.0; 
end
end