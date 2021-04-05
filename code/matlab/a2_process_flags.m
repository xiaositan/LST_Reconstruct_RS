%% statistic MOD11A1
year = 2009;
tif_dir = sprintf('E:/modis.lst.interp/hrb.%d.C6/lst.MOD11A1/600.661/tif',year);
lst_data_pat = sprintf('MOD11A1_%d*.LST*.tif',year);
lst_qc_pat= sprintf('MOD11A1_%d*.QC*.tif',year);

save_dir = sprintf('E:/modis.lst.interp/hrb.%d.C6/lst.flg.processed/MOD11A1/600.661',year);
stat_fn = 'MOD11A1.stat.txt';

lst_data_fns = dir(fullfile(tif_dir, lst_data_pat));
lst_qc_fns = dir(fullfile(tif_dir, lst_qc_pat));
if length(lst_data_fns) ~=length(lst_qc_fns)
    disp 'FATAL ERROR.'
    return;
end

if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end
          
fid = fopen(fullfile(save_dir,stat_fn), 'w');
fprintf(fid, 'Filename_MOD11A1_LST_Day, filename_MOD11A1_LST_Night,No_GoodQuality, No_Total, GoodQ_Rate\r\n');
for i=1:length(lst_data_fns)
%for i=1:5
    lst_data_fn = lst_data_fns(i).name;
    lst_qc_fn = lst_qc_fns(i).name;
    disp(lst_data_fn);
    disp(lst_qc_fn);
    disp('..Reading LST and QC data in');
    [tif_lst,R_lst,bbox_lst]=geotiffread(fullfile(tif_dir, lst_data_fn));
    [tif_qc,R_qc,bbox_qc]=geotiffread(fullfile(tif_dir, lst_qc_fn));

    %bit flags: 0 - 需内插，65535 - no data，边界外 
    % 1，2位上 10, 或者11, 去掉，因为云或者其它因素:
    % 5, 6位上 average emissivity error > 0.04去掉, 5, 6 = 11;
    % 7, 8位上 average lst error > 3K 的去掉, 7, 8 = 11;
    b1=  bitget(tif_qc, 1); 
    b2=  bitget(tif_qc, 2); 
    b5 = bitget(tif_qc, 5); 
    b6 = bitget(tif_qc, 6);
    b7 = bitget(tif_qc, 7); 
    b8 = bitget(tif_qc, 8);
    
    % MODIS本身没有值的数据等于0
    % tif_lst(~(b1==0&b2==0)) = 0; % old version
    % Wan Z.M.(2013), Collection-6 MODIS Land Surface Temperature Products Users' Guide, 
    % b2==1, means:
    %     10 = Pixel not produced due to cloud effects 
    %     11 = Pixel not produced primarily due to reasons other than 
    %           cloud (such as ocean pixel, poor input data)
    % b5==1 & b6 ==1, means:
    %     11 = average emissivity error > 0.04
    % b7==1 & b8 ==1, means:
    %     11 = average LST error > 3K
    tif_lst(b2==1 | (b5==1 & b6 ==1) | (b7==1 & b8 ==1)) = 0; % new version
    tif_lst(tif_qc == 65535) = 65535;
    % add by tanjl
    tif_qc(tif_lst == 0) = 255;
    % =============
    no_total = length(tif_lst(tif_lst~=65535)); 
    no_goodquality = no_total - length(tif_lst(tif_lst==0)); 
    fprintf('..good quality #: %d\r',no_goodquality);
    fprintf(fid, '%s, %s, %d, %d, %f\r\n',lst_data_fn, lst_qc_fn, no_goodquality ...
        ,no_total,no_goodquality/no_total);
    
    %
    geotiffwrite(fullfile(save_dir,lst_data_fn), tif_lst, R_lst);
    %add by tanjl
    geotiffwrite(fullfile(save_dir,lst_qc_fn), tif_qc, R_qc)
    %============
    %save(fullfile(save_dir,[lst_data_fn, '.mat']), 'tif_lst');
    disp('..Processed');
    %文件里保存了 lst 数据，0表示需插值，65535表示是边界外，其余表示lst值（需换算）
end
fclose(fid);


% % Special code for 2009026, there were some outliers in moidis lst
% year = 2009;
% tif_dir = sprintf('E:/modis.lst.interp/hrb.%d.C6/lst.MOD11A1/600.661/tif',year);
% lst_data_pat = sprintf('MOD11A1_%d*.LST*.tif',year);
% lst_qc_pat= sprintf('MOD11A1_%d*.QC*.tif',year);
% 
% save_dir = sprintf('E:/modis.lst.interp/hrb.%d.C6/lst.flg.processed/MOD11A1/600.661',year);
% stat_fn = 'MOD11A1.stat.txt';
% 
% lst_data_fns = dir(fullfile(tif_dir, lst_data_pat));
% lst_qc_fns = dir(fullfile(tif_dir, lst_qc_pat));
% if length(lst_data_fns) ~=length(lst_qc_fns)
%     disp 'FATAL ERROR.'
%     return;
% end
% 
% if ~exist(save_dir, 'dir')
%     mkdir(save_dir);
% end
%           
% for i=57:57
% %for i=1:5
%     lst_data_fn = lst_data_fns(i).name;
%     lst_qc_fn = lst_qc_fns(i).name;
%     disp(lst_data_fn);
%     disp(lst_qc_fn);
%     disp('..Reading LST and QC data in');
%     [tif_lst,R_lst,bbox_lst]=geotiffread(fullfile(tif_dir, lst_data_fn));
%     [tif_qc,R_qc,bbox_qc]=geotiffread(fullfile(tif_dir, lst_qc_fn));
% 
%     %bit flags: 0 - 需内插，65535 - no data，边界外 
%     % 1，2位上 10, 或者11, 去掉，因为云或者其它因素:
%     % 5, 6位上 average emissivity error > 0.04去掉, 5, 6 = 11;
%     % 7, 8位上 average lst error > 3K 的去掉, 7, 8 = 11;
%     b1=  bitget(tif_qc, 1); 
%     b2=  bitget(tif_qc, 2); 
%     b5 = bitget(tif_qc, 5); 
%     b6 = bitget(tif_qc, 6);
%     b7 = bitget(tif_qc, 7); 
%     b8 = bitget(tif_qc, 8);
%     
%     tif_lst(b2==1 | (b5==1 & b6 ==1) | (b7==1 & b8 ==1)) = 0; % new version
%     tif_lst(tif_lst>14567)=0;
%     tif_lst(tif_lst<=12500)=0;
%     tif_lst(tif_qc == 65535) = 65535;
%     % add by tanjl
%     tif_qc(tif_lst == 0) = 255;
%     % =============
%     no_total = length(tif_lst(tif_lst~=65535)); 
%     no_goodquality = no_total - length(tif_lst(tif_lst==0)); 
%     
%     %
%     geotiffwrite(fullfile(save_dir,lst_data_fn), tif_lst, R_lst);
%     %add by tanjl
%     geotiffwrite(fullfile(save_dir,lst_qc_fn), tif_qc, R_qc)
%     %============
%     %save(fullfile(save_dir,[lst_data_fn, '.mat']), 'tif_lst');
%     disp('..Processed');
%     %文件里保存了 lst 数据，0表示需插值，65535表示是边界外，其余表示lst值（需换算）
% end

% % -------------------------------------------------------------------------
% %%% statistic MYD11A1
% tif_dir = sprintf('../../hrb.%d.C6/lst.MYD11A1/600.661/tif',year);
% lst_data_pat = sprintf('MYD11A1_%d*.LST*.tif',year);
% lst_qc_pat= sprintf('MYD11A1_%d*.QC*.tif',year);
% 
% save_dir = sprintf('../../hrb.%d.C6/lst.flg.processed/MYD11A1/600.661.V2',year);
% stat_fn = 'MYD11A1.stat.txt';
% 
% lst_data_fns = dir(fullfile(tif_dir, lst_data_pat));
% lst_qc_fns = dir(fullfile(tif_dir, lst_qc_pat));
% if length(lst_data_fns) ~=length(lst_qc_fns)
%     disp 'FATAL ERROR.'
%     return;
% end
% 
% if ~exist(save_dir, 'dir')
%     mkdir(save_dir);
% end
% 
% fid = fopen(fullfile(save_dir,stat_fn), 'w');
% fprintf(fid, 'Filename_MYD11A1_LST_Day, filename_MYD11A1_LST_Night,No_GoodQuality, No_Total, GoodQ_Rate\r\n');
% for i=1:length(lst_data_fns)
% %for i=1:5
%     lst_data_fn = lst_data_fns(i).name;
%     lst_qc_fn = lst_qc_fns(i).name;
%     disp(lst_data_fn);
%     disp(lst_qc_fn);
%     disp('..Reading LST and QC data in');
%     [tif_lst,R_lst,bbox_lst]=geotiffread(fullfile(tif_dir, lst_data_fn));
%     [tif_qc,R_qc,bbox_qc]=geotiffread(fullfile(tif_dir, lst_qc_fn));
% 
%     %bit flags: 0 - 需内插，65535 - no data，边界外 
%     % 7, 8位上 average lst error > 3K 的去掉, 7, 8 = 11;
%     % 5, 6位上 average emissivity error > 0.04去掉, 5, 6 = 11;
%     % 1，2位上 10, 或者11, 去掉，因为云或者其它因素
%     b2=  bitget(tif_qc, 2); 
%     b5 = bitget(tif_qc, 5); 
%     b6 = bitget(tif_qc, 6);
%     b7 = bitget(tif_qc, 7); 
%     b8 = bitget(tif_qc, 8);
%     
%     % tif_lst(~(b1==0&b2==0)) = 0; 
%     tif_lst(b2==1 | (b5==1 & b6 ==1) | (b7==1 & b8 ==1)) = 0; % new version
%     %tif_lst(b2==1) = 0; 
%     tif_lst(tif_qc == 65535) = 65535;
%     % add by tanjl
%     tif_qc(tif_lst == 0) = 255;
%     %=============
%     no_total = length(tif_lst(tif_lst~=65535)); 
%     no_goodquality = no_total - length(tif_lst(tif_lst==0)); 
%     fprintf('..good quality #: %d\r',no_goodquality);
%     fprintf(fid, '%s, %s, %d, %d, %f\r\n',lst_data_fn, lst_qc_fn, no_goodquality ...
%         ,no_total,no_goodquality/no_total);
%     
%     %
% %    geotiffwrite(fullfile(save_dir,lst_data_fn), bbox_lst, tif_lst, -16);
%     geotiffwrite(fullfile(save_dir,lst_data_fn), tif_lst, R_lst);
%     %add by tanjl
%     geotiffwrite(fullfile(save_dir,lst_qc_fn), tif_qc, R_qc)
%     %============
%     %save(fullfile(save_dir,[lst_data_fn, '.mat']), 'tif_lst');
%     disp('..Processed');
%     %文件里保存了 lst 数据，0表示需插值，65535表示是边界外，其余表示lst值（需换算）
% end
% fclose(fid);
