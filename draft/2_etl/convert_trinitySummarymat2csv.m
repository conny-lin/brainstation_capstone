% convert trinitySummarymat2csv
% -------------------------------------------------------------------------
% Conny Lin | June 4, 2020
% -------------------------------------------------------------------------
% add path to chormaster function and general matlab local functions
addpath(genpath('/Users/connylin/Dropbox/Code/language/matlab_general'))
addpath(genpath('/Users/connylin/Dropbox/Code/proj/rankin_lab'))
% define path to MWTDB.mat
pDB = '/Volumes/COBOLT/MWT/MWTDB.mat';
pSave = '/Volumes/COBOLT/MWT/MWTDB.csv';
% load
load(pDB);

%% check if previous output exist
MWTDB_check= MWTDB.text;
MWTDB_check.trinitySummary = zeros(size(MWTDB_check.mwtid));
MWTDB_check.trinitymeta = zeros(size(MWTDB_check.mwtid));
MWTDB_check.trinitycsv = zeros(size(MWTDB_check.mwtid));

nfiles = size(MWTDB.text,1);
for i = 1:size(MWTDB.text,1)
    % reporting
    if rem(i, 20)==0
        fprintf('processing %d/%d files: \n',i,nfiles)  
    end
    % get path for mwt plate
    pmwt = MWTDB.text.mwtpath{i};
    % make output folder
    psavefolder = fileparts(strrep(ptrinity,'/Volumes/COBOLT','/Users/connylin/Dropbox/MWT/db'));
    % make expected trinitysummary path
    ptrinity = fullfile(pmwt, 'trinitySummary.mat');
    % get path for mwt plate
    pmwt = MWTDB.text.mwtpath{i};
    if exist(fullfile(pmwt, 'trinitySummary.mat')) ~= 0
        MWTDB_check.trinitySummary(i) = 1;
    end
    if exist(fullfile(psavefolder, 'trinity_meta.csv')) ~= 0
        MWTDB_check.trinitymeta(i) = 1;
    end
    if exist(fullfile(psavefolder, 'trinity.csv')) ~= 0
        MWTDB_check.trinitycsv(i) = 1;
    end
end

sum(MWTDB_check.trinitySummary)

%% reduce the database to only ones with trinity Symmary
i = MWTDB_check.trinitySummary==1;
MWTDB_process = MWTDB_check(i,:);

%% export to csv
% ERROR NOTE: 
% - 625 - Error using cat Dimensions of matrices being concatenated are not
% consistent. Did not solve this problem. Skipped it. It seems like it's
% number of characters for wormid are different. Solve it and combine data
% with worm id in it all numeric

i_start = 1;
nfiles = size(MWTDB_process,1);
for i = i_start:size(MWTDB_process,1)
    % get path for mwt plate
    pmwt = MWTDB_process.mwtpath{i};
    % make output folder
    psavefolder = fileparts(strrep(ptrinity,'/Volumes/COBOLT','/Users/connylin/Dropbox/MWT/db'));
    % make expected trinitysummary path
    ptrinity = fullfile(pmwt, 'trinitySummary.mat');
    % make expected save path
    psavearray = fullfile(psavefolder, 'trinity_idin.csv');
    % make meta save path
    psavemeta = fullfile(psavefolder, 'trinity_meta.csv');
    % reporting
    fprintf('processing %d/%d files: ',i,nfiles)

    % if psavearray or psave meta doesn't exist  and trinitySummary.mat exist
    % open .mat 
    load(ptrinity)
    % if psavemeta doesn't exist, convert meta
    % make empty row number container
    rownumber = nan(size(masterData(:,2)));
    column_number = nan(size(masterData(:,2)));
    % get row number and data column number
    for irows = 1:size(masterData,1)
        [rownumber(irows),column_number(irows)] = size(masterData{irows,2});
    end
    % check if column number matches, 
    if any(column_number~=18)
        fprintf('trinitySummary.mat file columns does not match') 
    else   

        % process meta first
        fprintf('-trinity_meta.csv-')

        % convert worm number to numeric values
        dmaster_wormid = masterData(:,1);
        wormnumber = regexpcellout(dmaster_wormid,'(?<=0*)[1-9]{1,}[0-9]{0,}','match');
        wormnumber = str2double(worm_id_out);
        % make row numbers in 
        index_array = cell(size(rownumber));
        for wormi = 1:numel(rownumber)
            index_array{wormi} = repmat(wormnumber(wormi), rownumber(wormi), 1);
        end
        index_array = cell2mat(index_array);
        % combine
        fprintf('-trinity_idin.csv-')
        a = [index_array, cell2mat(masterData(:,2))];
        % convert to mat and save to csv
        writetable(array2table(a), psavearray)
    end
    fprintf('done\n')
%     % if psavearray or psave meta doesn't exist  and trinitySummary.mat exist
%     ptrinity_exist = exist(ptrinity) ~= 0;
%     psavearray_exist = exist(psavearray) ~= 0;
%     psavemeta_exist = exist(psavemeta) ~= 0;
%     if ptrinity_exist && (~psavearray_exist || ~psavemeta_exist)
%         % open .mat 
%         load(ptrinity)
%         % if psavearray doesn't exist, convert psavearray
%         if ~psavearray_exist
%             fprintf('\t trinity.csv')
%             % convert to mat and save to csv
%             writetable(array2table(cell2mat(masterData(:,2))), psavearray)
%         end
%         % if psavemeta doesn't exist, convert meta
%         if ~psavemeta_exist
%             fprintf('\t trinity_meta.csv')
%             % convert to mat and save to csv
%             % make empty row number container
%             rownumber = nan(size(masterData(:,2)));
%             % get row number
%             for irows = 1:size(masterData,1)
%                 rownumber(irows) = size(masterData{irows,2},1);
%             end
%             % make empty table container
%             t = table;
%             % get worm number
%             t.worm_number = cell2mat(masterData(:,1));
%             t.row_number = rownumber;
%             % save matdata to csv
%             writetable(t, psavemeta)
%         end
%     else
%         fprintf('\tno need to process')
%     end
%     fprintf('\n')
end

%% save MWTDB_process in csv
writetable(MWTDB_process, fullpath(psavefolder, 'MWTDB_process_trinity.csv'))





