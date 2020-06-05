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

%% export to csv
nfiles = size(MWTDB.text,1);
for i = 1:size(MWTDB.text,1)
    pmwt = MWTDB.text.mwtpath{i};
    ptrinity = fullfile(pmwt, 'trinitySummary.mat');
    psavefolder = fileparts(strrep(ptrinity,'/Volumes/COBOLT','/Users/connylin/Dropbox/MWT/db'));
    fprintf('processing %d/%d files: ',i,nfiles)
    t = table;
    rownumber = nan(size(masterData(:,2)));
    if exist(ptrinity) == 2
        fprintf('trinity exists\n')
        load(ptrinity)
        % convert to matrix
        d = array2table(cell2mat(masterData(:,2)));
        % save table
        psavearray = fullfile(psavefolder, 'trinity.csv');
        % save csv
        writetable(d,psavearray)

        % make cell array of worm id
        % get worm number
        t.worm_number = cell2mat(masterData(:,1));
        % get row number
        for irows = 1:size(masterData,1)
            rownumber(irows) = size(masterData{irows,2},1);
        end
        t.row_number = rownumber;
        % save metadata
        psavemeta = fullfile(psavefolder, 'trinity_meta.csv');
        % save csv
        writetable(t,psavemeta) 
    else
        fprintf('no trinity\n')

    end
end






