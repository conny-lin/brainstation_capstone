% run choreography to get drunkposture2.dat output for the 272 files
% missing this output
% -------------------------------------------------------------------------
% Conny Lin | May 19, 2020
% -------------------------------------------------------------------------
% add path to chormaster function and general matlab local functions
computername = 'Angular Gyrus';
switch computername
    case 'PFC'
        homepath = '/Users/connylin/Dropbox/Code';
    case 'Angular Gyrus'
        homepath = '/Users/connylin/Code';
end
addpath(genpath(fullfile(homepath, 'language/matlab_general')))
addpath(genpath(fullfile(homepath, 'proj/rankin_lab')))

%% load csv
% load csv to get mwt paths
p = '/Volumes/COBOLT/MWT/MWTDB.csv';
T = readtable(p,'Delimiter','comma');

%% get only ISI=10, preplate=100, N2, N2_400mM
T(~(strcmp(T.groupname, 'N2') | strcmp(T.groupname, 'N2_400mM')),:) = [];
T(~strcmp(T.rc, '100s30x10s10s'),:) = [];
%% store this in 
psave = '/Users/connylin/Dropbox/CA/ED _20200119 Brain Station Data Science Diploma/Capstone/data/MWTDB_N240010sISI.csv';
writetable(T, psave);
pMWTS = T.mwtpath;

%% run on all
[Legend,pMWTpass,pMWTfailed] = chormaster5('Nutcracker',pMWTS);