% run choreography to get drunkposture2.dat output for the 272 files
% missing this output
% -------------------------------------------------------------------------
% Conny Lin | May 19, 2020
% -------------------------------------------------------------------------
% add path to chormaster function and general matlab local functions
addpath(genpath('/Users/connylin/Dropbox/Code/language/matlab_general'))
addpath(genpath('/Users/connylin/Dropbox/Code/proj/rankin_lab'))
% load csv to get mwt paths
p = '/Users/connylin/Dropbox/CA/ED _20200119 Brain Station Data Science Diploma/Capstone/data/path_mwt_no_drunkposture2.csv';
T = readtable(p,'Delimiter','comma');
% get paths into cell array
pMWTS = T.mwtpath;
% run on all
[Legend,pMWTpass,pMWTfailed] = chormaster5('DrunkPosture',pMWTS);