% export most recent MWTDB to csv
% -------------------------------------------------------------------------
% Conny Lin | May 20, 2020
% -------------------------------------------------------------------------
% add path to chormaster function and general matlab local functions
addpath(genpath('/Users/connylin/Dropbox/Code/language/matlab_general'))
addpath(genpath('/Users/connylin/Dropbox/Code/proj/rankin_lab'))
% define path to MWTDB.mat
pDB = '/Volumes/COBOLT/MWT/MWTDB.mat';
pSave = '/Volumes/COBOLT/MWT/MWTDB.csv';
% load
load(pDB);
% export to csv
writetable(MWTDB.text, pSave)

