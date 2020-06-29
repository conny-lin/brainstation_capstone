## ETL nutcracker
# Conny Lin | June 6, 2020
# transform data from raw to ML ready data in csv
# last file ran on June 6, 2020
# /Volumes/COBOLT/MWT/20140515B_SJ_100s30x10s10s_goa1/N2_400mM/20140515_160131
# ----------------------------------------------------------------------
# import libraries
import os, sys, socket, glob, pickle
import pandas as pd
import numpy as np

# add python modules paths based on computer hostname
import socket
hostname = socket.gethostname().split('.')[0]
if hostname == 'Angular-Gyrus':
    path_py_library = '/Users/connylin/Code/proj/brainstation_capstone'
    if path_py_library not in sys.path:
        sys.path.insert(1, path_py_library)

# get MWT database
from toolbox.database import getMWTdb
mwtpaths = getMWTdb('/Users/connylin/Dropbox/MWT/db', 'mwtpath_dropbox.csv')

# combine individual animal's data within a plate
from toolbox.datatransform import nutcracker_process_perplate
print('\ncombining individual nutcracker data per plate. this will take a while')
nutcracker_filepaths = nutcracker_process_perplate(mwtpaths, sourcedir_db, savedir_db)

# combine data across plates
print(f'\ncombining {len(nutcracker_filepaths)} plates of nutcracker data (memory intensive!)')
from toolbox.datatransform import nutcracker_combineall
data = nutcracker_combineall(nutcracker_filepaths)

# split into x and y data
print('\nsplit combine data into X y data set and save to savedir (large file! 20GB expected)')
print('\there: '+savedir)
from toolbox.datatransform import nutcracker_split_Xy
nutcracker_split_Xy(data, savedir)
