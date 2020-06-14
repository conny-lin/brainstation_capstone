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

# local variable settings
# check which computer this code is running on
hostname = socket.gethostname()
hostname = hostname.split('.')
hostname = hostname[0]
print(hostname)

# set local path settings based on computer host
if hostname == 'PFC':
    savedir_db = '/Users/connylin/Dropbox/MWT/db'
    mwtpath_csv_name = 'mwtpath_dropbox.csv'
    savedir = '/Users/connylin/Dropbox/CA/ED _20200119 Brain Station Data Science Diploma/Capstone/data'
    pylibrary = '/Users/connylin/Dropbox/Code/proj/brainstation_capstone/0_lib'
    sourcedir_db = '/Volumes/COBOLT'

elif hostname == 'Angular-Gyrus':
    savedir_db = '/Volumes/COBOLT'
    mwtpath_csv_name = 'mwtpath_cobolt.csv'
    savedir = '/Users/connylin/Dropbox/CA/ED _20200119 Brain Station Data Science Diploma/Capstone/data'
    pylibrary = '/Users/connylin/Code/proj/brainstation_capstone/0_lib'
    sourcedir_db = '/Volumes/COBOLT'

else:
    assert False, 'host computer not regonized'

# import local functions
sys.path.insert(1, pylibrary)
import BrainStationLib as bs

# get database MWT file paths
def getMWTdb(savedir, mwtpath_csv_name):
    pathcsv = os.path.join(savedir, mwtpath_csv_name)
    if os.path.isfile(pathcsv):
        print(f'loading mwtpath.csv from \n\t{pathcsv}')
        df = pd.read_csv(pathcsv)
        mwtpaths = df['mwtpath'].values
    else:
        mwtpaths = glob.glob(sourcedir_db+'/*/*/*/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]')
        print(f'saving mwtpath found to \n\t{pathcsv}')
        df = pd.DataFrame({'mwtpath':mwtpaths})
        df.to_csv(pathcsv)
    print(f'{len(mwtpaths)} mwt folders found')
    return mwtpaths

mwtpaths = getMWTdb(savedir, mwtpath_csv_name)

# combine individual nutcracker per plate
print('\ncombining individual nutcracker data per plate. this will take a while')
nutcracker_filepaths = bs.nutcracker_process_perplate(mwtpaths, sourcedir_db, savedir_db)

print(f'\ncombining {len(nutcracker_filepaths)} plates of nutcracker data (memory intensive!)')
data = bs.nutcracker_combineall(nutcracker_filepaths)

print('\nsplit combine data into X y data set and save to savedir (large file! 20GB expected)')
print('\there: '+savedir)
bs.nutcracker_split_Xy(data, savedir)
