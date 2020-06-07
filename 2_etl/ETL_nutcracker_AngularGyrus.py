## ETL nutcracker
# Conny Lin | June 6, 2020
# transform data from raw to ML ready data in csv

# local variable setting
pCapstone = '/Users/connylin/Dropbox/CA/ED _20200119 Brain Station Data Science Diploma/Capstone/data'
pDropboxdb = '/Users/connylin/Dropbox/MWT/db'
pCobolt = '/Volumes/COBOLT'
mwtpath_csv_name_cobolt = 'mwtpath_cobolt.csv'
mwtpath_csv_name_dropbox = 'mwtpath_dropbox.csv'

# LOCAL SETTINGS (FOR ANGULAR GYRUS)
sourcedir_db = pCobolt
savedir_db = pCobolt
savedir = pCapstone
mwtpath_csv_name = mwtpath_csv_name_cobolt

# ----------------------------------------------------------------------
# import libraries
import os, sys, glob, pickle
import pandas as pd
import numpy as np
# import local functions
sys.path.insert(1, '/Users/connylin/Dropbox/CA/ED _20200119 Brain Station Data Science Diploma/Capstone/brainstation_capstone/0_lib')
import BrainStationLib as bs

# get database MWT file paths
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


# combine individual nutcracker per plate
print('\ncombining individual nutcracker data per plate. this will take a while')
_, nutcracker_filepaths = bs.nutcracker_process_perplate(mwtpaths, sourcedir_db, savedir_db)

print('\ncombining all combined nutcracker data from each plate (memory intensive!)')
data = bs.nutcracker_combineall(nutcracker_filepaths)

print('\nsplit combine data into X y data set and save to savedir (large file! 20GB expected)')
print('\there: '+savedir)
bs.nutcracker_split_Xy(data, savedir)