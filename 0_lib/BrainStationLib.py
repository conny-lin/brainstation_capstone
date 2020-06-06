# Functions and Classes for BrainStation Projects
# Conny Lin | June 5, 2020

import os
import sys
import time
import re
import glob
import pandas as pd
import numpy as np

# Data processing
class TrinityData:
    """
    Process trinity data from a path of the data

    Paramters
    ---------
    dtype: string of a filepath

    Returns
    -------
    object with patha and default column values
    """
    # standard column information
    columns_number_expected = 18
    columns_index_keep = np.array([0,3,5,6,8,9,10,11,12,13,14,15,16,17], 
                                    dtype='int')
    columns_raw = np.array(["time", "number", "goodnumber", "speed", "speed_std", \
        "bias", "tap", "puff", "loc_x", "loc_y", "morphwidth", "midline", \
        "area", "angular", "aspect", "kink", "curve", "crab"],
        dtype='object')
    columns_standard = np.array(["mwtid", "ethanol", "worm_id", "time", "number",\
        "goodnumber", "speed", "speed_std", "bias", "tap", "puff", "loc_x", \
        "loc_y", "morphwidth", "midline", "area", "angular", "aspect", "kink", \
        "curve", "crab"],
        dtype='object')
    

    def __init__(self, datapath, loaddata=True, reduce=True):
        self.datapath = datapath
    

    def loaddata(self, standard=True):
        """
        Load data from the path 

        Parameters
        ---------
        self.datapath : path to the trinity file
        standard : bool, default True, which will keep the columns defined under
            column_index_keep. If False then will keep all columns
        """
        # load csv file (made by matlab, which has a1, a2, ... as header)
        if standard:
            self.data = pd.read_csv(self.datapath, usecols=self.columns_index_keep, 
                    names=self.columns_raw[self.columns_index_keep],
                    engine='c', skiprows=1, dtype='float64')
        else:
            self.data = pd.read_csv(self.datapath, names=self.columns_raw,
                    dtype='float64')
                #     df = pd.read_csv(self.datapath, usecols=col_ind_keep, 
                #     delim_whitespace=True, 
                #  header=None, names=v.loc[col_ind_keep,'name'].values)
        return self.data

    
    def pathcomponents(self,order='ascending'):
        # split components
        self.path_components = self.datapath.split('/')
        # reverse order if requested
        if order == 'descending':
            self.path_components.reverse()
        return self.path_components
        

    def addetohlabel(self, addposition=0):
        path_components = self.pathcomponents(order='descending')
        # find key word for etoh
        etoh_label_exist = '00mM' in path_components[2]
        # create numpy array the same size as data rows in integer
        d = np.tile(etoh_label_exist, self.data.shape[0]).astype('int')
        # insert into first row of data frame
        self.data.insert(addposition, 'ethanol', d)
        return self.data



# MWT functions ---------------------------------------------------------------
def searchMWTplates(path_input, search_param='structured'):

    """
    search for MWT plates in subdirectories of a given directory

    Parameter
    --------
    path_input : directory path
    search_param : will search for MWT plates name formatted in this way: 
        20180801_210102. Default "structured" search in a structured path: 
        db/expname/groupname/mwt_plate. "unstructured" will search for any 
        level of subdirectories and will take a lot longer.
    
    """
    # check inputs
    assert search_param in ['structured', 'unstructured'], \
                'search_param must be "structured" or "unstructured"'
    assert os.path.isdir(path_input), 'path_input must be a directory'
    # get all the mwt plate paths
    # example MWT/20140819B_SM_100s30x10s10s/N2/20140108_210021
    print(f'Searching for MWT folders in dir: {path_input}\n')
    print('\tThis will take a while...')
    plate_syntax = '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]'
    if search_param == 'structured':
        pMWT_found = glob.glob(path_input+'/*/*/*/'+plate_syntax)
    elif search_param == 'unstructured': 
        # this code will search all directories, but takes a long time
        os.chdir(path_input)
        pMWT_found = glob.glob('**/'+plate_syntax, recursive=True)
    print('\t\tdone')
    pMWT_found = np.array(pMWT_found)
    print(f'\t\t{pMWT_found.shape[0]} MWT folders found')
    return pMWT_found


def loadMWTDB(path_dbcsv):
    if os.path.isfile(path_dbcsv):
        print('load existing MWTDB.csv')
        # load the existing db
        MWTDB = pd.read_csv(path_dbcsv, index_col='mwtid')
    else:
        print('MWTDB.csv does not exist')
    return MWTDB

def parse_pMWT_paths(pMWT):
    pMWT_parsed = np.array(list(map(lambda x: x.split('/'), pMWT)))
    return pMWT_parsed

def MWTDB_makepath2plates(MWTDB, path_db=''):
    # reconstruct path from db
    path_columns = ['db','expname','groupname','platename']
    pMWT_db = MWTDB.loc[:,path_columns].apply(\
                lambda x: os.path.join(path_db,'/'.join(x.astype(str).values)), axis=1).values
    print(f'{MWTDB.shape[0]} MWT files found in MWTDB.csv')
    return pMWT_db
    
def compare_pMWT_list(pMWT_found, pMWT_db):
    '''
    inputs must be both numpy arrays
    '''
    pMWT_new = np.setdiff1d(pMWT_found, pMWT_db)
    return pMWT_new

def makeMWTDB(pMWT, addfullpath=True):
    # parse by
    pMWT_parsed = parse_pMWT_paths(pMWT)
    # define column names
    columns7 = ['h','v','drive','db','expname','groupname','platename']
    columns4 = ['db','expname','groupname','platename']
    # transform to dataframe and add column names
    if pMWT_parsed.shape[1]==7:
        MWTDB_pMWT = pd.DataFrame(pMWT_parsed, columns=columns7)
        MWTDB_pMWT.drop(columns=['h','v','drive'], inplace=True)
    elif pMWT_parsed.shape[1]==4:
        MWTDB_pMWT = pd.DataFrame(pMWT_parsed, columns=columns4)
    MWTDB_pMWT.index.name = 'mwtid'
    # add full path
    if addfullpath:
        MWTDB_pMWT.insert(0,'mwtpath', pMWT)
    return MWTDB_pMWT

def updateMWTDB(path_dbcsv, path_db, search_param='structured', addfullpath=False):
    pMWT_found = searchMWTplates(path_db, search_param)
    MWTDB = loadMWTDB(path_dbcsv)
    pMWT_db = MWTDB_makepath2plates(MWTDB, path_db)
    pMWT_new = compare_pMWT_list(pMWT_found, pMWT_db)
    if len(pMWT_new)>0:
        MWTDB_new = makeMWTDB(pMWT_new, addfullpath)
        MWTDB_append = MWTDB.append(MWTDB_new, ignore_index=True)
        MWTDB_append.to_csv(path_dbcsv)
        print('new MWTDB updated')
    else:
        print('nothing new to update')
    return MWTDB_append


# end - MWT functions -----------------------------------------------------------


# MWTDB class ------------------------------------------------------------------
class MWTDB:
    def __init__(self, path_input):
        self.path_input = path_input