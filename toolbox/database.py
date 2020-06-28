import os, sys, socket, time, re, glob, pickle
import pandas as pd
import numpy as np

# Database handling ==================================================================
# MWT functions ----------------------------------------------------------------------
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

def makeMWTDB(pMWT):
    # parse by
    pMWT_parsed = parse_pMWT_paths(pMWT)
    # transform to dataframe and add column names
    MWTDB_pMWT = pd.DataFrame({'mwtpath': pMWT})
    MWTDB_pMWT['expname'] = pMWT_parsed[:,-3]
    MWTDB_pMWT['groupname'] = pMWT_parsed[:,-2]
    MWTDB_pMWT['platename'] = pMWT_parsed[:,-1]
    return MWTDB_pMWT

def updateMWTDB(path_dbcsv, path_db, search_param='structured', addfullpath=False):
    pMWT_found = searchMWTplates(path_db, search_param)
    MWTDB = loadMWTDB(path_dbcsv)
    pMWT_db = MWTDB_makepath2plates(MWTDB, path_db)
    pMWT_new = compare_pMWT_list(pMWT_found, pMWT_db)
    if len(pMWT_new)>0:
        MWTDB_new = makeMWTDB(pMWT_new)
        MWTDB_append = MWTDB.append(MWTDB_new, ignore_index=True)
        MWTDB_append.to_csv(path_dbcsv)
        print('new MWTDB updated')
    else:
        print('nothing new to update')
        MWTDB_append = []
    return MWTDB_append

def make_chor_output_legend(chorlegendpath, chorjavacall):
    legend_chor = pd.read_csv(chorlegendpath)
    # create drunk moves .dat legend
    javacallletters = []
    for letters in chorjavacall:
        javacallletters.append(letters)
    column_names = pd.DataFrame(javacallletters,columns=['call'])
    column_names = column_names.merge(legend_chor,how='left', on='call')
    return column_names

# MWTDB class ------------------------------------------------------------------
class MWTDB:
    def __init__(self, path_input):
        self.path_input = path_input

# end - MWT functions -----------------------------------------------------------
# end - Database handling =======================================================
