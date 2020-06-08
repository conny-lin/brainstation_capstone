# Functions and Classes for BrainStation Projects
# Conny Lin | June 5, 2020
# ------------------------------------------------------------------------------------
import os, sys, socket, time, re, glob, pickle
import pandas as pd
import numpy as np

# system handling ====================================================================
def getcomputername():
    hostname = socket.gethostname()
    hostname = hostname.split('.')
    return hostname[0]
# end -- system handling =============================================================

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

# Data processing ===============================================================
# Trinity -----------------------------------------------------------------------
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
# end -- Trinity --------------------------------------------------------------------

# Nutcracker ------------------------------------------------------------------------
def nutcracker_process_rawdata(pdata, mwtid):
    column_names_raw = np.array(['time','id','frame','persistence','area','midline',\
            'morphwidth','width','relwidth','length','rellength','aspect',\
            'relaspect','kink','curve','speed','angular','bias','persistence',\
            'dir', 'loc_x','loc_y','vel_x','vel_y','orient','crab','tap','puff',\
            'stim3','stim4'])
    column_index_keep = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,22,23,24,25]
    column_names = column_names_raw[column_index_keep]
    # load data put in data frame
    df = pd.read_csv(pdata, delimiter=' ', header=None, usecols=column_index_keep, 
                     names=column_names, dtype=np.float64, engine='c')
    # remove data before 100s
    df.drop(axis=0, index=df.index[df['time']>100], inplace=True)
    # remove nan
    df.dropna(axis=0, inplace=True)
    # add mwtid column
    df.insert(0,'mwtid', np.tile(mwtid, df.shape[0]))
    # add etoh column
    if ('/N2_400mM/' in pdata):
        df.insert(0,'etoh', np.tile(1, df.shape[0]))
    else:
        df.insert(0,'etoh', np.tile(0, df.shape[0]))
    # make sure etoh data is integer
    df['etoh'].astype(int)
    return df

def nutcracker_process_perplate(mwtpaths_db, sourcedbdir, savedbdir, overwrite=False):
    print('\n')
    # look for nutcracker files in this plate
    output_name = 'nutcracker_100s.csv'
    nutcracker_filelist = []
    df_mwt = []
    for imwt, pmwt in enumerate(mwtpaths_db):
        print(pmwt)
        # reset run status
        run_status = True
        # update output mwt path to savedbdir
        pmwt_dropbox = str.replace(pmwt, sourcedbdir, savedbdir)
        # create output path
        pdata_save_path = os.path.join(pmwt_dropbox, output_name)
        # do not run if overwrite=False and output already exist
        if not overwrite:
            if os.path.isfile(pdata_save_path):
                print(f'\t{output_name} already exist. skip')
                nutcracker_filelist.append(pdata_save_path)
                run_status = False
        # do not run if no nutcracker.*.dat
        if run_status:
            pnutcracker = glob.glob(pmwt+'/*.nutcracker.*.dat')
            if len(pnutcracker) == 0:
                run_status = False
        if run_status:
            # make storage for df
            df_store = []
            for ifile, pdata in enumerate(pnutcracker):
                print(f'\tprocessing {ifile}', end='\r')
                # get time data
                df = pd.read_csv(pdata, delimiter=' ', header=None, usecols=[0], 
                                names=['time'], dtype=np.float64, engine='c')
                # see if data has time before 100s
                if sum(df['time']<100) > 0:
                    df = nutcracker_process_rawdata(pdata, imwt)
                    # add df to storage
                    df_store.append(df)
            # combine multiple nutcracker files (just before tap and only non NAN)
            df_mwt = pd.concat(df_store, ignore_index=True)
            print(f'\n\t{df_mwt.shape[0]} rows')
            # add the file list
            nutcracker_filelist.append(pdata_save_path)
            # save csv in savedir
            df_mwt.to_csv(pdata_save_path, index=False)
            print(f'\tsaved {output_name}')
    return df_mwt, nutcracker_filelist

def nutcracker_combineall(nutcracker_filepaths):
    # load and combine nutcracker_filelist
    df_store = []
    for filepath in nutcracker_filepaths:
        df_store.append(pd.read_csv(filepath, dtype=np.float64, engine='c'))
    data = pd.concat(df_store, ignore_index=True)
    return data

def nutcracker_split_Xy(data, dir_save):
    # split X/y
    # y column
    y_column = ['etoh']
    y = data[y_column].copy()
    data.drop(columns=y_column, inplace=True)
    y.to_csv(os.path.join(dir_save, 'nutcracker_y.csv'), index=False)
    # identifier column
    identifier_column = ['id','mwtid']
    data_identifiers = data[identifier_column].copy()
    data.drop(columns=identifier_column, inplace=True)
    data_identifiers.to_csv(os.path.join(dir_save, 'nutcracker_identifier.csv'), index=False)
    # save X
    data.to_csv(os.path.join(dir_save, 'nutcracker_X.csv'), index=False)
    print('saving done')
# end -- Nutcracker -----------------------------------------------------------------

# end -- Data processing ============================================================



