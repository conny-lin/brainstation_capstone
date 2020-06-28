import os, socket
import pandas as pd
import numpy as np

class Nutcracker:
    pCapstone = '/Users/connylin/Dropbox/CA/ED _20200119 Brain Station Data Science Diploma/Capstone/data'
    datapath = os.path.join(pCapstone, 'nutcracker_sample_1Meach.csv')
    names = {
            'y':['etoh'],
            'identifier': ['id','mwtid','frame'],
            'X':['time', 'persistence', 'area', 'midline', 'morphwidth',
                'width', 'relwidth', 'length', 'rellength', 'aspect', 'relaspect',
                'kink', 'curve', 'speed', 'angular', 'bias', 'dir', 'vel_x',
                'vel_y', 'orient', 'crab']
            }
    top_features = {'bycategory': ['speed', 'aspect', 'width', 'rellength', 'dir', 'area']}

    def __init__(self, datapath=datapath):
        # update user input datapath
        self.datapath = datapath

    def loaddata(self):
        self.data = pd.read_csv(self.datapath)
        return self.data

    def reduce_feature(self, feature_reduction):
        print(f'feature reduction method: {feature_reduction}')
        if feature_reduction == 'standard':
            self.names['X'] = ['area', 'midline', 'morphwidth', 'width', 'relwidth', 
                'length', 'rellength', 'aspect', 'relaspect', 'kink', 'curve', 'speed', 'angular', 
                'bias', 'dir', 'vel_x', 'vel_y', 'crab']
        elif feature_reduction == 'keep_identifier':
            self.names['X'] = ['id','mwtid','frame','area', 'midline', 'morphwidth', 'width', 'relwidth', 
                'length', 'rellength', 'aspect', 'relaspect', 'kink', 'curve', 'speed', 'angular', 
                'bias', 'dir', 'vel_x', 'vel_y', 'crab']
        elif feature_reduction == 'None':
            pass
        
    def transform(self):
        # prepare y data
        y = self.data[self.names['y']].values
        y = y.transpose()
        y = y[0]
        self.y = y
        # prepare X data
        X = self.data[self.names['X']].values
        self.X = X
        return X, y
    
    def split_test_train(self, test_size = 0.2, random_state = 318):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
                                                            test_size=test_size, 
                                                            random_state=random_state)
        self.X_train = X_train 
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        return X_train, X_test, y_train, y_test
    
    def scaledata(self):
        # scaled data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train_scaled = scaler.transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        return self.X_train_scaled, self.X_test_scaled
    
    def transform_full(self, **kwargs):
        random_state = kwargs.pop('random_state', 318)
        test_size = kwargs.pop('test_size', 0.2)
        feature_reduction = kwargs.pop('feature', 'None')
        self.loaddata()
        self.reduce_feature(feature_reduction)
        self.transform()
        self.split_test_train(test_size, random_state)
        self.scaledata()
        # create dictionary
        transform_dict = {'X_train': self.X_train,
                            'X_test': self.X_test, 
                            'y_train': self.y_train, 
                            'y_test':self.y_test}
        return transform_dict

    def mldata(self, **kwargs):
        # process kwargs
        random_state = kwargs.pop('random_state', 318)
        test_size = kwargs.pop('test_size', 0.2)
        feature_reduction = kwargs.pop('feature_reduction', 'None')
        print(f'{feature_reduction}')
        self.loaddata()
        self.reduce_feature(feature_reduction)
        self.transform()
        self.split_test_train(test_size, random_state)
        self.scaledata()
        # create dictionary
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

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

# -----------------------------------------------------------------------
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
            if len(df_store)==0:
                print('\tno nutcracker.*.dat loaded')
            elif len(df_store)==1:
                print('\tonly one nutcracker.*.dat loaded')
                print('\texclude this plate')
            else:
                print('\tconcat multiple nutcrakcer.*.dat')
                # combine multiple nutcracker files (just before tap and only non NAN)
                df_mwt = pd.concat(df_store, ignore_index=True)
                print(f'\t{df_mwt.shape[0]} rows')
                # save csv in savedir
                df_mwt.to_csv(pdata_save_path, index=False)
                print(f'\tsaved {output_name}')
                # add the file list
                nutcracker_filelist.append(pdata_save_path)
    return nutcracker_filelist

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
   