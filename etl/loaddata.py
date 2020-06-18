# collection of load data for this project
import os
import pandas as pd
import numpy as np

def nutcracker(localpaths, dataname, datatype):
    """
    datatype = ['X_train','X_test','y_train','y_test']
    dataname = 'nutcracker'

    """
    dir_datafolder = os.path.join(localpaths['Capstone'], 'data')

    datadict = dict()
    for dname in datatype:
        filename = dataname + '_' + dname + '.csv'
        filepath = os.path.join(dir_datafolder, filename)
        data = pd.read_csv(filepath, header=None, index_col=False)
        datadict[dname] = data.to_numpy()
    return datadict