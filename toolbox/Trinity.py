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
