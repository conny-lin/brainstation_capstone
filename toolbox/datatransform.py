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

   