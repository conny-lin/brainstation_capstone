# ML on nutcracker
# Conny Lin | June 6, 2020
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# local variable definition
pCapstone = '/Users/connylin/Dropbox/CA/ED _20200119 Brain Station Data Science Diploma/Capstone/data'
pylibrary = '/Users/connylin/Dropbox/CA/ED _20200119 Brain Station Data Science Diploma/Capstone/brainstation_capstone/0_lib'
X_filename = 'nutcracker_X.csv'
y_filename = 'nutcracker_y.csv'
fig_accuracy_name = 'nutcracker_decisiontree_acc.png'

# import libraries
import os, sys, glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
# import local functions
sys.path.insert(1, pylibrary)
import BrainStationLib as bs

# load data
print(f'loading data {y_filename}')
y = pd.read_csv( os.path.join(pCapstone, y_filename))
print(f'\t{y.shape[0]} rows {y.shape[1]} columns')
print(f'loading data {X_filename}')
X = pd.read_csv( os.path.join(pCapstone, X_filename))
print(f'\t{X.shape[0]} rows {X.shape[1]} columns')

print('transforming data into machine learning dataframe')
# store column names
y_columns = y.columns.values
X_columns = X.columns.values

# transform data into ML dataframe
X = X.values
y = y.values.astype(int).transpose()
y = y[0]

# split data
print('split data into test and train')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)

# scale data
print('scale data')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# decision tree
print('run decision tree')
from sklearn.tree import DecisionTreeClassifier
# Decision Tree -  data
train_accs = []
test_accs = []
depth_values = list(range(1,15))

# Loop over different max_depths
print('\tfit different depths')
for d in depth_values:
    print(f'\t\tfitting {d} depth', end='\r')
    # Instantiate & fit
    my_dt = DecisionTreeClassifier(max_depth = d, random_state=318)
    my_dt.fit(X_train_scaled, y_train)
    # Evaluate on train & test data
    train_accs.append( my_dt.score(X_train_scaled, y_train) )
    test_accs.append( my_dt.score(X_test_scaled, y_test) )
print('\n')
# Plot the results
print('making graphs')
plt.figure()
plt.plot(depth_values, train_accs, label='train')
plt.plot(depth_values, test_accs, label='test')
plt.legend()
plt.xlabel('max depth')
plt.ylabel('accuracy')
plt.title('DecisionTree Accuracy')
plt.savefig( os.path.join(pCapstone, fig_accuracy_name))

print('complete')