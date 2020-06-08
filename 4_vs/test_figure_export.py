# test figure export from matplotlib
# Conny Lin | June 6, 2020
# ----------------------------------------------------------------------------
# code works.
# ----------------------------------------------------------------------------

# local var
pCapstone = '/Users/connylin/Dropbox/CA/ED _20200119 Brain Station Data Science Diploma/Capstone/data'
pylibrary = '/Users/connylin/Dropbox/CA/ED _20200119 Brain Station Data Science Diploma/Capstone/brainstation_capstone/0_lib'
X_filename = 'nutcracker_y_sample.csv'
y_filename = 'nutcracker_X_sample.csv'
figsavename = 'test.pdf'

import os, sys, glob
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt

# define data
depth_values = list(range(1,15))

train_accs_s = [0.8277941841910976,
 0.8290742813594036,
 0.8745266617447427,
 0.8904891757365024,
 0.9056419538450546,
 0.9259984757912787,
 0.9421455619328872,
 0.9543035080616352,
 0.9658005668151183,
 0.9759818047583891,
 0.9831920264831265,
 0.9880325799614185,
 0.9917061611374408,
 0.9942961251756413]

test_accs_s = [0.824233836236837,
    0.8284015448306521,
    0.8746353254980411,
    0.8888472118029508,
    0.9036842543969326,
    0.9232585924258843,
    0.938998638548526,
    0.9498346808924454,
    0.9604484454446945,
    0.9698257897807785,
    0.9756189047261815,
    0.9788280403434192,
    0.9806062626767803,
    0.9818287905309661]


## graph
plt.figure()
plt.plot(depth_values, train_accs_s, label='train')
plt.plot(depth_values, test_accs_s, label='test')
plt.legend()
plt.xlabel('max depth')
plt.ylabel('accuracy')
plt.title('DecisionTree Accuracy')
# plt.show()

psavepath = os.path.join(pCapstone, figsavename)
plt.savefig(psavepath, transparent=True)
print(f'\nsaved figure in \n\t{psavepath}')