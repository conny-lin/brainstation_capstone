import socket, os, sys

def get(hostname):
    # define hostspecific paths
    print('getting host computer specific paths')
    # set local path settings based on computer host
    if hostname == 'PFC':
        localpath = {
            'Capstone':'/Users/connylin/Dropbox/CA/ED _20200119 Brain Station Data Science Diploma/Capstone',
            'datapath':'/Users/connylin/Dropbox/CA/ED _20200119 Brain Station Data Science Diploma/Capstone/data/nutcracker_sample_1Meach.csv'}
    elif hostname == 'Angular Gyrus':
        localpath = {
            'Capstone':'/Users/connylin/Dropbox/CA/ED _20200119 Brain Station Data Science Diploma/Capstone',
            'datapath':'/Users/connylin/Dropbox/CA/ED _20200119 Brain Station Data Science Diploma/Capstone/data/nutcracker_sample_1Meach.csv'}
    else:
        assert False, 'host computer not regonized'
    return localpath
    