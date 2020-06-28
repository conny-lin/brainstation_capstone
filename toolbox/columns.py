import numpy as np

def nutcracker_get_variablenames():
    column_y = ['etoh']
    column_id = ['id','mwtid','frame']
    column_x = ['time', 'persistence', 'area', 'midline', 'morphwidth',
        'width', 'relwidth', 'length', 'rellength', 'aspect', 'relaspect',
        'kink', 'curve', 'speed', 'angular', 'bias', 'dir', 'vel_x',
        'vel_y', 'orient', 'crab']
    return column_names, y_column, id_column
