
import numpy as np
import pandas as pd

def get_chor_legend(type='table'):
    chor_legend = [ 't time -- time of the frame',
        'f frame -- the frame number ',
        'p persistence -- length of time object is tracked ',
        'D id -- the object ID ',
        'n number -- the number of objects tracked ',
        'N goodnumber -- the number of objects passing the criteria given ',
        'e area -- body area',
        'm midline -- length measured along the curve of object ',
        'M morphwidth -- mean width of body about midline ',
        'w width -- width of the rectangle framing the body',
        'W relwidth -- instantaneous width/average width ',
        'l length -- measured along major axis, not curve of object ',
        'L rellength -- instantaneous length/average length',
        'a aspect -- length/width ',
        'A relaspect -- instantaneous aspect/average aspect',
        'k kink -- head/tail angle difference from body (in degrees) ',
        'c curve -- average angle (in degrees) between body split into 5 segments ',
        's speed -- speed of movement',
        'S angular -- angular speed',
        'b bias -- fractional excess of time spent moving one way',
        'P pathlen -- distance traveled forwards (backwards=negative) ',
        'd dir -- consistency of direction of motion ',
        'x loc_x -- x coordinate of object (mm) ',
        'y loc_y -- y coordinate of object (mm) ',
        'u vel_x -- x velocity (mm/sec) ',
        'v vel_y -- y velocity (mm/sec) ',
        'o orient -- orientation of body (degrees, only guaranteed modulo pi) ',
        'r crab -- speed perpendicular to body orientation ',
        '1 tap -- whether a tap (stimulus 1) has occurred ',
        '2 puff -- whether a puff (stimulus 2) has occurred',
        '3 stim3 -- whether the first custom stimulus has occurred ',
        '4 stim4 -- whether the second custom stimulus has occurred. ',
        '^ :max -- maximum value ',
        '_ :min -- minimum value ',
        '# :number -- number of items considered in this statistic ',
        '- :median -- median value ',
        '* :std -- standard deviation ',
        ':sem :sem -- standard error ',
        ':var :var -- variance ',
        ':? :exists -- 1 if the value exists, 0 otherwise ',
        ':p25 :p25 -- 25th percentile ',
        ':p75 :p75 -- 75th percentile ',
        ':jitter :jitter -- estimate of measurement precision ']

    if type == 'raw':
        return chor_legend
    elif type == 'table':
        # convert to table
        df = pd.DataFrame({'ref':chor_legend})
        df_split = df['ref'].str.split(r'\s',expand=True)
        chor_table = df_split.iloc[:,:2].copy()
        chor_table.columns=['call','name']
        # add type: stats or measure. stats name start with :
        chor_table['type'] = np.tile('measure',chor_table.shape[0])
        # where name starts with :
        i = chor_table['name'].str.contains(pat=':').values
        chor_table.loc[i, 'type'] = 'stats'
        # get description. split by
        description_split = df['ref'].str.split(r'--',expand=True)
        chor_table['description'] = description_split.iloc[:,1]
        return chor_table
