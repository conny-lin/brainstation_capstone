# Functions and Classes for BrainStation Projects
# Conny Lin | June 5, 2020


# Data processing
class TrinityData:
    """
    Process trinity data from a path to the data
    """
    # standard column information
    columns_number_expected = 18
    columns_index_keep = [0,3,5,6,8,9,10,11,12,13,14,15,16,17]
    columns_raw = ["time", "number", "goodnumber", "speed", "speed_std", \
        "bias", "tap", "puff", "loc_x", "loc_y", "morphwidth", "midline", \
        "area", "angular", "aspect", "kink", "curve", "crab"]
    columns = columns_raw[columns_index_keep]

    def __init__(self, datapath):
        # store raw data
        self.datapath = datapath
    
    def standard_cleaning(self):
        

        

        




