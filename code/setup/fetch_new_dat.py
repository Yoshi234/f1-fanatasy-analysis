import pandas as pd
try:
    from get_track_augment import get_track_speeds
except ImportError:
    from .get_track_augment import get_track_speeds
# from fastf1 import SessionNotAvailableError

def fetch_new():
    '''
    get data for all events which have occurred
    up until now
    '''
    # old_dat = pd.read_feather("../../data/clean_model_data.feather")
    # print("[INFO]: old_dat.keys = {}".format(old_dat.keys()))
    
    # track_dat = pd.read_feather("../../data/track_data.feather")
    # print("[INFO]: track_dat.keys = {}".format(track_dat.keys()))
    
    results = pd.read_csv("../../data/results.csv")
    print("[INFO]: results.keys() = {}".format(results.keys()))
    x1 = results.loc[results['raceId']==18]
    print(f"[INFO]: ranks = {x1['rank'].unique()}")
    
if __name__ == "__main__":
    fetch_new()