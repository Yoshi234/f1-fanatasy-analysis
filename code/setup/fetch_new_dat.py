'''
races.csv: 
    + circuitId, raceId
        + circuitId - cross reference circuits.csv
        + raceId - just make it the max raceId in clean model data +1
            and increment this value for all new races added
    + round, event_name <- name
    + year
weather api
    + alt, lng, lat, location, country
    + tempmax, tempmin, temp, dew, humidity, 
    + precip, precipprob, precipcover, preciptype, 
    + windspeed, winddir, visibility
results.csv
    + resultId, raceId, driverId, constructorId, number, grid
    + position, positionText, positionOrder (Final race outcome position)
    + points (points scored), laps (laps completed), time, milliseconds,
    + fastestLap, rank, fastestLapTime, fastestLapSpeed, statusId
track_data.feather
    + all track data [ref_year, ref_name, circuitId, strt_len_mean, ..., circuit_len]
regulations.csv
'''

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
    old_dat = pd.read_feather("../../data/clean_model_data.feather")
    print("[INFO]: old_dat.keys = {}".format(old_dat.keys()))
    
    track_dat = pd.read_feather("../../data/track_data.feather")
    print("[INFO]: track_dat.keys = {}".format(track_dat.keys()))
    
    results = pd.read_csv("../../data/results.csv")
    print("[INFO]: results.keys() = {}".format(results.keys()))
    x1 = results.loc[results['raceId']==18]
    print(f"[INFO]: ranks = {x1['rank'].unique()}")

    weather_dat = pd.read_feather("../../data/races_circuits_weather.feather")
    print(f"[INFO]: weather dat keys = {weather_dat.keys()}")
    
if __name__ == "__main__":
    fetch_new()