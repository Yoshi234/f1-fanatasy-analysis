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
    + points (points scored), laps (laps completed), time in milliseconds,
    + fastestLap, rank, fastestLapTime, fastestLapSpeed, statusId
standings data
    + prev_round, prev_driver_points, prev_driver_position
    + prev_driver_wins, prev_construct_wins, prev_construct_position
    + prev_construct_wins
track_data.feather
    + all track data [ref_year, ref_name, circuitId, strt_len_mean, ..., circuit_len]
regulations.csv [EXCLUDE - set nan]
    + regulation_id, year, engine_reg (bin), tire_reg (bin), aero_reg (bin)
    + chastech_reg (bin), sporting_reg (bin), pitstop_reg (bin), years_since_major_cycle
    + is_major_reg, cycle

Algorithm: 
1. call the fastf1 calendar api for 2024 and subset races where date <= current_date
2. for each race in subset (2024):
    a. get circuit name - cross-reference circuit ID from circuits.csv
    b. raceId - get max raceId in clean_model_data.feaather and add 1 
       (if the first new record). Otherwise, make it plus the new max - save 
       as an additional variable
    c. round = 1 for the first record, 2 for the second, and so on - increment
       manually
    d. event_name - the race name given (australian grand prix, etc.)
    e. year = 2024 for all races in the year <- set the year from current date (we
       want to fetch races from the current year, no matter when the program is 
       run)
    f. run get_weather to all of the weather data
    g. set a new resultId for each of the results from a given race 
       - for first race of season - increment from the max resultId from 
         clean_model_data. 
       - else: increment from the current max resultId
    h. cross reference driverId's from drivers.csv
    i. cross reference constructor Ids from constructors.csv
    j. cross-ref driver number from drivers.csv
    k. fastf1 - fetch data
       - starting grid position for each driver (race session)
         (get from driver results data - grid position)
       - position - fetch finishing position for the race session
         (get from the result data for race session)
         * leave positionText and position as nan values
       - points - get from the result of the session
       - get fastest lap time for each driver
         * fastestlap - lap number from pick_drivers
         * fastestlaptime - lap time of the fastest lap
         * rank - did the driver ID match pick_fastest() for all laps ID?
           1 if yes, 0 otherwise
         * fastest lap speed - use SpeedFL attribute from pick_fastest() for
           given driver
       - status - get from results
    l. track data - call get_track_speeds and concatenate the 
       summary data to the record
    m. regulation data - exclude (set nan for each column here)
    n. prev_round_data - manually define this from the data
       which is gathered unless data is available from 
       the most recent race of the year
       (i.e. if race 2024 in raceyears.unique(), then 
       add the next race in sequence to the round and use
       that data to calculate the new standings information)
'''
from datetime import datetime
import pandas as pd
import fastf1
try:
    from get_weather_dat import get_weather_dat
    from get_track_augment import get_track_speeds
except ImportError:
    from .get_weather_dat import get_weather_dat
    from .get_track_augment import get_track_speeds
# from fastf1 import SessionNotAvailableError

def see_keys():
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

    regulations = pd.read_csv("../../data/regulations.csv")
    print(f"[INFO]: regulations keys = {regulations.keys()}")
    
def fetch_new(current_date=None):
    '''
    fetches most recent data for records not contained clean_model_data.feather
    generates matching dataframe and updates existing clean_model_data.feather
    file with additional data
    '''
    # read cross-ref data
    drivers = pd.read_csv("../../data/drivers.csv")
    circuits = pd.read_csv("../../data/circuits.csv")
    og_dat = pd.read_feather('../../data/clean_model_data.feather')
    
    # get current datetime
    if current_date is None:
        current_date = datetime.now()
        curr_yr = current_date.year
    else:
        current_date = datetime.strptime(current_date, "%Y-%m-%d")
        curr_yr = current_date.year

    if curr_yr in og_dat['year'].unique(): # get the most recent recorded round
        prev_round = og_dat.loc[(og_dat['round']==og_dat.loc[og_dat['year']==2023]['round'].max()) &
                                (og_dat['year']==curr_yr)]
    else: prev_round = None # set initially to None for standings and such

    # schedule includes the round data - check against the most
    # recent round to see which one we start from
    schedule = fastf1.get_event_schedule(curr_yr)
    if prev_round == None: p_r = 0
    else: p_r = prev_round['round'].unique()[0]

    # update schedule
    schedule = schedule.loc[(schedule['RoundNumber']>0) & 
                            (schedule['EventDate']<current_date)]

    # iterate over schedule and fill records for each race
    for idx, record in schedule.iterrows():
        ### set dataframe on driver specific features
        ### and then set the universal features

        ### universal ..................................
        # get circuit id
        l1 = record['Location']
        c1 = circuits.loc[circuits['location']==l1]

        # set race id
        if prev_round is None:
            r1_id  = og_dat['raceId'].max() + 1
        else:
            r1_id = prev_round['raceId']
        
        # set round value
        prev_round = record['RoundNumber'] - 1
        
        # set event name
        ### end universal variables ...................

        # set base dataframe
        s1 = fastf1.get_session(curr_yr, record['RoundNumber'], 'R')
        base = s1.results
        base = base.rename({'GridPosition':'grid',
                            'Position':'positionOrder',
                            'Points':'points',})
        for idy, row in base.iterrows():
            driver = drivers.loc[drivers['driverRef']==row['DriverId']]
            base.loc[(base['DriverId']==driver['driverRef']), 'driverId']=driver['driverId']
        
        
        # reset column orderings

if __name__ == "__main__":
    fetch_new()