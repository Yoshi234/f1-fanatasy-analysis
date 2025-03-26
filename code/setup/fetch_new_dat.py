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
import os
import numpy as np
try:
    from set_standings import get_standings_data
    from get_weather_dat import get_weather_dat
    from get_track_augment import get_track_speeds
except ImportError:
    from .set_standings import get_standings_data
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
    
def set_prev_round_data(df:pd.DataFrame, yr=2024, rnd=None):
    '''
    Set prev round data from standings. Input dataframe should 
    contain data from the aggregated season. Sets the maximum 
    round of the season up to the current date by default
    if the year is the current year. Otherwise, it sets the maximum
    round of the given year

    Args:
    - df --- input dataframe to add standings information onto. Note,
             that this MUST include the driver_ref variable so that
             merging can be performed from the standings data 
    - yr --- the year to get standings data from
    - rnd -- the round to get standings up until (-1)
    '''
    if rnd is None:
        rnd = df['round'].max()+1

    all_cols = [
        'driverId',
        'round',
        'prev_round', 
        'prev_driver_points', 
        'prev_driver_position',
        'prev_driver_wins', 
        'prev_construct_points', 
        'prev_construct_wins', 
        'prev_construct_position'
    ]
    
    #get standings and reset variable names
    standings = get_standings_data(cur_round=rnd, year=yr)
    rename_cols = {'cum_points':'prev_driver_points', 
                   'driver_standing':'prev_driver_position', 
                   'cum_driver_wins':'prev_driver_wins', 
                   'construct_points':'prev_construct_points',
                   'cum_constructor_wins':'prev_construct_wins',
                   'construct_rank':'prev_construct_position'}
    standings = standings.rename(columns={'round':'prev_round'})
    standings = standings.rename(columns=rename_cols)
    standings['round'] = standings['prev_round'] + 1

    # subset the round data
    zero_cols = ['prev_driver_points', 'prev_construct_points']
    nan_cols = ['prev_driver_position', 'prev_driver_wins', 
                'prev_construct_position', 'prev_construct_wins']
    standings.loc[standings['round']==1, zero_cols]=0
    standings.loc[standings['round']==1, nan_cols]=np.nan
    standings.loc[standings['round']==1, 'prev_round']=0
    standings = standings[all_cols] # subset the standings data

    result = pd.merge(
        left=df, 
        right=standings, 
        on=['round', 'driverId'],
        how='left'
    )
    return result

def fetch_new(current_date=None, key=None, test=False, no_key=True):
    '''
    fetches most recent data for records not contained clean_model_data.feather
    generates matching dataframe and updates existing clean_model_data.feather
    file with additional data
    '''
    if 'private.txt' in os.listdir():
        key = 'private.txt'

    # read cross-ref data
    constructors = pd.read_csv("../../data/constructors.csv")
    drivers = pd.read_csv("../../data/drivers.csv")
    circuits = pd.read_csv("../../data/circuits.csv")
    if "clean_model_data2.csv" in os.listdir("../../data"):
        og_dat = pd.read_csv("../../data/clean_model_data2.csv")
    else:
        og_dat = pd.read_feather('../../data/clean_model_data.feather')
        print("[INFO]: Reading feather starter data")
    
    # get current datetime
    if current_date is None:
        current_date = datetime.now()
        curr_yr = current_date.year
    else:
        current_date = datetime.strptime(current_date, "%Y-%m-%d")
        curr_yr = current_date.year

    if curr_yr in og_dat['year'].unique(): # get the most recent recorded round
        prev_round = og_dat.loc[(og_dat['round']==og_dat.loc[og_dat['year']==curr_yr, 'round'].max()) &
                                (og_dat['year']==curr_yr)]
        p_r = prev_round['round'].unique()[0]
    else: 
        prev_round = None # set initially to None for standings and such
        p_r = 0
    print("[INFO]: starting round = {}".format(p_r))
    
    # schedule includes the round data - check against the most
    # recent round to see which one we start from
    schedule = fastf1.get_event_schedule(curr_yr)

    # update/subset schedule
    schedule = schedule.loc[(schedule['RoundNumber']>p_r) & 
                            (schedule['EventDate']<current_date)]
    
    if schedule.shape[0] == 0: 
        return og_dat

    res_id = og_dat['resultId'].max() + 1
    r1_id = None
    full_dat = None
    # iterate over schedule and fill records for each race
    for idx, record in schedule.iterrows():
        ### set dataframe on driver specific features
        ### and then set the universal features

        ### universal ..................................
        # get circuit id
        l1 = record['Location']
        c1 = circuits.loc[circuits['location']==l1]
        if test: 
            print("[INFO]: c1 = {}".format(c1))
        if c1.shape[0] == 0: 
            print("[ERROR]: no circuits found for {}".format(l1))
        else: print("[SUCCESS]: Circuits found for {}".format(l1))

        # set race id
        if r1_id is None and og_dat.shape[0] != 0:
            r1_id = og_dat['raceId'].max() + 1
        else:
            r1_id += 1
        
        # set event name
        ### end universal variables ...................

        # set base dataframe
        s1 = fastf1.get_session(curr_yr, record['RoundNumber'], 'R')
        s1.load(telemetry=False, weather=False)
        base = s1.results
        
        # catch potential issues
        if 'alt' in base.keys():
            if len(base['alt'] == 1):
                base['alt'] = c1['alt'].item()
            elif len(base['alt'] > 1): 
                base['alt'] = base['alt'][0]
        else:
            base['alt'] = np.nan
            
        if test: 
            print("[INFO]: c1['alt'] = {}".format(c1['alt']))
            print("[INFO]: c1['circuitId'] = {}".format(c1['circuitId']))
        base['raceId'] = r1_id
        base['round'] = record['RoundNumber']
        base = base.rename(columns={'GridPosition':'grid',
                            'Position':'positionOrder',
                            'Points':'points',
                            'Q1':'q1',
                            'Q2':'q2',
                            'Q3':'q3'})
        base['ref_name'] = s1.event.EventName
        base['event_name'] = s1.event.EventName
        
        if 'circuitId' in c1.keys():
            if len(c1['circuitId']) == 1:
                tmp_cid = c1['circuitId'].item() 
                base['circuitId'] = tmp_cid
            elif len(c1['circuitId'] > 1): 
                tmp_cid = c1['circuitId'][0]
                base['circuitId'] = tmp_cid
            else:
                base['circuitId'] = np.nan
        else:
            base['circuitId'] = np.nan

        # add track speed data
        # print("[DEBUG]: {}".format(base['event_name']))
        evnt_qry = pd.DataFrame(
            {"year":[curr_yr], 
             "name":[s1.event.EventName],
             "circuitId":[tmp_cid]})
        speeds = get_track_speeds(event=evnt_qry) # returns none upon failure
        
        if speeds is None:
            continue # move on to next event on the calendar if track speeds fail to aggregate
            
        speeds = pd.concat([speeds]*len(base), ignore_index=True)
        base = pd.concat([base.reset_index().drop('index', axis=1), speeds], axis=1)

        print("[INFO]: base.keys() = {}".format(base.keys()))
        print("[INFO]: fullnames = {}".format(base['FullName']))
        print("[INFO]: first name = {}".format(base['FirstName']))
        print("[INFO]: last name = {}".format(base['LastName']))
        # set local and api-based keys for the driver information
        
        if base['DriverId'].nunique() > 1:
            print("[INFO]: DriverId available. Using 'DriverId' value")
            print("[INFO]: DriverIds = {}".format(base['DriverId'].unique()))
            print("[INFO]: number of unique ids = {}".format(len(base['DriverId'].unique())))
            fastf1_dkey=['DriverId']
            local_dkey=['driverRef']
            
        elif (base['LastName'].nunique() > 1) and (base['FirstName'].nunique() > 1):
            print("[INFO]: DriverId unavailable, using first and last name")
            fastf1_dkey="FullName"
            local_dkey="fullname"

        # TODO TODO TODO TODO : merge the 'DriverId' values based on Abbreviation values
        # and then drop the Abbreviation column. Actually, this isn't necessary, 
        # somewhere later, the 
            
        for driver in base[fastf1_dkey].unique():
            driver_x = drivers.loc[drivers[local_dkey]==driver]
            
            print("[INFO]: driver_x = {}".format(driver_x))
            if len(driver_x[local_dkey]) > 0:
                if len(driver_x['driverId']) > 1:
                    d_id_val = driver_x['driverId'][0]
                elif len(driver_x['driverId']) == 1:
                    d_id_val = driver_x['driverId'].item()
                else:
                    print("[ERROR]: driverId not available for {}".format(driver))
                    
                base.loc[base[fastf1_dkey]==driver, 'driverId']= d_id_val
                base.loc[base[fastf1_dkey]==driver, 'resultId']=res_id
                res_id += 1
            else:
                print("[INFO]: add {} info to the drivers.csv file".format(driver))
        
        print("[INFO]: base.keys() [1] = {}".format(base.keys()))
                
        # check availability of TeamId value
        for construct in base['TeamId'].unique():
            construct_x = constructors.loc[constructors['constructorRef']==construct]
            if len(construct_x['constructorRef']) > 0:
                base.loc[base['TeamId']==construct, 'constructorId']=int(construct_x['constructorId'].item())
            else:
                print("[INFO]: add {} info to the constructors.csv file".format(construct))

        print("[INFO]: base.keys() [2] = {}".format(base.keys()))
        
        # reset column orderings
        base['lng'] = c1['lng']
        base['lat'] = c1['lat']
        base['date'] = s1.date.strftime("%Y-%m-%d")
    
        if no_key==False:
            api_key = ''
            with open(key, "r") as f:
                api_key = f.readlines()[0].strip('\n')
        else:
            api_key=None

        # update and join weather data
        base = get_weather_dat(
            loc_dat=base,
            api_key=api_key,
            debug=False
        )   
        # drop lng and lat from the dataframe
        base = base.drop(['lng', 'lat'], axis=1)    
        if full_dat is None: 
            full_dat = base
        else: 
            # concatenate new df column-wise to the full_dat for the season
            full_dat = pd.concat([full_dat, base], axis=0)

        if test: break

    full_dat = set_prev_round_data(full_dat, yr=curr_yr)
    full_dat[['regulation_id', 'engine_reg', 'tire_reg', 'aero_reg', 
              'chastech_reg', 'sporting_reg', 'pitstop_reg', 'years_since_major_cycle', 
              'is_major_cycle', 'is_major_reg', 'cycle', 'quali_position']] = np.nan
    full_dat['ref_year'] = curr_yr
    full_dat['year'] = curr_yr
    
    full_dat[['fastestLap', 'rank', 'fastestLapTime', 'fastestLapSpeed', 'laps',
              'statusId']] = np.nan
    
    # subset the necessary columns of the data set
    result = pd.concat([full_dat[og_dat.keys()].reset_index(drop=True),
                        og_dat.reset_index(drop=True)], 
                        axis=0)
    # save resulting data
    result.to_csv("../../data/clean_model_data2.csv", index=False)
    # fastestLap, rank, fastestLapTime, fastestLapSpeed is not 
    # useful since we can't access this before the race
    return result

if __name__ == "__main__":
    fetch_new()