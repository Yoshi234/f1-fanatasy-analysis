import pandas as pd
import numpy as np
import pickle
import requests
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import fastf1
from datetime import datetime, timedelta
import os

def load_model(folder, mod_name):
    filename = "{}/{}.pkl".format(folder, mod_name)
    print("reading from {}".format(filename))
    with open(filename, "rb") as f:
        model = pickle.load(f)
    with open("{}/{}_features.pkl".format(folder, mod_name), "rb") as f2:
        features = pickle.load(f2)
    return model, features

def get_weather_dat(loc_dat:pd.DataFrame, api_key):
    '''
    takes as input the locations data frame (circuits merged with races) and 
    outputs the weather information associated with the following attributes
    required to be held within it:
    + latitude `lat` - case sensitive
    + longitude `lng` - case sensitive
    + date `date` - case sensitive

    See https://gitlab.com/kikefranssen/thesis_kf_f1/-/blob/main/1_load_weather.py?ref_type=heads
    for the code which is referenced here
    '''
    APIkey = api_key
    BaseURL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
    EndUrl = f"?unitGroup=metric&elements=datetime%2Ctempmax%2Ctempmin%2Ctemp%2Cdew%2Chumidity%2Cprecip%2Cprecipprob%2Cprecipcover%2Cpreciptype%2Cwindspeed%2Cwinddir%2Cvisibility&include=days&key={APIkey}&contentType=json"
    
    # iterate through loc_dat dataframe and obtain weather data
    for idx, row in loc_dat.iterrows():
        lat = str(row['lat'])
        lng = str(row['lng'])
        date = str(row['date'])
        query = BaseURL + lat + "%2C" + lng + "/" + date + "/" + date + EndUrl
        response = requests.get(query)
        w_dat = json.loads(response.text)
        w_dat = w_dat['days'] 
        w_dict = {key:value for entry in w_dat for (key,value) in entry.items()}

        for key in w_dict:
            loc_dat.loc[idx,key] = w_dict[key]
    
    # return location dataframe with added weather information
    return loc_dat

def get_features(data:pd.DataFrame, features=[], select=False):
    '''
    encodes categorical features for the data 
    '''
    if len(features) == 0: features = ['driverId', 'constructorId']
    encoder = OneHotEncoder()
    one_hot = encoder.fit_transform(data[features])
    feature_names = encoder.get_feature_names_out()
    df_features = pd.DataFrame(one_hot.toarray(), columns=feature_names)
    
    if select==True:
        df_study = pd.concat([data[['positionOrder','quali_position']], df_features], axis=1)
    else:
        df_study = pd.concat([data, df_features], axis=1)

    print(df_study.isna().sum())

    return df_study

def get_encoded_data(dat:pd.DataFrame):
    '''
    runs encoding for all of the features of interest and returns list of new 
    categorical variables
    '''
    # we need lists of these variables so we can create interactions between the encoded
    # categoricals and the other variables of interest
    driver_vars = ['driverId_{}'.format(id) for id in dat['driverId'].unique()] 
    construct_vars = ['constructorId_{}'.format(id) for id in dat['constructorId'].unique()]
    cycle_vars = ['cycle_{}'.format(id) for id in dat['cycle'].unique()]

    # test track characteristics - corner_spd_mean, num_fast_corners, num_slow_corners, strt_len_max
    # test all constructors
    # test regulation types - engine_reg, aero_reg, years_since_major_cycle
    #       engine_reg * years_since_major_cycle + engine_reg
    #       aero_reg * years_since_major_cycle + aero_reg
    encoded_dat = get_features(dat, ['constructorId','driverId', 'cycle'], select=False)
    return driver_vars, construct_vars, cycle_vars, encoded_dat

def get_standings_data(cur_round:int, year:int):
    '''
    aggregates data up to the current race round using the fastf1 api

    collects following information for races up to given point in the season
    - cumulative driver wins (prev_driver_wins)
    - cumulative constructor wins (prev_construct_wins)
    - cumulative driver points (prev_driver_points)
    - cumulative constructor points (prev_construct_points)
    - driver rank / standing (prev_driver_position)
    - constructor rank / standing (prev_construct_position)
    '''
    standings = None

    # iterate up until the round before the current round
    for i in range(1,cur_round):
        # get data for round i
        session = fastf1.get_session(year, i, "R")
        session.load(telemetry=False, laps=False, weather=False, messages=False)
        
        if i==1: 
            standings = session.results
            standings['round'] = i
        else:
            # add the round data into the dataframe before hand
            x = session.results
            x['round'] = i
            standings = pd.concat([standings, x], axis=0)

    # for each driver, calculate the cumulative points across the first j races
    standings.loc[standings['round'] == 1, 'cum_points'] = standings.loc[standings['round'] == 1, 'Points']
    standings.loc[standings['round'] == 1, 'driver_standing'] = standings.loc[standings['round'] == 1, 'cum_points'].rank(ascending=False, method='first')

    # set default driver wins
    standings.loc[(standings['round'] == 1) & (standings['Position'] == 1.0), 'cum_driver_wins'] = 1.0
    standings.loc[(standings['round'] == 1) & (standings['Position'] != 1.0), 'cum_driver_wins'] = 0.0
    
    # return if we are not on the second round at least
    if cur_round == 1: return standings

    for driver in standings['DriverId'].unique():
        # for each driver, set a unique number of personal wins

        cur_wins = 0
        for j in range(2,cur_round):
            # this series will be empty if no matches are found
            prev_cum_points = standings.loc[(standings['round'] == j-1) & (standings['DriverId'] == driver), 'cum_points'] 

            # curr points
            curr_points = standings.loc[(standings['round'] == j) & (standings['DriverId'] == driver), 'Points']

            if curr_points.empty: continue # skip this iteration

            # handle drivers who miss a given round for some reason
            if prev_cum_points.empty:
                k = j-1 
                while (prev_cum_points.empty) and k>1:
                    k -= 1
                    prev_cum_points = standings.loc[(standings['round'] == k) & (standings['DriverId'] == driver), 'cum_points']

            # set the points
            standings.loc[(standings['round'] == j) & (standings['DriverId'] == driver), 'cum_points'] = prev_cum_points + curr_points
            
            # set the current rank
            standings.loc[standings['round'] == j, 'driver_standing'] = standings.loc[standings['round'] == j, 'cum_points'].rank(ascending=False, method='first')  

            if standings.loc[(standings['round'] == j) & (standings['DriverId'] == driver), 'Position'].item() == 1:
                cur_wins += 1
            
            standings.loc[(standings['round'] == j) & (standings['DriverId'] == driver), 'cum_driver_wins'] = cur_wins

    # for constructor in standings['TeamName'].unique():
    # iterate through the rounds
    for j in range(1, cur_round):
        # get the constructor points
        for constructor in standings['TeamName'].unique():
            standings.loc[(standings['TeamName'] == constructor) & (standings['round'] == j), "construct_points"] = standings.loc[(standings['TeamName'] == constructor) & (standings['round'] == j), 'cum_points'].sum()

            standings.loc[(standings['TeamName'] == constructor) & (standings['round'] == j),
            "cum_constructor_wins"] = standings.loc[(standings['TeamName'] == constructor) & (standings['round'] == j), 'cum_driver_wins'].sum()

        # get ranks for all teams
        ranks = standings.loc[(standings['round'] == j), ['TeamName', 'construct_points']].drop_duplicates()
        ranks['rank'] = ranks['construct_points'].rank(ascending=False, method='first')

        # set the ranks
        for driver in standings['DriverId'].unique():
            # if j==1: print(driver)         
            # curr points
            curr_points = standings.loc[(standings['round'] == j) & (standings['DriverId'] == driver), 'Points']

            if curr_points.empty: continue # skip this iteration

            team = standings.loc[standings['DriverId'] == driver, 'TeamName'].drop_duplicates().item()
            standings.loc[(standings['round'] == j) & (standings['DriverId'] == driver), "construct_rank"] = ranks.loc[ranks['TeamName']==team, 'rank'].item()
    
    # set the year of the data
    standings['year'] = year

    return standings

def add_interaction(data, vars=[], drivers=[], constructors=[]):
    '''
    Args:
    - vars --------- list of the additional variables to include in 
                     each interaction term. For example, if vars held
                     track_min_speed and engine_reg, then we would 
                     generate interaction terms
                     driver * constructor * engine_reg * track_min_speed
                     This is always assumed to have at least one 
                     value held in it
    - drivers ------ list of drivers to create an interaction term for
    - constructors - list of constructors to create an interaction term for

    We use copies of the numpy arrays every time because not doing so 
    will overwrite the original data which would mess everything up
    '''
    data2 = data.copy()
    drivers = drivers.copy()
    constructors = constructors.copy()

    if len(drivers) == 0: drivers.append("any_driver")
    if len(constructors) == 0: constructors.append("any_constructor")
    for i in range(len(drivers)):
        # skip max verstappen and red bull
        # if drivers[i] == "driverId_830": continue
        for j in range(i,len(constructors)):
            # skip red bull
            # if constructors[j] == "constructorId_9": continue
            # set the initial value for the array
            interact = data[vars[0]].copy()

            v_string = ""
            # handle using driver as an interaction
            if drivers[i] != "any_driver":
                interact *= data[drivers[i]].copy()
                drive_val = drivers[i]
                v_string += f'{drive_val}-'

            # handle using constructor as an interaction 
            if constructors[j] != "any_constructor":
                interact *= data[constructors[j]].copy()
                construct_val = constructors[j]
                v_string += f'{construct_val}-'
            
            v_string += vars[0]
            for k in range(1, len(vars)):
                # print('loop executes?')
                interact *= data[vars[k]].copy()
                v_string += "-{}".format(vars[k])
            
            df = pd.DataFrame({
                v_string: interact
            })
            data2 = pd.concat([data2, df], axis=1)
            # # add interaction to the dataframe
            # data[v_string] = interact
            # print(v_string)
    return data2    

def load_and_run_model(event_date=None):
    '''
    Runs the model from a specified .pkl file

    Requires further work to make more flexible
    '''
    # load in the pretrained logistic regression model
    mod, features = load_model("../code/pretrained", "smot_f_30_norm")

    # get the current date
    if event_date is None:
        today = datetime.now()
        year = today.year
        schedule = fastf1.get_event_schedule(year)
    
    else:
        today = datetime.strptime(event_date, "%Y-%m-%d")
        year = today.year
        schedule = fastf1.get_event_schedule(year)

    # find the closest event to the current date (in the future)
    for idx, evnt in schedule.iterrows():
        schedule.loc[idx, 'diff'] = evnt['EventDate'] - today
    schedule.loc[schedule['diff'] < timedelta(0), 'diff'] = np.nan

    min_value_row = schedule.loc[schedule['diff'].idxmin()]
    
    # read in circuits, track, and regulation data
    circuits = pd.read_csv("../data/circuits.csv")
    track_dat = pd.read_feather("../data/track_data.feather")
    regs = pd.read_csv("../data/regulations.csv")
    drivers = pd.read_csv("../data/drivers.csv")
    constructors = pd.read_csv("../data/constructors.csv")
    

    # get track Id of the track for the next event
    circId = circuits.loc[circuits['location'] == min_value_row['Location'], 'circuitId'].item()
    # use the circId to get the relavent track info
    circInfo = track_dat.loc[track_dat['circuitId'] == circId]

    # get the circuit of interest
    min_value_row = min_value_row.to_frame().T
    circ_spec = circuits.loc[circuits['location'] == min_value_row['Location'].item()]

    # combine the circuit and date information together
    evnt_dat = pd.concat([min_value_row.reset_index().drop("index", axis=1), 
                          circ_spec.reset_index().drop("index", axis=1)], axis=1)
    
    # set api key for getting weather data
    key = ''
    with open("../code/setup/private.txt", "r") as f:
        x = f.readlines()[0].strip('\n')
        key = x

    date = evnt_dat['EventDate'].item()
    date = str(date).split(' ')[0]

    evnt_dat['date'] = date

    # get the forecast weather information
    weather_dat = get_weather_dat(loc_dat=evnt_dat[['lat', 'lng', 'date']], api_key=key)

    # concatenate the evnt_dat df with the weather data (combine the columns)
    evnt_dat2 = pd.concat([evnt_dat, weather_dat.drop(['date', 'lat', 'lng'],axis=1), 
                           circInfo.drop('circuitId', axis=1)], axis=1)
    
    # get the standings data and save it if it is not formatted correctly or not available
    if "standings.csv" in os.listdir():
        standings = pd.read_csv("standings.csv")
    else:
        standings = get_standings_data(min_value_row['RoundNumber'].item(), year)
        standings.to_csv("standings.csv", index=False)
    
    if min_value_row['RoundNumber'].item() - 1 not in standings['round'].unique(): 
        standings = get_standings_data(min_value_row['RoundNumber'].item(), year)
        standings.to_csv("standings.csv", index=False)

    # merge all of the data together
    drivers = drivers[['driverRef', 'driverId']]
    constructors = constructors[['constructorRef', 'constructorId']]

    # reformat the standings data
    standings2 = pd.merge(standings, drivers, how='inner', left_on='DriverId', right_on='driverRef')
    standings2 = pd.merge(standings2, constructors, how='left', left_on='TeamId', right_on='constructorRef')
    standings2 = pd.merge(standings2, regs, how='inner', on='year')

    # run code to get the encoded data
    driver_vars, construct_vars, _, standings2 = get_encoded_data(standings2)

    # NOTE / WARNING: putting this statement before the encoding step breaks the code
    # further investigation required to determine the cause of this bug
    standings2 = standings2.loc[standings2['round'] == min_value_row['RoundNumber'].item() - 1]

    # # check na values
    # print("----- printing standings 2 -----\n")
    # print(standings2.isna().sum())

    # fit interactions to the data
    interactions = [
        ['aero_reg'],
        ['years_since_major_cycle'],
        ['years_since_major_cycle','round'],
        ['corner_spd_min','aero_reg'],
        ['corner_spd_max'],
        ['corner_spd_min'],
        ['round'],
        ['round', 'years_since_major_cycle'],
        ['windspeed'],
        ['strt_len_median'],
        ['strt_len_max'],
        ['avg_track_spd'],
        ['max_track_spd'],
        ['num_fast_corners'],
        ['num_slow_corners'],
        ['num_corners'],
        ['circuit_len'],
    ]

    # the dataframe type being used is an f1 session results object
    # this needs to be converted into a pandas dataframe first
    standings2 = pd.DataFrame(standings2)

    # add the circuitId of the specific track into the standings data of interest
    standings2['circuitId'] = evnt_dat2['circuitId'].item()
    standings2.rename(columns={"cum_points":"prev_driver_points", 
                            "driver_standing":"prev_driver_position", 
                            "construct_rank":"prev_construct_position", 
                            "construct_points": "prev_construct_points",
                            "cum_driver_wins": "prev_driver_wins",
                            "cum_constructor_wins": "prev_construct_wins"
                            }, inplace=True)

    standings3 = pd.merge(standings2, evnt_dat2, on='circuitId', how='inner')

    # # check na values
    # print("\n----- printing standings 3 -----\n")
    # for x, y in standings3.isna().sum().items():
    #     print('{} | {}'.format(x, y))
    # # print(standings3.isna().sum())

    # set a subset of the constructors and drivers we want
    driver_vars = ['driverId_844', 'driverId_815', 'dirverId_830']
    construct_vars = ['constructorId_9', 'constructorId_6', 'constructorId_131']

    # just generate all of the interactions and get the feature subset features
    for interaction in interactions:
        standings3 = add_interaction(standings3, constructors=construct_vars, vars=interaction)
    for interaction in interactions:
        standings3 = add_interaction(standings3, drivers=driver_vars, constructors=[], vars=interaction)

    # run predictions and obtain results
    probs_1 = []
    probs_0 = []
    drivers = []

    for driver in standings3['FullName'].unique():
        subset = standings3.loc[standings3['FullName'] == driver]
        # if subset[features].isna().sum().any():
        #     print("Missing values")
        #     print(subset[features])
        #     continue

        probs = mod.predict_proba(subset[features])
        probs_1.append(probs[0][1])
        probs_0.append(probs[0][0])
        drivers.append(driver)
        # print(probs)

    results_df = pd.DataFrame({
    "prob of top 3 finish": probs_1,
    "prob of bottom 17 finish": probs_0,
    "driver name": drivers
    })

    return (results_df, min_value_row['EventName'].item(), 
            year, min_value_row['RoundNumber'].item())
    
def main():
    results, event, year, round_num = load_and_run_model(event_date='2024-04-19')
    if results is None: 
        return
    else: 
        results.to_csv("new_results.csv", index=False)   
        with open("info.txt", 'w') as f:
            f.write("{}\n".format(event))
            f.write("{}\n".format(year))
            f.write("{}\n".format(round_num))

if __name__ == "__main__":
    main()