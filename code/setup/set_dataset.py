'''
This file compiles all of the relavent data, merges it together, 
drops irrelavent columns, and cleans data in preparation for 
analysis

Author: Yoshi234
Date: 4/4/24
'''
import pandas as pd
import numpy as np

def fill_driver_dat(data:pd.DataFrame, missing:pd.DataFrame):
    '''
    missing contains fields: driverId, round, prev_round, and year
    '''
    for idx, record in missing.iterrows():
        cur_values = data.loc[(data['driverId'] == record['driverId'].item()) &
                              (data['round'] == record['round'].item()) & 
                              (data['year'] == record['year'].item()),
                              ['points', 'positionOrder']]
        pts = cur_values['points'].item()
        fp = cur_values['positionOrder'].item()
        win = 0
        # set a win indicator for missing entries
        if fp == 1: win = 1

        # get the prev_round points data from the record which isn't missing
        next_values = data.loc[(data['driverId'] == record['driverId'].item()) &
                              (data['round'] == record['round'].item() + 1) & 
                              (data['year'] == record['year'].item()), 
                              ['prev_driver_points', 'prev_driver_position',
                               'prev_driver_wins']]

        # deal with values that don'thave a consecutive round to reference
        if len(next_values) == 0:
           final_pos = data.loc[(data['year'] == record['year'].item()) &
                                (data['round'] == record['round'].item()),
                                'prev_driver_position'].max()
           data.loc[(data['driverId'] == record['driverId'].item()) &
                    (data['round'] == record['round'].item()) & 
                    (data['year'] == record['year'].item()), 
                    ['prev_driver_points', 'prev_driver_wins',
                     'prev_driver_position']] = [0, 0, final_pos]
           continue

        fill_wins = next_values['prev_driver_wins'].item() - win
        fill_pts = next_values['prev_driver_points'].item() - pts
        fill_pos = next_values['prev_driver_position'].item()

        data.loc[(data['driverId'] == record['driverId'].item()) &
                 (data['round'] == record['round'].item()) & 
                 (data['year'] == record['year'].item()), 
                 ['prev_driver_points', 'prev_driver_wins',
                  'prev_driver_position']] = [fill_pts, fill_wins, fill_pos]

    return data

def clean_dat(raw_data:pd.DataFrame, q_dat:pd.DataFrame):
    '''
    raw_data is a dataframe with the uncleaned training data, 
    we also add the qualifying data here
    '''
    # drop usesless date information
    raw_data = raw_data.drop(['fp1_date',
                          'fp1_time', 
                          'fp2_date', 
                          'fp2_time', 
                          'fp3_date', 
                          'fp3_time',
                          'quali_date', 
                          'quali_time', 
                          'sprint_date',
                          'sprint_time'],
                          axis=1)
    # drop unecessary weather data
    raw_data = raw_data.drop(['visibility'], axis=1)
    # drop other unecessary information 
    raw_data = raw_data.drop(['circuitRef', 'circuit_name',
                          'location', 'country', 'lat',
                          'lng', 'url_x', 'url_y', 'date',
                          'race_start_time', 'datetime', 
                          'precipprob', 'number', 'position',
                          'positionText', 'race_duration', 
                          'milliseconds'], axis=1)
    # get missing values
    fields = ['driverId', 'round', 'prev_round', 'year']
    missing = raw_data.loc[raw_data['prev_driver_points'].isna(), fields]

    # fill missing records
    raw_data = fill_driver_dat(raw_data, missing)

    raw_data.loc[(raw_data['alt'].isna()) & 
             (raw_data['event_name'] == 'Miami Grand Prix'),
             'alt'] = 13
    raw_data.loc[(raw_data['alt'].isna()) &
                (raw_data['event_name'] == 'Qatar Grand Prix'),
                'alt'] = 12

    # set the correct type on the column?
    raw_data['alt'] = raw_data['alt'].astype('float64')

    # drop useless data from the qualifying file
    q_dat.drop(['number','qualifyId'], axis=1, inplace=True)
    q_dat.rename(columns={'position':'quali_position'}, inplace=True)

    # merge quali dat into the full dataframe
    full_dat = pd.merge(raw_data, q_dat, on=['raceId', 
                                            'driverId',
                                            'constructorId'], how='left')
    
    # set any missing qualifying information as the grid order for the race
    missing_quali = full_dat.loc[full_dat['quali_position'].isna()]
    for idx, record in missing_quali.iterrows():
        full_dat.loc[(full_dat['driverId'] == record['driverId']) &
                    (full_dat['round'] == record['round']) & 
                    (full_dat['year'] == record['year']),
                    'quali_position'] = record['grid']
    return full_dat

def add_aug_dat(
        race_results:pd.DataFrame,
        track_data:pd.DataFrame,
        reg_data:pd.DataFrame
    ):
    '''
    Adds regulation data and track data for each row of the 
    race results data frame (each individual record)
    '''
    results = pd.merge(race_results, track_data, on='circuitId', how='left')
    results = pd.merge(results, reg_data, on='year', how='left')

    return results

def add_prev_round_dat(
        races_w_weather:pd.DataFrame, 
        d_standings:pd.DataFrame, 
        c_standings:pd.DataFrame, 
        results:pd.DataFrame,
        folder="../../data"
    ):
    '''
    adds a round variable to the races_w_weather df and combines it with the 
    driver and constructor standings which led up to the previous round

    We only include data from 2010 onwards for the sake of scoring consistency across
    seasons.
    '''
    # replace null value characters with NaNs
    results = results.replace(to_replace="\\N", value=np.nan)
    races_w_weather = races_w_weather.replace(to_replace="\\N", value=np.nan)

    # rename redundant columns to avoid loss of meaning
    races_w_weather.rename(columns={"time":"race_start_time"}, inplace=True)
    results.rename(columns={'time':'race_duration'}, inplace=True)
       
    races = pd.merge(races_w_weather, results, on='raceId', how='inner')
    races['prev_round'] = races['round'] - 1

    # # merge races and results together to get the constructor and driver IDs
    # # there shouldn't be any overlap between these attribute names
    # races = pd.merge(races, results, on=['raceId'], how='inner')
    
    # merge races with driver and constructor standings and merge on raceId
    # rename the round variable generated for standings to pre_round
    # the raceId is not necessary for merging data, so just drop it
    d_standings = pd.merge(races_w_weather[['raceId', 'round', 'year']], d_standings, on='raceId', how='inner')
    d_standings = d_standings.drop(['driverStandingsId','positionText', 'raceId'], axis=1)

    c_standings = pd.merge(races_w_weather[['raceId', 'round', 'year']], c_standings, on='raceId', how='inner')
    c_standings = c_standings.drop(['constructorStandingsId', 'positionText', 'raceId'], axis=1)

    d_standings.rename(columns={
                        'round':'prev_round',
                        'raceId':'prev_raceId',
                        'points':'prev_driver_points',
                        'position':'prev_driver_position',
                        'wins':'prev_driver_wins'}, 
                    inplace=True)
    c_standings.rename(columns={
                        'round':'prev_round',
                        'raceId':'prev_raceId', 
                        'points':'prev_construct_points',
                        'wins':'prev_construct_wins',
                        'position':'prev_construct_position'}, 
                    inplace=True)

    # merge the driver and constructor standings into races on `prev_round`
    # first, merge in the driver standings rounds -> rename redundant columns
    # second, merge in the constructor standings rounds -> rename redundant columns

    # drivers + results/races -> merge on driverId, prev_round, year (how='left')
    # constructors + results/races -> merge on constructorId, prev_round, year (how='left')
    races = pd.merge(races, d_standings, on=['driverId', 'prev_round', 'year'], how='left')
    races = pd.merge(races, c_standings, on=['constructorId', 'prev_round', 'year'], how='left')

    # fill in missing points data for prev_round = 0 - fill previous position cells with the median 
    races.loc[races['prev_round'] == 0, ['prev_driver_position', 
                                         'prev_driver_points',
                                         'prev_driver_wins',
                                         'prev_construct_points',
                                         'prev_construct_wins', 
                                         'prev_construct_position']] = [10, 0, 0, 0, 0, 5]

    return races


def main(folder:str = "../../data"):
    '''
    read in all data and format properly for analysis
    '''
    track_aug_dat = pd.read_feather("{}/track_data.feather".format(folder))
    races_w_weather = pd.read_feather("{}/races_circuits_weather.feather".format(folder))
    regs = pd.read_csv("{}/regulations.csv".format(folder))
    results = pd.read_csv("{}/results.csv".format(folder))
    driver_standings = pd.read_csv("{}/driver_standings.csv".format(folder))
    cons_standings = pd.read_csv("{}/constructor_standings.csv".format(folder))
    constructors = pd.read_csv("{}/constructors.csv".format(folder))
    drivers = pd.read_csv("{}/drivers.csv".format(folder))
    quali = pd.read_csv("{}/qualifying.csv".format(folder))

    # get races from 2010 on
    races_w_weather = races_w_weather.loc[races_w_weather['year'] >= 2010]

    # add in information regarding constructor and driver standings ahead
    # of each race which occurs
    race_results = add_prev_round_dat(
                        races_w_weather, 
                        driver_standings, 
                        cons_standings, 
                        results
                    )
    race_results = add_aug_dat(
                        race_results,
                        track_aug_dat,
                        regs
                    )
    
    # race_results.to_feather("{}/raw_model_data.feather".format(folder))
    model_data = clean_dat(race_results, quali)
    model_data.to_feather("{}/clean_model_data.feather".format(folder))

if __name__ == "__main__":
    main()