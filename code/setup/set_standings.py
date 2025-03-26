import fastf1
import pandas as pd

def get_standings_data(cur_round:int, year:int, drivers_pth:str='../../data/drivers.csv', debug=False):
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
    drivers = pd.read_csv(drivers_pth)
    standings = None

    # iterate up until the round before the current round
    for i in range(1,cur_round):
        # get data for round i
        session = fastf1.get_session(year, i, "R")
        session.load(telemetry=False, laps=False, weather=False, messages=False)
        
        if i==1: 
            standings = session.results
            print("[DEBUG-27]: drivers = {}".format(standings['Abbreviation'].unique()))
            print("[DEBUG]: standings keys = {}".format(standings.keys()))
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

            tmp = standings.loc[(standings['round'] == j) & (standings['DriverId'] == driver), 'Position']
            if len(tmp) == 1:
                if tmp.item() == 1:
                    cur_wins += 1
            elif len(tmp) > 1:
                if tmp[0] == 1:
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
        
        print("[DEBUG]: drivers = {}".format(standings['DriverId'].unique()))
        exit()

        # set the ranks
        for driver in standings['DriverId'].unique():
            # if j==1: print(driver)         
            # curr points
            curr_points = standings.loc[(standings['round'] == j) & (standings['DriverId'] == driver), 'Points']

            if curr_points.empty: continue # skip this iteration

            if debug:
                try:
                    x_out = standings.loc[standings['DriverId']==driver,'TeamName'].drop_duplicates().items()[0] # index 1
                except:
                    x_out = standings.loc[standings['DriverId']==driver,'TeamName'].drop_duplicates()
                    print("[DEBUG]: team = {}".format(x_out))
            team = standings.loc[standings['DriverId'] == driver, 'TeamName'].drop_duplicates().iloc[0]
            standings.loc[(standings['round'] == j) & (standings['DriverId'] == driver), "construct_rank"] = ranks.loc[ranks['TeamName']==team, 'rank'].item()
    
    # set the year of the data
    standings['year'] = year
    for driver in standings['DriverId'].unique():
        driver_x = drivers.loc[drivers['driverRef']==driver]
        if len(driver_x['driverRef']) > 0:
            standings.loc[standings['DriverId']==driver, 'driverId']=driver_x['driverId'].item()
        else:
            print("[INFO]: add {} info to the drivers.csv file".format(driver))
        
    return standings