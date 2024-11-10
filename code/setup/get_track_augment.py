'''
python functions and script for loading circuit information
for each track, and storing that information as a separate 
file.

Author: Yoshi234
Date: 4/3/2024
'''
import pandas as pd
import numpy as np
import fastf1
# from fastf1 import SessionNotAvailableError

def get_track_speeds(event, mode='Q'):
    '''
    Uses the fastest qualifying lap from a given event to obtain 
    summary information about the speeds of the track, specifically
    in regards to the cornering speeds and straight lengths of the 
    track. I may also include speed information with regards to the
    maximum speed, average, and minimum overall. This information 
    may be redundant and will need to be accounted for in preprocessing

    Args:
    - event --- pandas dataframe object which holds following 
    data 
        - year (year the event occurred)
        - name (e.g. Spanish Grand Prix)
        - circuitId (circuitId of the track)
    - mode --- the session to obtain track data from. By default, this is
    given as "Q" for qualifying session. 
    Returns:
    - track_speeds --- dataframe of summary statistics for the 
    speed data around a given track (see code)
    '''
    # get the session desired
    try:
        session = fastf1.get_session(event['year'].item(), event['name'].item(), mode)
        session.load()
    except:
        print("[ERROR]: queried session not available - ret. None")
        return None 

    # get fastest lap car data and circuit data from session
    fastest_lap = session.laps.pick_fastest()
    car_data = fastest_lap.get_car_data().add_distance()
    circuit_info = session.get_circuit_info()

    # set list to save data correct speed data for each corner
    speed_vals = []

    # use l2 norm for distance calculation
    for i in range(len(circuit_info.corners['Distance'])):
        c_diff = np.sqrt(
            (circuit_info.corners['Distance'][i] - car_data['Distance'])**2
        )
        min_idx = np.where(c_diff == c_diff.min())
        speed_vals.append(car_data['Speed'][min_idx[0][0]])
    
    # convert to array to get efficient computations
    speed_vals = np.array(speed_vals)

    # get the straight lengths
    corner_dists = circuit_info.corners["Distance"]
    corner_dists = np.concatenate(([0], corner_dists))
    straight_lens = np.ediff1d(corner_dists)
    
    # account for distance between final corner and the starting line
    # add this distance to the main straight
    straight_lens[0] += corner_dists[-1] - corner_dists[-2]

    # NOTE: use the same name for circuit / event used in races.csv
    # NOTE: hard coded values for low and high speed corners obtained from 
    # summary data about mean 1st and 3rd quartile corner speeds across these tracks
    track_dat = pd.DataFrame(
        {
         "ref_year": [event['year'].item()],
         "ref_name": [event['name'].item()],
         "circuitId": [event['circuitId'].item()],
         "strt_len_mean": [straight_lens.mean()],
         "strt_len_q1": [np.quantile(straight_lens, q=0.25)],
         "strt_len_median": [np.quantile(straight_lens, q=0.50)],
         "strt_len_q3": [np.quantile(straight_lens, q=0.75)], 
         "strt_len_max": [straight_lens.max()],
         "strt_len_min": [straight_lens.min()],
         "str_len_std": [straight_lens.std()],
         "avg_track_spd": [car_data["Speed"].mean()],
         "max_track_spd": [car_data['Speed'].max()],
         "min_track_spd": [car_data['Speed'].min()],
         "std_track_spd": [car_data['Speed'].std()],
         "corner_spd_mean": [speed_vals.mean()],
         "corner_spd_q1": [np.quantile(speed_vals, q=0.25)],
         "corner_spd_median": [np.quantile(speed_vals, q=0.50)],
         "corner_spd_q3": [np.quantile(speed_vals, q=0.75)],
         "corner_spd_max": [speed_vals.max()],
         "corner_spd_min": [speed_vals.min()],
         "num_slow_corners": [len(speed_vals[speed_vals < 136.86290322580646])],
         "num_fast_corners": [len(speed_vals[speed_vals > 235.9758064516129])],
         "num_corners": [len(speed_vals)],
         "circuit_len": [car_data['Distance'].max()]
        }
    )
    # return the dataframe with relavent track summary information
    return track_dat


def get_track_aug_dat(folder:str = "../../data"):
    '''
    Obtains track data for each of the circuits listed in the 
    circuits.csv files and were raced at from 2018 to 2024. 
    We do not use data which is older than this since it is not
    available to us. 

    In every case, the fastest lap from the qualifying session
    is used as a proxy for the track characteristics. 

    Args:
    - None
    Returns:
    - Dataframe with relavent data to store
    '''
    # load races and circuits data to merge into a singular dataframe
    # rename the 'name' column to avoid duplicate columns upon merging
    races = pd.read_csv("{}/races.csv".format(folder))

    valid_tracks = []
    for c_id in races['circuitId'].unique():
        # get events with the corresponding circuit id
        c_events = races.loc[races['circuitId'] == c_id]

        # find events which occurred at each track for the most recent year
        event_choice = c_events.loc[c_events['year'] == c_events['year'].max()]

        # if event year in range, add first entry to pool (there is fastf1 api support)
        if event_choice['year'].item() >= 2018: valid_tracks.append(event_choice[["name",
                                                                                  "circuitId",
                                                                                  "year"]])

    first = True
    track_dat = None

    # fetch the data for each of the valid tracks
    for track_event in valid_tracks:
        if first == True: 
            track_dat = get_track_speeds(track_event)
            first = False
        else:
            track_dat = pd.concat([track_dat, get_track_speeds(track_event)])

    return track_dat


if __name__ == "__main__":
    folder = "../../data"
    full_track_dat = get_track_aug_dat(folder)
    full_track_dat.to_feather("{}/track_data.feather".format(folder))



