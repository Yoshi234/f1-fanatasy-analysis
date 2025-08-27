'''
Author: Yoshi236
Date: 8/26/2025

This file cotnains source code related to the sourcing of fp2 laptime data, 
and clustering those lap times for each driver
'''
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_lap_rep(
    session_obj = None,
    driver_code:str = 'STR',
    approach:str = 'stint',
    selection:str = 'min',
    agg_func:str = 'median',
    lap_thresh:float = 1.1
):
    '''
    Returns a representative pace summary for the input driver over a
    particular session based on the provided parameters. By default, 
    this procedure is built to use the median lap time of the fastest
    stint for the fp2 session of a given weekend

    Parameters:
    - session (fastf1.Session.laps): The laps for a given session
    - driver_code (str): The driver code for the desired driver
    - race_round (int): The integer of the race round
    - year (int): The year of the session to pull data from
    - approach (str): The lap selection approach to use. Options include
      'stint', 'quantile' and 'cluster'. If stint is chosen, data will be 
      grouped by the stint, and statistics run over stints with at least
      one lap in them. 
      If 'quantile' is chosen, the quantile
      corresponding to the selection criteria will be chosen. For example,
      if selection='min', then the quantile with the minimum lap time will
      be chosen.
    - selection (str): options = ['min', 'max']. If 'min', chooses the 
      least lap time cluster or quantile, whereas max will choose the greatest.
    - agg_func (str): options = ['median', 'mean']. The aggregation function
      used to 'summarize' lap speeds across a given set of laps. Median is good
      for avoiding emphasis on outlier laps in a stint.
    - lap_thresh (float): the maximum threshold of lap time for consideration
      as part fo the representative session lap data. By default, this is set to 
      1.1, or 110% of the quickest lap time for the driver.

    Returns
    - lap_rep (float): a float encoded lap time (in seconds) of the representative
      pace of the input driver for the given session
    '''
    lap_data = session_obj.laps.pick_driver(driver_code).pick_quicklaps(threshold=lap_thresh)
    
    if approach == "stint":
        summary_stats = lap_data.groupby('Stint')['LapTime'].agg([agg_func, 'count'])
        try: 
            if selection == 'min':
                rep_val = summary_stats.loc[summary_stats['count']>1, agg_func].dt.total_seconds().min()
            else:
                rep_val = summary_stats.loc[summary_stats['count']>1, agg_func].dt.total_seconds().max()
        except:
            if selection == 'min':
                rep_val = summary_stats[agg_func].dt.total_seconds().min()
            else:
                rep_val = summary_stats[agg_func].dt.total_seconds().max()
    
    return rep_val


def get_driver_session_ranks(
    round = 15, 
    session_type = 'FP2', 
    year = 2025,
    base_predictions = None
):
    '''
    Obtains the lap times and associated ranks for each driver from a given 
    fp2 session.

    Args:
    - round (int)
    - session (str)
    - year (int)
    - base_predictions (pd.DataFrame or path[str]): Either the output predictions
      dataframe itself, or the path to the dataframe to read for predictions
    '''
    if base_predictions is None: 
        print("[ERROR]: No predictions input is provided")
        return

    if isinstance(base_predictions, str):
        base_predictions = pd.read_csv(base_predictions)

    session = fastf1.get_session(year=year, gp=round, identifier=session_type)

    driver = []
    rep_lap = []
    for driver_code in base_predictions['Driver'].unique():
        driver.append(driver_code)
        rep_lap.append(get_lap_rep(session, driver_code))
    
    rep_laps = pd.DataFrame({
        "Driver": driver, 
        "Session_Lap": rep_lap
    })

        

    