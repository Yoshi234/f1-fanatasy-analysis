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
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_lap_rep(
    session_obj = None,
    driver_code:str = 'STR',
    approach:str = 'stint',
    selection:str = 'min',
    agg_func:str = 'median',
    lap_thresh:float = None,
    kvals:list = [2]
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
    lap_data = session_obj.laps.pick_drivers(driver_code).pick_quicklaps(threshold=lap_thresh)
    
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
    
    elif approach == 'quantile':
        lap_quantiles = np.quantile(lap_data['LapTime'], q=[0.33, 0.75])
        lap_data.loc[lap_data.index,'quantile'] = np.where(lap_data['LapTime'] < lap_quantiles[0], 1,
                                        np.where(lap_data['LapTime'] < lap_quantiles[1], 2,
                                                 3))
        summary_stats = lap_data.groupby('quantile')['LapTime'].agg([agg_func])
        if selection == 'min':
            rep_val = summary_stats[agg_func].dt.total_seconds().min()
        else:
            rep_val = summary_stats[agg_func].dt.total_seconds().max()

    elif approach == 'cluster':
        lap_data.loc[:,'lap_seconds'] = lap_data['LapTime'].dt.total_seconds() # convert to seconds (float)
        lap_data['lap_seconds_norm'] = (lap_data['lap_seconds'] - lap_data['lap_seconds'].mean())\
                                    /lap_data['lap_seconds'].std()
        
        current_score = 0
        print("[INFO]: driver = \n{}".format(lap_data['Driver'].unique()))
        fit_array = lap_data['lap_seconds_norm'].dropna().to_numpy().reshape(-1,1)
        
        if len(fit_array) == 0: # return a maximum value if no laps were recorded
            return np.nan
        
        elif len(fit_array) == 1:
            return fit_array[0]
        
        # print('[INFO]: fit array = \n{}'.format(fit_array))  
        for i in range(len(kvals)):
            kmeans = KMeans(
                n_clusters = kvals[i],
                random_state=42,
                n_init='auto'
            ).fit(fit_array)
            try:
                s_score = silhouette_score(fit_array, kmeans.labels_)
            except Exception as e:
                print('[ERROR]: fit array = \n{}'.format(fit_array))
                print('[FULL ERROR]: {}'.format(e))
                lap_data['lap_cluster'] = kmeans.labels_
                s_score = 0
            
            if s_score > current_score:
                current_score = s_score
                lap_data['lap_cluster'] = kmeans.labels_
        
        # debug remove this
        print("[INFO]: Lap Data after Clustering Laps\n{}".format(lap_data))
            
        summary_stats = lap_data.groupby('lap_cluster')['lap_seconds'].agg([agg_func])
        if selection == 'min':
            rep_val = summary_stats[agg_func].min()
        else:
            rep_val = summary_stats[agg_func].max()

    return rep_val


def get_driver_session_ranks(
    round = 15, 
    session_type = 'FP2', 
    year = 2025,
    base_predictions = None,
    approach = 'quantile'
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
    - approach (str): the approach to use for grouping laps. options=['stint', 'quantile', 'cluster']
    '''
    if base_predictions is None: 
        print("[ERROR]: No predictions input is provided")
        return

    if isinstance(base_predictions, str):
        base_predictions = pd.read_csv(base_predictions)

    session = fastf1.get_session(year=year, gp=round, identifier=session_type)
    session.load() # load session data

    driver = []
    rep_lap = []
    for driver_code in base_predictions['Driver'].unique():
        driver.append(driver_code)
        rep_lap.append(get_lap_rep(session, driver_code, agg_func='mean', approach=approach, lap_thresh=1.1))
    
    rep_laps = pd.DataFrame({
        "Driver": driver, 
        "Session_Lap": rep_lap
    })
    # take the max session lap time
    rep_laps.loc[pd.isnull(rep_laps['Session_Lap'])] = rep_laps['Session_Lap'].max()

    rep_laps['lap_seconds_norm'] = (rep_laps['Session_Lap'] - rep_laps['Session_Lap'].mean())\
                                   / rep_laps['Session_Lap'].std()
    
    
    rep_laps = rep_laps.sort_values(by='Session_Lap', ascending=True)
    ranks = rep_laps['Session_Lap'].rank(method='min').astype(int)
    rep_laps['Rank'] = ranks

    # print(rep_laps[['Driver', 'Session_Lap', 'Rank', 'lap_seconds_norm']]\
    #       .sort_values(by='Rank'))

    return rep_laps 


def get_new_pred(row):
    ratio = (row['positionOrder'] - row['Rank'])/10
    if ratio * 1.96 < -2.576:
        prod = -2.576
    elif ratio * 1.96 > 2.576:
        prod = 2.576
    else:
        prod = 1.96 * ratio
    
    return row['positionOrder'] - prod*row['std_err_r']


def get_new_pred_alt(row):
    ratio = (row['positionOrder']-row['Rank'])/10
    new_ratio = (1 + math.fabs(row['lap_seconds_norm'])) * ratio
    prod = new_ratio * 1.96
    if prod > 2.576: prod = 2.576
    elif prod < -2.576: prod = -2.576
    return row['positionOrder'] - prod*row['std_err_r']


def recalc_fantasy_score(preds, dq_scores, dr_scores):
    preds['position_change'] = preds['sp'] - preds['fp']
    preds['fantasy_pts'] = preds['position_change']
    for idx, pred in preds.iterrows():
        preds.loc[idx, 'fantasy_pts'] += dq_scores[pred['sp']]
        preds.loc[idx, 'fantasy_pts'] += dr_scores[pred['fp']]
    return preds

        
def test_main(
    predictions_file_path:str = '../../results/hungary/predictions.csv'
):
    preds = pd.read_csv(predictions_file_path)
    ranks = get_driver_session_ranks(round=15, base_predictions=preds, approach='cluster', session_type='FP2')
    
    ex_cols = ranks.drop('Driver', axis=1).columns
    if len(set(ex_cols) - set(preds.columns)) < len(ex_cols):
        preds = preds.drop(ex_cols, axis=1)
    
    z = pd.merge(preds, ranks, on='Driver')
    z['adj_pred_order1'] = z.apply(get_new_pred, axis=1)
    z['adj_pred_order2'] = z.apply(get_new_pred_alt, axis=1)
    z['new_fp_1'] = z['adj_pred_order1'].rank(method='min', ascending=True).astype(int)
    z['new_fp_2'] = z['adj_pred_order2'].rank(method='min', ascending=True).astype(int)
    z = z.sort_values(by = 'adj_pred_order2')

    print(z[['Driver', 'new_fp_1', 'new_fp_2', 'positionOrder', 'adj_pred_order1', 'adj_pred_order2', 'Session_Lap', 'lap_seconds_norm']])


if __name__ == "__main__":
    test_main("../../results/zandfort_post_fp1_rf/predictions.csv")