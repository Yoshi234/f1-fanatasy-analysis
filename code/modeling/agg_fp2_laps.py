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
from sklearn.preprocessing import PowerTransformer


def agg_lap_rep_selection(
    session_obj,
    kvals=[2,3],
    agg_func='mean',
    min_sample=2,
    lap_thresh=1.1
) -> pd.DataFrame:
    '''
    Performs clustering over all driver laps and then 
    picks representative times for each driver
    based on the following procedure:
    1. look at the cluster with the fastest mean lap first. 
       for each driver, pick the average of their laps in 
       this cluster (or somet other summary statistic)
       and take that as their "representative" lap. 
    2. For each driver with no lap in this cluster, 
       look at the next fastest cluster, and aggregate
       based on laps from that cluster - take the mean/
       summary stat for that cluster as the representative
       lap. If not available, continue until all clusters
       have been exhausted. If no representative lap is 
       available, ignore this driver.

    Inputs:
    -------
    - session_obj (fastf1.session): the fastf1.session object you want
      to query lap data from
    - kvals (list[int]): the different number of clusters to try with
      silhouette analysis to choose the best value
    - agg_func (str): options=['mean', 'median']. The aggregation function
      used to define the representative lap time for drivers in a given 
      free practice session
    - min_sample (int): the minimum number of samples which must be used to 
      derive a representative lap for each driver
    '''
    full_laps = session_obj.laps.pick_quicklaps(threshold=lap_thresh).dropna(subset=['LapTime'])

    # get seconds format of lap times for all drivers
    full_laps.loc[:,'lap_seconds'] = full_laps['LapTime'].dt.total_seconds() # convert to seconds (float)
    full_laps['lap_seconds_norm'] = (full_laps['lap_seconds'] - full_laps['lap_seconds'].mean())\
                                /full_laps['lap_seconds'].std()
    
    current_score = 0
    fit_array = full_laps[['lap_seconds_norm', 'LapNumber']].dropna().to_numpy()
    
    if len(fit_array) == 0: # return a null value if no laps were recorded
        return np.nan
    
    elif len(fit_array) == 1:
        return fit_array[0]
    
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
            full_laps['lap_cluster'] = kmeans.labels_
            s_score = 0
        
        if s_score > current_score:
            current_score = s_score
            full_laps['lap_cluster'] = kmeans.labels_
    
    # perform analysis over end data
    # pseudocode
    # create a df to hold cluster information
    # col1: cluster num
    # col2: cluster min
    # while rep_laps df not full:
    #   take the cluster with minimum lap time average
    #   drop it from the cluster information df
    #   take a grouped average / median (whatever summary statistic you want)
    #   -> and return the information for each driver as like the 'base' 
    #   -> driver rep laps data frame
    #   check if all driver names in the resulting df. if so, stop loop, 
    #   -> otherwise continue

    rep_laps = pd.DataFrame()

    cluster_id_df = full_laps.groupby('lap_cluster')['lap_seconds'].\
                              agg(agg_func).\
                              reset_index().\
                              rename(columns={"index": "lap_cluster"})

    # print("CLUSTER ID DF")
    # print("---")
    # print(cluster_id_df)
    # print("---")

    continue_agg = True
    while continue_agg:
        c_id = cluster_id_df.loc[cluster_id_df['lap_seconds']==cluster_id_df['lap_seconds'].min(), 'lap_cluster'].values[0]
        cluster_id_df = cluster_id_df[cluster_id_df['lap_cluster']!=c_id] # eliminate from id_df

        tmp = full_laps.loc[full_laps['lap_cluster']==c_id]

        # filter outlier laps for each driver from the cluster (greater than 1.05 * the minimum lap time)
        min_driver_laps = tmp.groupby("Driver")['lap_seconds'].\
                              agg(['min', 'count']).\
                              reset_index().\
                              rename(columns={'index': 'Driver'})
        # print("MIN DRIVER LAPS\n{}".format(min_driver_laps))
        for driver in tmp['Driver'].unique():
            out_thresh = 1.05 * min_driver_laps[min_driver_laps['Driver']==driver]['min'].values[0]
            tmp = tmp.loc[
                ~((tmp['Driver']==driver) &
                (tmp['lap_seconds'] > out_thresh))
            ]

        # for driver in tmp['Driver'].unique():
        #     print(tmp.loc[tmp['Driver']==driver, ['Driver', 'lap_seconds']])

        if rep_laps.empty:
            rep_laps = tmp.groupby("Driver")['lap_seconds'].\
                           agg([agg_func, 'count']).\
                           reset_index().\
                           rename(columns={"index": "Driver"})
        
        else:
            tmp = tmp.loc[~tmp['Driver'].isin(rep_laps['Driver'].unique())]
            tmp_drivers = tmp.groupby("Driver")['lap_seconds'].\
                              agg(agg_func).\
                              reset_index().\
                              rename(columns={"index": "Driver"})
            
            rep_laps = pd.concat([rep_laps, tmp_drivers], 
                                 axis=0,
                                 ignore_index=True)
            
        if set(full_laps['Driver'].unique()) == set(rep_laps['Driver'].unique()):
            continue_agg = False

    return rep_laps.sort_values(by=agg_func)


def get_lap_rep(
    session_obj = None,
    driver_code:str = 'STR',
    approach:str = 'stint',
    selection:str = 'min',
    agg_func:str = 'median',
    lap_thresh:float = None,
    kvals:list = [2, 3]
):
    '''
    Returns a representative pace summary for the input driver over a
    particular session based on the provided parameters. By default, 
    this procedure is built to use the median lap time of the fastest
    stint for the fp2 session of a given weekend

    Inputs:
    -------
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

    Outputs:
    --------
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
        # print("[INFO]: driver = \n{}".format(lap_data['Driver'].unique()))
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
        # print("[INFO]: Lap Data after Clustering Laps\n{}".format(lap_data))
            
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
    approach = 'quantile',
    use_i_clusters = False
):
    '''
    Obtains the lap times and associated ranks for each driver from a given 
    fp2 session.

    Inputs:
    -------
    - round (int)
    - session (str)
    - year (int)
    - base_predictions (pd.DataFrame or path[str]): Either the output predictions
      dataframe itself, or the path to the dataframe to read for predictions
    - approach (str): the approach to use for grouping laps. options=['stint', 'quantile', 'cluster']
    - use_i_clusters (boolean): If True, clusters lap over individual drivers instead of all together
      If false, will use the `agg_lap_rep_selection()` function for this purpose
    '''
    if base_predictions is None: 
        print("[ERROR]: No predictions input is provided")
        return

    if isinstance(base_predictions, str):
        base_predictions = pd.read_csv(base_predictions)

    session = fastf1.get_session(year=year, gp=round, identifier=session_type)
    session.load() # load session data

    if use_i_clusters:
        driver = []
        rep_lap = []
        for driver_code in base_predictions['Driver'].unique():
            driver.append(driver_code)
            rep_lap.append(get_lap_rep(session, driver_code, agg_func='mean', approach=approach, lap_thresh=1.1))
        
        rep_laps = pd.DataFrame({
            "Driver": driver, 
            "Session_Lap": rep_lap
        })
    else:
        rep_laps = agg_lap_rep_selection(session, agg_func='mean').rename(columns={"mean": "Session_Lap"}).drop("count", axis=1)

    # take the max session lap time
    rep_laps.loc[pd.isnull(rep_laps['Session_Lap'])] = rep_laps['Session_Lap'].max()

    rep_laps['lap_seconds_norm'] = (rep_laps['Session_Lap'] - rep_laps['Session_Lap'].median())\
                                   / rep_laps['Session_Lap'].std()
    
    
    rep_laps = rep_laps.sort_values(by='Session_Lap', ascending=True)
    ranks = rep_laps['Session_Lap'].rank(method='min').astype(int)
    rep_laps['Rank'] = ranks

    # print(rep_laps[['Driver', 'Session_Lap', 'Rank', 'lap_seconds_norm']]\
    #       .sort_values(by='Rank'))

    return rep_laps 


def get_new_pred(row):
    # DEBUG - increased the scaling value requirement from 10 to 20
    ratio = (row['positionOrder'] - row['Rank'])/20 
    if ratio * 1.96 < -2.576:
        prod = -2.576
    elif ratio * 1.96 > 2.576:
        prod = 2.576
    else:
        prod = 1.96 * ratio
    
    return row['positionOrder'] - prod*row['std_err_r']


def get_new_pred_alt(row):
    # DEBUG - increased the scaling value requirement from 10 to 20
    ratio = (row['positionOrder']-row['Rank'])/20
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


def test_main2(
    predictions_file_path:str = "../../results/monza/predictions.csv"
):
    '''
    Inputs:
    -------
    - predictions_file_path (str): the path to the base predictions file to use
    '''
    preds = pd.read_csv(predictions_file_path)
    ranks = get_driver_session_ranks(round=16, base_predictions=preds, approach='cluster', use_i_clusters=False)

    ex_cols = ranks.drop('Driver', axis=1).columns
    if len(set(ex_cols) - set(preds.columns)) < len(ex_cols):
        preds = preds.drop(ex_cols, axis=1)
    
    z = pd.merge(preds, ranks, on='Driver', how='left')
    z['adj_pred_order1'] = z.apply(get_new_pred, axis=1)
    z['adj_pred_order2'] = z.apply(get_new_pred_alt, axis=1)

    # if missing - just ignore the data
    z.loc[pd.isnull(z['adj_pred_order1']), 'adj_pred_order1'] = z.loc[pd.isnull(z['adj_pred_order1']), 'positionOrder']
    z.loc[pd.isnull(z['adj_pred_order2']), 'adj_pred_order2'] = z.loc[pd.isnull(z['adj_pred_order2']), 'positionOrder']

    z['new_fp_1'] = z['adj_pred_order1'].rank(method='min', ascending=True).astype(int)
    z['new_fp_2'] = z['adj_pred_order2'].rank(method='min', ascending=True).astype(int)
    z = z.sort_values(by = 'adj_pred_order2')

    print(z[['Driver', 'new_fp_2']])
        
def test_main1(
    predictions_file_path:str = '../../results/monza/predictions.csv'
):
    preds = pd.read_csv(predictions_file_path)
    ranks = get_driver_session_ranks(round=16, base_predictions=preds, approach='cluster', session_type='FP2')
    
    ex_cols = ranks.drop('Driver', axis=1).columns
    if len(set(ex_cols) - set(preds.columns)) < len(ex_cols):
        preds = preds.drop(ex_cols, axis=1)
    
    z = pd.merge(preds, ranks, on='Driver', how='left')
    z['adj_pred_order1'] = z.apply(get_new_pred, axis=1)
    z['adj_pred_order2'] = z.apply(get_new_pred_alt, axis=1)

    # if missing - just ignore the data
    z.loc[pd.isnull(z['adj_pred_order1']), 'adj_pred_order1'] = z.loc[pd.isnull(z['adj_pred_order1']), 'positionOrder']
    z.loc[pd.isnull(z['adj_pred_order2']), 'adj_pred_order2'] = z.loc[pd.isnull(z['adj_pred_order2']), 'positionOrder']

    z['new_fp_1'] = z['adj_pred_order1'].rank(method='min', ascending=True).astype(int)
    z['new_fp_2'] = z['adj_pred_order2'].rank(method='min', ascending=True).astype(int)
    z = z.sort_values(by = 'adj_pred_order2')

    ax = z['Session_Lap'].dropna().hist()
    fig = ax.get_figure()
    fig.savefig("plot.png", dpi=300, bbox_inches='tight')

    plt.clf()
    pt = PowerTransformer(method='yeo-johnson')
    tmp_series = pd.Series(pt.fit_transform(z['Session_Lap'].dropna().to_numpy().reshape(-1,1)).squeeze())
    tmp_series = (tmp_series - tmp_series.mean())/tmp_series.std()
    ax2 = tmp_series.dropna().hist()
    fig2 = ax2.get_figure()
    fig2.savefig("plot2.png", dpi=300, bbox_inches='tight')

    plt.clf()
    tmp_series = pd.Series(z['Session_Lap'] - z['Session_Lap'].median()).dropna()
    tmp_series /= tmp_series.std()
    # print("[TEMP SERIES]: \n{}".format(tmp_series))
    ax3 = tmp_series.dropna().hist()
    fig3 = ax3.get_figure()
    fig3.savefig("plot3.png", dpi=300, bbox_inches='tight')

    # print(z[['Driver', 'new_fp_1', 'new_fp_2', 'fp', 'positionOrder', 'adj_pred_order1', 'adj_pred_order2', 'Session_Lap', 'lap_seconds_norm']])


if __name__ == "__main__":
    test_main2("../../results/monza/predictions.csv")
