import pandas as pd
import matplotlib.pyplot as plt 
try:
    from ISLP.models import (ModelSpec as MS, summarize)
    import statsmodels.api as sm
    from ISLP import confusion_table
except:
    print("[ERROR]: MISSING ISLP and STATSMODELS")
import numpy as np
import fastf1
import os

#from imblearn.over_sampling import (
#    SMOTENC,
#    SMOTE
#)
from sklearn.metrics import f1_score
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

# local imports
try:
    # from param_train import (
    #     logistic_fit, 
    #     xgb_fit
    # )
    from selection import (
        get_data_in_window, 
        get_features, 
        get_encoded_data, 
        add_interaction,
        add_podiums
    )
    from mod_point_distrib import std_pt_distrib
except: # import as module
    # from .param_train import (
    #     logistic_fit,
    #     xgb_fit
    # )
    from modeling.selection import (
        get_data_in_window, 
        get_features, 
        get_encoded_data, 
        add_interaction,
        add_podiums
    )
    from modeling.mod_point_distrib import std_pt_distrib
    # can only be accomplished by modular import
    from setup.set_standings import (
        get_standings_data
    )
    from setup.get_track_augment import (
        get_track_speeds
    )
    
import warnings
warnings.filterwarnings('ignore')

dq_scores = {
    1: 10, 2: 9, 3: 8, 4: 7, 5:6,
    6:5, 7:4, 8:3, 9:2, 10:1, 11:0,
    12:0, 13:0, 14:0, 15:0, 16:0, 
    17:0, 18:0, 19:0, 20:0
}
dr_scores = {
    1:25, 2:18, 3:15, 4:12, 5:10,
    6:8, 7:6, 8:4, 9:2, 10:1, 11:0, 
    12:0, 13:0, 14:0, 15:0, 16:0, 
    17:0, 18:0, 19:0, 20:0
}


def get_forecast_data(
    rnd=3,
    drivers_data="../data/drivers.csv",
    year=2025, 
    main_keys=[
        # 'prev_driver_points',
        'prev_driver_position',
        'prev_driver_wins',
        # 'prev_construct_points',
        'prev_construct_position',
        'prev_construct_wins'
    ],
    circuits_data="../data/circuits.csv",
    constructors_data="../data/constructors.csv",
    main_features=[],
    vars=[],
    fitted_drivers=[],
    fitted_constructors=[],
    full_vars=[]
):
    '''
    return the X data matrix to use in making predictions
    for the next race in sequence
    '''
    # fetch the current round
    schedule = fastf1.get_event_schedule(year)
    event = schedule.loc[(schedule['RoundNumber']==rnd)]
    
    # get the circuits data for reference
    drivers = pd.read_csv(drivers_data)
    constructors = pd.read_csv(constructors_data)
    circuits = pd.read_csv(circuits_data)
    c1 = circuits.loc[circuits['location']==event['Location'].item()]
    
    # 1 get subset of data frame from previous race and use that data
    # to perform updating
    # print("[INFO]: round = {}".format(rnd))
    standings = get_standings_data(rnd, year, drivers_data)
    # print("[INFO]: standings.shape = {}".format(standings.shape))
    # print("[INFO]: standings.keys() = {}".format(standings.keys()))
    
    # 2 get 
    rename_cols = {'cum_points':'prev_driver_points', 
                   'driver_standing':'prev_driver_position', 
                   'cum_driver_wins':'prev_driver_wins', 
                   'construct_points':'prev_construct_points',
                   'cum_constructor_wins':'prev_construct_wins',
                   'construct_rank':'prev_construct_position'}
    standings = standings.rename(columns=rename_cols)
    
    # get the standings for just the most recent round
    base = standings.loc[standings['round']==rnd-1]
    
    # get driver Ids for each driver and team id
    for driver in base['DriverId'].unique():
        driver_x = drivers.loc[drivers['driverRef']==driver]
        if len(driver_x['driverRef']) > 1:
            d_id_val = driver_x['driverId'].values[0]
        elif len(driver_x['driverRef'])==1:
            d_id_val = driver_x['driverId'].item()
        else:
            print("[ERROR]: Missing {} from Drivers data".format(driver))
            print("[INFO]: Unique drivers = {}".format(base['DriverId'].unique()))
        base.loc[base['DriverId']==driver, 'driverId']=d_id_val
    
    # get constructor Ids for eachh constructor and team id
    # print("--- Unique TeamId Values ---")
    # print(base['TeamId'].unique())
    for constructor in base['TeamId'].unique():
        construct_x = constructors.loc[constructors['constructorRef']==constructor]
        if len(construct_x['constructorRef']) == 1:
            c_id_val = construct_x['constructorId'].item()
        elif len(construct_x['constructorRef']) > 1:
            c_id_val = construct_x['constructorId'].values[0]
        base.loc[base['TeamId']==constructor, 'constructorId']=c_id_val
    
    # set event query and get the track speed data 
    evnt_qry = pd.DataFrame(
        {"year":[year-1], # get the most recent year's speed data
         "name":[event['EventName'].item()],
         "circuitId":[c1['circuitId'].item()]}
    )
    speeds = get_track_speeds(event=evnt_qry)
    # print("[INFO]: track_speeds = \n{}".format(speeds)) # set base keys based on speed values
    speeds = pd.concat([speeds]*len(base), ignore_index=True)
    base = pd.concat([base.reset_index().drop('index', axis=1), speeds], axis=1)
    
    # standardize the points distribution
    fit_data, std_pt_features = std_pt_distrib(base)
    # print(fit_data.loc[fit_data['constructorId']==215, ['driverId', 'constructorId']])
    
    # set the data encodings
    # print(fitted_constructors)
    # print(fitted_drivers)
    _, _, _, data_window = get_encoded_data(
        fit_data, driver_vars=fitted_drivers, construct_vars=fitted_constructors)
    d_interactions = []
    c_interactions = []
    for var in vars: # add all interactions one-by-one
        data_window, d_interact = add_interaction(
            data_window, vars=[var], drivers=fitted_drivers, ret_term_names=True, debug=False, print_debug=False)
        d_interactions += d_interact
        
    # add null entries for all of the missing dirvers
        
    # get the subset of features we actually want
    m_vars = main_features + d_interactions + c_interactions + std_pt_features
    scaler = StandardScaler()
    data_window[m_vars] = scaler.fit_transform(data_window[m_vars])
    
    # print("[INFO]: data keys:")
    # for key in data_window.keys():
    #     print(key)
    
    return data_window[full_vars] 

def _fit_model(
    input_data,
    main_vars,
    response_var,
    cat_features,
    ratio={
        0:1,
        1:1,
        2:1,
        3:2,
        4:2,
        5:2
    },
    save_feature_coeffs=True,
    resample_data=False,
    dest_file='../../results/lasso_coeffs.csv'
):
    '''
    Args:
    - non_cats ------ list of variables that are not categorical
      variables and should be scaled.
    - ratio --------- dictionary of ratios for rounds 1 to n. 
      For example - the round number is given by x duplicates in 
      the data. This should set a much heavier weighting for the 
      most recent races.  
    - exclude_vars -- variables to be excluded from model fitting
    '''
    final_dat = None    
    # before duplicating the data - apply standard scaling to all 
    # of the numeric features
    scaler = StandardScaler()
    input_data[main_vars] = scaler.fit_transform(input_data[main_vars])
    
    # get the sorted list of race_ids based on the date of the event?
    input_data = input_data.sort_values(
        by=['year', 'round'], ascending=[True,True]
    )
    # print("[INFO]: na_dat = \n{}".format(input_data.isna().sum()))
    
    fit_dat = input_data.loc[~input_data['raceId'].isnull()]
    # get the series of raceIds - should be sorted
    unique_ids = fit_dat['raceId'].unique()
    
    # for r_id in input_data['raceId'].unique():
    #     print("race = {} | n = {}".format(r_id, input_data.loc[input_data['raceId']==r_id].shape[0]))
    
    # null_races = input_data.loc[~input_data['raceId'].isnull()]
    # print("[INFO]: null_races = \n{}".format(null_races))
    # print("[INFO]: Non-null matrix = {}".format(input_data.shape[0] - null_races.shape[0]))
    for i in range(len(unique_ids)):
        for r in range(ratio[i]):
            # duplicate data n times
            duplicates = input_data.loc[input_data['raceId']==unique_ids[i]]
            if final_dat is None:
                final_dat = duplicates
            else:
                final_dat = pd.concat([final_dat, duplicates], ignore_index=True)
        
    all_vars = main_vars + cat_features + [response_var] + ['raceId']
    # print("[INFO]: before dropping na's")
    # for r_id in final_dat['raceId'].unique():
    #     sub = final_dat.loc[final_dat['raceId']==r_id]
    #     na_sums = sub.isna().sum()
    #     na_sums = na_sums[na_sums>0]
    #     print("[INFO]: null vals = \n{}".format(na_sums))
    #     print("[INFO]: entries race {} | n = {} p = {}".format(r_id, sub.shape[0], sub.shape[0]/final_dat.shape[0]))
    #     print("---")
        
    final_dat = final_dat[all_vars].dropna() # drop null rows
    
    # resample the data for model fitting
    if resample_data==True:
        final_dat = final_dat.sample(n=len(final_dat), replace=True)
    # count number of rows for each unique race id
    
    # for r_id in final_dat['raceId'].unique():
    #     sub = final_dat.loc[final_dat['raceId']==r_id]
    #     print("[INFO]: entries race {} | n = {} p = {}".format(r_id, sub.shape[0], sub.shape[0]/final_dat.shape[0]))
    
    # drop the unimportant features
    y = final_dat[response_var]
    
    # include categoricals and main features
    full_vars = main_vars + cat_features
    # print("[INFO]: --- all fitted features ---\n{}".format(full_vars))
    
    X = final_dat[full_vars]
    # X = input_data.drop(exclude_vars, axis=1)
    
    # cat_indices = []
    # for col in X.keys():
    #     if col in cat_features:
    #         cat_indices.append(X.columns.get_loc(col))          
    # try:
    #     oversample = SMOTENC(
    #         categorical_features=cat_indices,
    #         # k_neighbors=y.sum()-1, 
    #         random_state=0
    #     )
    #     X, y = oversample.fit_resample(X, y)
    # except:
    #     oversample = SMOTE(
    #         k_neighbors=y.sum()-1
    #     )
    #     X, y = oversample.fit_resample(X, y)
    
    model = LassoCV(cv=5, random_state=42)
    model.fit(X, y)
    r2 = model.score(X, y)
    
    if not resample_data: print("[INFO]: Model Training Fit R2 = {}".format(r2))
    
    model_coefficients = pd.DataFrame({
        'Feature': X.columns, 
        'Coefficient': model.coef_
    })
    if save_feature_coeffs and not resample_data: 
        print("[INFO]: ---- saving lasso model coefficients ----")
        model_coefficients.to_csv(dest_file, index=False)
        
    return model

            
def fit_eval_window_model(
    main_features, # main features like prev points, etc.
    vars, # variables to interact with driver / team
    year=2025,
    k=6,
    round=3,
    target='positionOrder',
    predictions_folder="../../results",
    start_data="../../data/clean_model_data2.csv",
    drivers_data="../../data/drivers.csv",
    dest_file="../../results/lasso_coeffs.csv",
    constructors_data="../../data/constructors.csv",
    pred_round=None,
    boot_trials=1000,
    std_errors=True
):    
    '''
    Enumerate list of features to be included for fitting in the model
    '''
    # load the data and fetch the correct data window
    drivers_db = pd.read_csv(drivers_data)
    constructors_db = pd.read_csv(constructors_data)
    all_data = pd.read_csv(start_data)
    fit_data = get_data_in_window(k=k, yr=year, r_val=round, track_dat=all_data)
    # keys = ['TeamI]
    # print(fit_data.loc[fit_data['constructorId']==215, ['driverId', 'constructorId']])
    # print(fit_data.keys())
    # exit()
    
    # print("[INFO]: number of unique rounds = {}".format(fit_data['raceId'].unique()))
    
    # get standardized point distributions
    fit_data, std_pt_features = std_pt_distrib(fit_data)
    
    # NO MISSING DATA HERE
    # print("[INFO]: (after data loading) na data = \n{}".format(fit_data[vars].isna().sum()))
    # print("non missing = \n{}".format(fit_data[vars].count()))
    # fit lasso models to get the right features
    # no more podium prediction - just regress onto finishing position
    
    # reset indices to avoid concatenation issues 
    fit_data = fit_data.reset_index(drop=True)
    drivers, constructors, _, data_window = get_encoded_data(fit_data)
    d_interactions = []
    c_interactions = []
    
    check_vars = vars + drivers
    
    # print("[INFO]: (after feature encoding) na data = \n{}".format(data_window[check_vars].isna().sum()))
    # print("= non missing = \n{}".format(data_window[check_vars].count()))
    
    # re-subset the data for only non-na values
    data_window = data_window.loc[data_window[drivers].notna().any(axis=1)]
    
    # print("[INFO]: (after subsetting) na data = \n{}".format(data_window[check_vars].isna().sum()))
    # print("= non missing = \n{}".format(data_window[check_vars].count()))
    
    # print("[DEBUG]: --- drivers --- \n{}".format(drivers))
    # print("[DEBUG]: --- constructors --- \n{}".format(constructors))
    for var in vars: # add all interactions one-by-one
        data_window, d_interact = add_interaction(
            data_window, vars=[var], drivers=drivers, ret_term_names=True, debug=False, print_debug=False)
        # data_window, c_interact = add_interaction(
        #     data_window, vars=[var], constructors=constructors, ret_term_names=True)
        
        # print("[INFO]: d_interact = {}".format(d_interact))
        # print("[INFO]: c_interact = {}".format(c_interact))
        d_interactions += d_interact
        # c_interactions += c_interact
    
    # print("[INFO]: (after adding interactions) na data = \n{}".format(data_window[d_interactions].isna().sum()))
    # print("= non missing = \n{}".format(data_window[d_interactions].count()))
    
    m_feats = main_features + d_interactions + c_interactions + std_pt_features
    # print("[INFO]: ---- main features ---- \n{}".format(m_feats))
    # if len(m_feats) > len(fit_data):
    #     print("[ERROR]: full rank matrix - number of features = {} max features = {}".format(len(m_feats), len(fit_data)))
    #     exit()
    # else:
    # print(constructors)
    # print("[INFO]: total features = {} max features = {}".format(len(m_feats), len(fit_data)))
    if pred_round == None:
        pred_round = round+1
    
    X2 = get_forecast_data(
        pred_round, drivers_data=drivers_data, year=year, constructors_data=constructors_data,
        main_features=main_features, vars=vars, fitted_drivers=drivers, fitted_constructors=constructors,
        full_vars=m_feats + drivers + constructors
    )
    base_results = X2.copy()

    ## PROCEDURE
    #  1. iterate over n trials (n=1000) 
    #  2. at each trial, resample the training data and fit a model
    #  3. make predictions over the forecast data and save for each driver
    #  4. compute standard deviation over n predictions - set as the standard error of predictions
    #  5. make predictions over the normal fit data and plot the confidence intervals
    
    # TODO finish bootstrapping the standard errors of predictions for a lasso regression model
    if std_errors == True:
        q_results = pd.DataFrame({key:[] for key in drivers})
        r_results = pd.DataFrame({key:[] for key in drivers})
        for i in tqdm(range(boot_trials), ncols=100,
                  desc='processing trials', dynamic_ncols=True, leave=True):
            model1 = _fit_model(
                data_window,
                main_vars = m_feats,
                response_var=target[0],
                cat_features= drivers + constructors,
                save_feature_coeffs=False,
                resample_data=True,
                dest_file="../results/round{}_grid_lasso-coefs.csv".format(pred_round)
            )
            model2 = _fit_model(
                data_window,
                main_vars = m_feats,
                response_var=target[1],
                cat_features= drivers + constructors,
                save_feature_coeffs=False,
                resample_data=True,
                dest_file="../results/round{}_position-order_lasso-coefs.csv".format(pred_round)
            )

            #_x2 = X2.isna().sum()
            #print(_x2[_x2 > 0])
            #print(X2.loc[X2.isna().any(axis=1)][''])
            X2['prev_driver_position'] = X2['prev_driver_position'].fillna(20)
            X2 = X2.fillna(0)

            y1 = model1.predict(X2)
            y2 = model2.predict(X2)
            
            # set results
            base_results[target[0]] = y1
            base_results[target[1]] = y2
            
            # retrieve results from data frame
            r_res_dict = dict()
            q_res_dict = dict()
            
            for d in drivers:
                tmp = base_results.loc[base_results[d]==1]
                # set results in a list format
                r_res_dict[d] = [tmp[target[1]].values[0]]
                q_res_dict[d] = [tmp[target[0]].values[0]]
            
            r_df_tmp = pd.DataFrame(r_res_dict)
            q_df_tmp = pd.DataFrame(q_res_dict)
            
            r_results = pd.concat([r_results, r_df_tmp], axis=0).reset_index(drop=True)
            q_results = pd.concat([q_results, q_df_tmp], axis=0).reset_index(drop=True)
                 
    # 2. evaluate over all drivers
    # 3. evaluate over all constructors
    # 4. generate (track) interactions over the significant drivers
    # 5. generate (track) interactions over significant constructors
    
    # make predictions for input race year=2025, round=3
    print('ok 1')
    model1 = _fit_model(
        data_window,
        main_vars = m_feats,
        response_var=target[0],
        cat_features= drivers + constructors,
        save_feature_coeffs=True,
        resample_data=False,
        dest_file="../results/round{}_grid_lasso-coefs.csv".format(pred_round)
    )
    print('ok 2')
    model2 = _fit_model(
        data_window,
        main_vars = m_feats,
        response_var=target[1],
        cat_features= drivers + constructors,
        save_feature_coeffs=True,
        resample_data=False,
        dest_file="../results/round{}_position-order_lasso-coefs.csv".format(pred_round)
    )
    y1 = model1.predict(X2)
    y2 = model2.predict(X2)
    
    X2[target[0]] = y1
    X2[target[1]] = y2
    
    print("--- {} Predictions for Round {} of {} ---".format(target, pred_round, year))
    
    X2['Driver'] = np.nan
    X2['Constructor']=np.nan
    for d in drivers:
        id_val = float(d.split("_")[-1])
        driver_name = drivers_db.loc[drivers_db['driverId']==id_val, 'code'].values[0]
        q_std_err = q_results[d].std()
        r_std_err = r_results[d].std()
        X2.loc[X2[d]==1.0, ['Driver', 'std_err_q', 'std_err_r']] = [driver_name, q_std_err, r_std_err]
    for c in constructors:
        id_val = float(c.split("_")[-1])
        # print("id_val",id_val)
        constructor_name = constructors_db.loc[constructors_db['constructorId']==id_val, 'constructorRef'].values[0]
        # print("constructor",constructor_name)
        
        X2.loc[X2[c]==1.0, 'Constructor']=constructor_name
        # print(X2.loc[X2[c]==1.0, ['Driver', 'Constructor']])
    
    preds = X2[['Driver', 'Constructor', target[0], 'std_err_q', target[1], 'std_err_r']]
    preds['sp'] = preds['grid'].rank(method='min').astype(int)
    preds['fp'] = preds['positionOrder'].rank(method='min').astype(int)
    preds['position_change'] = preds['sp'] - preds['fp']
    
    preds['fantasy_pts'] = preds['position_change']*1
    for idx, pred in preds.iterrows():
        preds.loc[idx, 'fantasy_pts'] += dq_scores[pred['sp']]
        preds.loc[idx, 'fantasy_pts'] += dr_scores[pred['fp']]
    
    if not os.path.exists(predictions_folder): 
        os.mkdir(predictions_folder)
        
    preds = preds.sort_values(by='fp')
    print(preds)
    preds.to_csv(f"{predictions_folder}/predictions.csv", index=False)
    
    if std_errors == True:
        color_dict = {
            "PIA": "#FF8000",
            "NOR": "#FF8000", 
            "RUS": "#27F4D2",
            "ANT": "#27F4D2",
            "LEC": "#E80020", 
            "HAM": "#E80020",
            "VER": "#3671C6", 
            "TSU": "#3671C6", 
            "HAD": "#6692FF",
            "LAW": "#6692FF", 
            "STR": "#229971", 
            "ALO": "#229971", 
            "GAS": "#0093CC", 
            "DOO": "#0093CC",
            "COL": "#0093CC",  
            "ALB": "#64C4FF", 
            "SAI": "#64C4FF", 
            "BEA": "#B6BABD",
            "OCO": "#B6BABD",
            "BOR": "#52E252", 
            "HUL": "#52E252"
        }
        # plot qualifying results
        plt.clf()
        plt.figure(figsize=(12,6))
        for driver in preds['Driver'].unique(): 
            tmp = preds.loc[preds['Driver']==driver]
            plt.errorbar(driver, tmp['grid'].item(),
                         yerr=1.96*tmp['std_err_q'].item(), 
                         fmt='o', capsize=3, color=color_dict[driver])
        plt.xlabel("Driver")
        plt.ylabel("Expected Qualifying Position")
        plt.title("Qualifying Position by Driver with Deviations")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{predictions_folder}/quali_plot.jpg")
        
        # plot race results
        plt.clf()
        plt.figure(figsize=(12,6))
        for driver in preds['Driver'].unique(): 
            tmp = preds.loc[preds['Driver']==driver]
            plt.errorbar(driver, tmp['positionOrder'].item(),
                         yerr=1.96*tmp['std_err_r'].item(), 
                         fmt='o', capsize=5, color=color_dict[driver])
        plt.xlabel("Driver")
        plt.ylabel("Expected Race Results")
        plt.title("Race Finishing Position by Driver with Deviations")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{predictions_folder}/race_plot.jpg")
    
    
def models_by_window_2(
    start_yr, 
    start_r, 
    n, 
    data='../../data/clean_model_data.feather', 
    max_year=2024, 
    debug=False,
    model_type='xgb',
    fit_func=None,
    cat_features=None,
    feature_vals=None,
    dbg_count=None
):
    """
    Summarizes accuracy and f1 score for each window across all races in span
        
    Args:
    - start_yr ------ the first year of the span on which to test
    - start_r ------- the first round to be tested on in start_yr
    - n ------------- the biggest window size to test; windows from size 1 to n will be tested
    - data ---------- data file to choose
    - fit_func ------ a function to pass which automatically fits from the features. Default
      fit funcs are defined in param_train.py
    - cat_features -- list of categorical features to set as category types (optional)
    - features ------ total list of features to use (optional)

    Under the hood:
    - The function consists of 3 nested for loops (n, year, round)
    - Every iteration of the round loop collects the correct training and test windows for that specific race
    - Each iteration also creates a logistic regression model and predicts using the test window
    - These predictions are then classified where the top 3 probabilities in each round are given a Podium Finish
    - Then the accuracy and f1_score are calculated and added to a dataframe for the current value of n
    - Once this process has been completed for all races in an iteration of n,
    the average accuracy and f1_score for that value of n is added to a different dataframe
    - After all iterations of n, the dataframe containing the averages is returned
    """
    if n < 2: 
        print("[ERROR]: n races training window must be greater than 2")
        return 
    
    get_podiums = True # set true for using the add_podiums func to augment data

    if data.split(".")[-1] == "feather":
        all_data = pd.read_feather(data)
    elif data.split(".")[-1] == 'csv':
        all_data = pd.read_csv(data)

    if debug: print("[DEBUG]: before standardizing point distribution --- \n{}".format(all_data))
    all_data, std_pt_features = std_pt_distrib(all_data) # return feature names 
    if debug: 
        print("[DEBUG]: after standardizing point distribution ---- \n{}".format(all_data))
        if dbg_count == 1: exit()
    
    results_df = pd.DataFrame(columns=['n', 'Accuracy', 'F1 Score'])

    # For each model of window size 1 to n
    for i in tqdm(range(2,n+1), ncols=100, total=n-1, 
                  desc='processing trials', dynamic_ncols=True, leave=True):
        single_n_results_df = pd.DataFrame(columns=['Accuracy', 'F1 Score'])

        # for race from start_yr, start_r to most recent race
        for year in range(start_yr, max_year):
            if year == start_yr:
                num_rounds = len(all_data[all_data['year'] == year]['round'].unique())
                first_r = start_r 
            elif year == 2023:
                num_rounds = 12 
                first_r = 1
            else:
                num_rounds = len(all_data[all_data['year'] == year]['round'].unique()) 
                first_r = 1

            for round in range(first_r, num_rounds+1):
                # print(i, year, round) # just a test
                
                # get data window of current size and current race
                # r_val should be r_val - 1
                train_window = get_data_in_window(k=i+1, yr=year, r_val=round, track_dat=all_data)
                # train_window, podium_keys = add_podiums(n=i+1, data=all_data, year = year, round = round)
                # if debug: print("[DEBUG]: podium_keys = {}".format(podium_keys))
                # train_window['alt'] = train_window['alt'].astype(float)
                # test_window['Podium Finish'] = ['Yes' if position <= 3 else 'No' for position in test_window['positionOrder']]
                train_window['Podium Finish'] = ['Yes' if position <= 3 else 'No' for position in train_window['positionOrder']]
                train_window = train_window.reset_index().drop(['index'],axis=1)
                
                # handle the encoding directly now!!
                if model_type == 'xgb':
                    drivers_train, constructors_train, _, train_window = get_encoded_data(train_window)
                
                # if not xgb or lightgbm - handle directly
                if model_type!='xgb' and model_type!='lightgbm':
                    drivers_train, constructors_train, _, df_train = get_encoded_data(train_window)

                # remove interactions for all tree models except Logistic
                if model_type == 'logistic':
                    train_window = add_interaction(df_train, vars=['corner_spd_min'], drivers=drivers_train)

                test_window = train_window[(train_window['round'] == round) & (train_window['year'] == year)]
                train_window = train_window.drop(train_window[(train_window['round'] == round) & (train_window['year'] == year)].index)
                
                if debug: print("[DEBUG]: train_window.keys = {}".format(train_window.keys()))

                #print(train_window['round'].unique()) # another test
                # select features using the data window
                if debug: 
                    print("[DEBUG]: train_window.keys() = \n{}".format(train_window.keys()))
                
                # set training features
                if feature_vals is None:
                    feature_list = [
                                    'Podium Finish', 
                                    'prev_driver_points',
                                    # 'constructorId_1',
                                    # 'constructorId_9',
                                    # 'max_track_spd',
                                    # 'prev_construct_wins',
                                    # 'driverId_844'
                                    ]
                    # feature_list += podium_keys[1:4]
                    if debug: print("[DEBUG]: podium keys = {}".format(podium_keys))
                     # add top 3 podiums
                    train_features = train_window[feature_list]
                else:
                    if 'Podium Finish' not in feature_vals:
                        feature_vals.append('Podium Finish')
                    train_features = train_window[feature_vals + drivers_train + constructors_train + std_pt_features]
                # set categorical feature dtypes for xgboosting
                # if cat_features is not None:
                #     train_features[cat_features] = train_features[cat_features].astype('category')

                train_features = train_features.dropna() # drop everything with na values
                # train_features['alt'] = train_features['alt'].astype(float)
                features = train_features.columns.drop(['Podium Finish'])

                if debug: 
                    print("[DEBUG]: y output = {}".format(
                        train_features['Podium Finish'] == 'Yes')
                    )
                    print("[DEBUG]: features.dtypes = {}".format(train_features.dtypes))
                
                # design = MS(features)
                # X = design.fit_transform(train_features)
                # y = train_features['Podium Finish'] == 'Yes'
                # lr = sm.GLM(y,
                #             X,
                #             family = sm.families.Binomial())
                # lr_results = lr.fit()

                # # get predicted probabilities for current race
                # test = MS(features).fit_transform(test_window)
                # probabilities = lr_results.predict(test)

                # skip empty iterations
                if train_features.shape[0] == 0: continue
                if test_window.shape[0] == 0: continue

                probabilities = fit_func(train_features, test_window, info=False, smote=True, f_select=False)
                
                # classify predictions
                n_outs = test_window.shape[0]
                labels = np.array(['Yes'] * n_outs) # match num yesses to num outputs
                no_indices = np.argpartition(probabilities, n_outs-3)[:n_outs-3] # where n_outs = 20, index is 17
                labels[no_indices] = 'No'
                
                # Find accuracy and f1 score
                if debug: 
                    print("[DEBUG]: labels.shape = {}".format(labels.shape))
                    print("[DEBUG]: test_window['Podium Finish'].shape = {}".format(test_window['Podium Finish'].shape))
                    print("[DEBUG]: test_window['driverId'].unique() = {}".format(test_window['driverId'].unique()))
                
                test_accuracy = np.mean(labels == test_window['Podium Finish'])
                test_f1 = f1_score(np.array(test_window['Podium Finish']), labels, pos_label='Yes')

                # Add to singlular df
                single_n_results_df = single_n_results_df._append({'Accuracy': test_accuracy, 'F1 Score': test_f1}, ignore_index=True) 

                # if debug:
                #     return None
        
        # take average accuracy and f1 score for window size and add to a dataframe
        results_df = results_df._append({'n': i, 
                                         'Accuracy': single_n_results_df['Accuracy'].mean(), 
                                         'F1 Score': single_n_results_df['F1 Score'].mean(),
                                         'Acc_Max': single_n_results_df['Accuracy'].max(),
                                         'Acc_Min': single_n_results_df['Accuracy'].min(),
                                         'F1_Max': single_n_results_df['F1 Score'].max(),
                                         'F1_Min': single_n_results_df['F1 Score'].min()}, ignore_index = True)
    
    # return results dataframe
    return results_df

def main1():
    cat_features = ['circuitId', 'driverId', 'constructorId']
    features = [
        # 'circuitId', 
        'driverId', 'constructorId', #'alt', 
        # 'tempmax', 'tempmin', 'temp', 'dew', 
        # 'humidity', 'precip', 'precipcover', 
        # 'preciptype', 'windspeed', 'winddir', 
        'prev_driver_points', 'prev_construct_points', 
        'prev_construct_position', 'prev_construct_wins', 
        'strt_len_mean', 'strt_len_q1', 'strt_len_median', 
        'strt_len_q3', 'strt_len_max', 'strt_len_min', 
        'str_len_std', 'avg_track_spd', 'max_track_spd', 
        'min_track_spd', 'std_track_spd', 'corner_spd_mean', 
        'corner_spd_q1', 'corner_spd_median', 'corner_spd_q3', 
        'corner_spd_max', 'corner_spd_min', 'num_slow_corners', 
        'num_fast_corners', 'num_corners', 'circuit_len', 'year'
    ]
    # x = pd.read_csv("../../data/clean_model_data2.csv")
    # print(x['alt'])
    # print(x['preciptype'].isna().sum())
    # print(x.loc[x['year']==2024].shape, 'n results from 2024')
    # print(x.shape, 'before drop na')
    # x = x[features].dropna()
    # print(x.loc[x['year']==2024].shape, 'n results from 2024')
    # print(x.shape, 'after drop na')
    # print((x == '\\N').sum())
    confidence_eval= False
    if confidence_eval == False:
        res = models_by_window_2(2024, 1, 10, 
                            data='../../data/clean_model_data2.csv',
                            max_year=2025, 
                            model_type='xgb', 
                            debug=False,
                            fit_func=xgb_fit,
                            cat_features=cat_features,
                            feature_vals=None,
                            dbg_count=None)
        print(res)
        res.to_csv("../experiments/un-scaled_points_sample.csv", index=False)
    
    elif confidence_eval: 
        n_pts = 15
        results = np.zeros(n_pts)
        for i in range(n_pts):
            res = models_by_window_2(2024, 1, 14, 
                            data='../../data/clean_model_data2.csv',
                            max_year=2025, 
                            model_type='xgb', 
                            debug=False,
                            fit_func=xgb_fit,
                            cat_features=cat_features,
                            feature_vals=None,
                            dbg_count=None)
            results[i] = res.loc[res['n']==7,'F1 Score'].item()
        
        from scipy.stats import norm

        alpha = 0.05
        crit_point = norm.ppf(1 - alpha/2)
        # normal distribution by CLT

        dev = crit_point * (results.std()/(n_pts)**0.5)
        min_pt = results.mean() - dev
        max_pt = results.mean() + dev
        
        print(f"[INFO]: confidence interval = {results.mean()} +/- {dev}")
        print(f"[INFO]: explicit confidence interval = [{min_pt}, {max_pt}]")

def main2():
    '''
    Some stuff
    '''
    main_features = [
        # 'prev_driver_points',
        'prev_driver_position',
        'prev_driver_wins',
        # 'prev_construct_points',
        'prev_construct_position',
        'prev_construct_wins',
    ]
    vars = [
        # 'strt_len_mean',
        # 'strt_len_q1',
        'strt_len_median',
        # 'strt_len_q3',
        # 'strt_len_max',
        'strt_len_min',
        # 'str_len_std',
        'avg_track_spd',
        # 'max_track_spd',
        'corner_spd_median',
        # 'corner_spd_q3',
        'corner_spd_max',
        'corner_spd_min',
        'num_slow_corners',
        'num_fast_corners',
        # 'num_corners',
        # 'circuit_len'
    ]
    if __name__ == "__main__":
        start_data = "../../data/clean_model_data2.csv"
        drivers_data="../../data/drivers.csv"
        dest_file="../../results/lasso_coeffs.csv",
        constructors_data="../../data/constructors.csv"
    else:
        start_data = "../data/clean_model_data2.csv"
        drivers_data="../data/drivers.csv"
        dest_file="../results/lasso_coeffs.csv"
        constructors_data="../data/constructors.csv"
        
    # NOTE: code currently set up to run from main.py in 
    # code. Do not run this file directly
    fit_eval_window_model(
        main_features=main_features,
        vars=vars,
        k=5,
        round=14,
        year=2025,
        target=['grid','positionOrder'],
        predictions_folder="../results/hungary",
        start_data=start_data,
        drivers_data=drivers_data,
        dest_file=dest_file,
        constructors_data=constructors_data,
        pred_round=14,
        std_errors=True,
        boot_trials=100
    )
    
if __name__ == "__main__":
    print("[ERROR]: DO NOT RUN MODULE DIRECTLY")
    # main2()