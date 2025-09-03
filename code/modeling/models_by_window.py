import pandas as pd

try:
    from ISLP.models import (ModelSpec as MS, summarize)
    import statsmodels.api as sm
    from ISLP import confusion_table
except:
    print("[ERROR]: MISSING ISLP and STATSMODELS")
import numpy as np
import fastf1
import os

from sklearn.metrics import f1_score, r2_score
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# predictions report
from treeinterpreter import treeinterpreter as ti

# visualization
import seaborn as sns
import matplotlib.pyplot as plt 

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
    from colors import Colors
    from agg_fp2_laps import (
        get_driver_session_ranks,
        get_new_pred,
        get_new_pred_alt
    )
except: # import as module
    # from .param_train import (
    #     logistic_fit,
    #     xgb_fit
    # )
    from modeling.agg_fp2_laps import (
        get_driver_session_ranks,
        get_new_pred_alt, 
        get_new_pred,
        recalc_fantasy_score
    )
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
    from modeling.colors import Colors
    
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
    print(event)
    print(c1) # DEBUG - multi/single item event query for circuit?

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
    model_type = 'LASSO',
    main_features_only = False, 
    save_feature_coeffs=True,
    show_tree = True,
    resample_data=False,
    dest_file='../../results/lasso_coeffs.csv',
    results_folder='../../results',
    output_tree_file_name='sample_dt_quali.jpg'
):
    '''
    NOTE: this function works only with LASSO in its feature
    selection mode. Of course, you can just fit a lasso regression model
    with all features and use the output model directly. Otherwise,
    using `main_features_only=True` with `model_type='LASSO'` will not 
    work as expected. The same kind of logic is applied to 'full_vars'

    Args:
    - non_cats ------ list of variables that are not categorical
      variables and should be scaled.
    - ratio --------- dictionary of ratios for rounds 1 to n. 
      For example - the round number is given by x duplicates in 
      the data. This should set a much heavier weighting for the 
      most recent races.  
    - exclude_vars -- variables to be excluded from model fitting
    - model_type (str): Three options -
      - LASSO: uses the LassoCV regression module in order to 
        automatically determine the best l1 penalization parameter value
        and fit a regression model
      - RF: uses random forest random forest regression model
      - DT: uses decision tree regression model
    - main_features_only (bool): set to true if you have a final 
      list of features to use, and do not want to deal with categorical
      feature encodings explicitly
    Returns:
    - fitted model
    '''
    if model_type != "LASSO" and main_features_only == False:
        print("{}[WARNING]: model type is {} but main_features_only=False{}".format(
            Colors.YELLOW, model_type, Colors.ENDC)
        )
    final_dat = None    
    # before duplicating the data - apply standard scaling to all of the numeric features
    scaler = StandardScaler()
    
    # because the forecast data uses centered data, so does the training data
    input_data[main_vars] = scaler.fit_transform(input_data[main_vars])
    
    fit_dat = input_data.loc[~input_data['raceId'].isnull()]
    # get the series of raceIds - should be sorted
    unique_ids = fit_dat['raceId'].unique()
    
    for i in range(len(unique_ids)):
        for r in range(ratio[i]):
            # duplicate data n times
            duplicates = input_data.loc[input_data['raceId']==unique_ids[i]]
            if final_dat is None:
                final_dat = duplicates
            else:
                final_dat = pd.concat([final_dat, duplicates], ignore_index=True)
        
    if main_features_only == True:
        all_vars = main_vars + [response_var] + ['raceId']
    else:
        all_vars = main_vars + cat_features + [response_var] + ['raceId']
        
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
    if not main_features_only:
        full_vars = main_vars + cat_features
    else:
        full_vars = main_vars
    
    X = final_dat[full_vars] # subset data according to full_vars list
    
    if model_type == "LASSO":
        model = LassoCV(
            alphas = np.logspace(-4, 2, 50),  # evaluate 50 alpha values from 10^-4 to 10^2
            cv=5, random_state=42
        )
    if model_type == "RF":
        model = RandomForestRegressor(
            # min_samples_leaf = 6,
            # max_depth = 8,
            # min_samples_split = 10,
            # max_features = 4  # increase max number of features
        )
    if model_type == 'DT':
        model = DecisionTreeRegressor()

    model.fit(X, y)
    r2 = model.score(X, y)

    if show_tree and model_type == "DT":
        plt.figure(figsize=(15,15))
        plot_tree(model, feature_names=X.columns, class_names=[response_var], filled=True)
        plt.savefig(f'{results_folder}/{output_tree_file_name}')
    
    if not resample_data: print("[INFO]: Model Training Fit R2 = {}".format(r2))
    
    # if using LASSO regression, save and return the feature coefficients
    if model_type == "LASSO":
        model_coefficients = pd.DataFrame({
            'Feature': X.columns, 
            'Coefficient': model.coef_
        })
        # select features with non-zero coefficients
        model_coefficients = model_coefficients[np.abs(model_coefficients['Coefficient']) > 0]
        if save_feature_coeffs and not resample_data: 
            print("[INFO]: ---- saving lasso model coefficients ----")
            model_coefficients.to_csv(dest_file, index=False)

        return (model, model_coefficients['Feature'].tolist())
    # if using any other type of modeling, just return the fitted model
    else:
        return (model, None)
    

def plot_coeffs(
    coefficients_df:pd.DataFrame,
    drivers_df:pd.DataFrame,
    out_path:str,
    plot_title:str = 'Race Model Coefficient Scores', 
    xlab:str = 'Coefficient Scores',
    ylab:str = 'Feature'
):
    '''
    Args:
    - coefficients_df (pd.DataFrame): pandas dataframe with two columns
      col1 = 'Feature' and col2 = 'Coefficient'. The first contains a column 
      with features - many of which will contain a driver ID indicator, whereas
      the coefficient column simply contains the corresponding regression coefficient
      for the given feature
    - drivers_df (pd.DataFrame): pandas dataframe which at minimum contains columns
      'driverId' and 'driverRef' - driver Id's correspond to the driverId's mentioned
      in many features, and the driverRef corresponds to their most representative
      name/identifier
    - out_path (str): path to save the output plot to
    Returns:
    - new_coeff_df (pd.DataFrame): an updated version of the existing coefficient 
      dataframe with updated (more interpretable) feature names 
    '''
    new_coeff_df = coefficients_df.copy()
    new_driver_df = drivers_df.copy()

    # extract driver Id's from feature names
    new_coeff_df['Feature'] = new_coeff_df['Feature'].str.replace('-', '_') 
    ids = [x[1] for x in new_coeff_df['Feature'].str.split("_").tolist()]

    new_coeff_df['driverId'] = ids
    new_driver_df['driverId'] = new_driver_df['driverId'].astype(float).astype(str)

    new_coeff_df = pd.merge(
        left = new_coeff_df,
        right = new_driver_df[['driverId', 'driverRef']],
        how = 'left', 
        on = 'driverId'
    ) # fill null values with empty strings

    new_coeff_df.loc[pd.isnull(new_coeff_df['driverRef']), 'driverRef'] =\
          new_coeff_df.loc[pd.isnull(new_coeff_df['driverRef']), 'driverId']

    # use driver Id values to fill in names
    new_coeff_df['Feature'] = new_coeff_df.apply(
        lambda row: row['Feature'].replace(row['driverId'], row['driverRef']), axis=1
    )

    # sort the data for plotting
    new_coeff_df = new_coeff_df.sort_values(by='Coefficient')

    # plot the coefficients
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10,10))
    sns.barplot(x='Coefficient', y='Feature', data=new_coeff_df, palette='viridis')
    plt.title(plot_title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(out_path)

    # return the updated df
    return new_coeff_df


def print_report(
    feature_contributions,
    report_path
):
    '''
    Prints predictions report for each of the drivers - top 10 features and 
    such using treeinterpreter

    Parameters:
    - feature_contributions (dict): contains a dictionary of format 
      'driver': contributions (contributions output from ti.predict)
    - report_path (path[str]): path to the report output file.
    Returns:
    - None
    '''
    with open(report_path, 'w') as f:
        for driver in feature_contributions:
            pred, out_df, bias = feature_contributions[driver]
            f.write(f'[DRIVER]: {driver} [PRED]: {pred} [BIAS]: {bias}\n')
            f.write(f'---')
            f.write(f'{out_df.to_string()}\n\n')

            
def fit_eval_window_model(
    main_features, # main features like prev points, etc.
    vars, # variables to interact with driver / team
    year=2025,
    k=6,
    round=3,
    target=['positionOrder', 'grid'],
    predictions_folder="../../results",
    start_data="../../data/clean_model_data2.csv",
    drivers_data="../../data/drivers.csv",
    dest_file="../../results/lasso_coeffs.csv",
    constructors_data="../../data/constructors.csv",
    pred_round=None,
    boot_trials=1000,
    std_errors=True,
    print_preds=True,
    plot_model_coef=True,
    eval_mode=True,
    output_feature_report=True,
    output_dt_vis=False,
    fp2_adjust=True,
    model_type='LASSO',
    adjustment_session='FP2'
):    
    '''
    Follows the specified regression approach to model race outcomes
    for the given prediction round based on data from the prior k
    races. 

    Parameters:
    - main_features (list[str]): list of features names to use in 
      fitting the model in addition to individual drivers and constructors.
      These are included by default - ALWAYS
    - vars (list[str]): list of features which should be interacted with 
      drivers / constructors. For example, one might 'num_med_corners'
      which would result in an interaction like 
      yuki_tsunoda * num_med_corners - how is Yuki's mean performance affected
      by the number of medium speed corners at a given track?
    - year (int): The year over which to do the modeling
    - k (int): Number of prior racing rounds of data to fit on
    - target (list[str]): A two-tuple will also work here. The first target
      should be the feature name for the race outcome, and the second the qualifying
      outcome. 
    - predictions_folder (path[str] REQUIRED): a path to save all modeling outputs
      to. 
    - start_data (path[str]): path to the data required
    - drivers_data (path[str]): path to the drivers reference data required
    - constructors_data (path[str]): path to the constructors reference data required
    - pred_round (int, None): The round number to make predictions for
    - boot_trials (int): The number of trials to use in bootstrapping the standard
      errors of model predictions. Only relevant if `std_errors = True`
    - std_errors (bool): If true, will use bootstrapping the standard error of
      predictions for each driver
    - print_preds (bool): If true, will print the predictions for a given race
    - plot_model_coef (bool): If true, will plot the coefficients of the fitted
      feature selection (lasso regression) model and save it to the predictions
      folder.
    - eval_mdoe (bool): If true, will return
    - model_type (str): type of model to use. options=['RF', 'LASSO'] 
    - adjustment_session (str): options=['FP2', 'FP1', 'FP3'] - the free practice
      session to use as a pace benchmark to adjust predictions

    Returns:
    - predictions (pd.DataFrame, None): (if eval_mode = True)
      the output predictions for the given 
      prediction round in dataframe form. If eval_mode is false, nothing will
      be returned by this function
    '''
    # load the data and fetch the correct data window
    drivers_db = pd.read_csv(drivers_data)
    constructors_db = pd.read_csv(constructors_data)
    all_data = pd.read_csv(start_data)
    fit_data = get_data_in_window(k=k, yr=year, r_val=round, track_dat=all_data)
    
    # get standardized point distributions
    fit_data, std_pt_features = std_pt_distrib(fit_data)
    
    # reset indices to avoid concatenation issues 
    fit_data = fit_data.reset_index(drop=True)
    drivers, constructors, _, data_window = get_encoded_data(fit_data)
    d_interactions = []
    c_interactions = []
    
    check_vars = vars + drivers
    
    # re-subset the data for only non-na values
    data_window = data_window.loc[data_window[drivers].notna().any(axis=1)]
    for var in vars: # add all interactions one-by-one
        data_window, d_interact = add_interaction(
            data_window, vars=[var], drivers=drivers, ret_term_names=True, debug=False, print_debug=False)
            
        d_interactions += d_interact
    
    m_feats = main_features + d_interactions + c_interactions + std_pt_features
    
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

    # obtain model features for race and qualifying models

    # only use lasso feature selection for 'non-lasso' runs
    # if not model_type == "LASSO":
    _, race_features = _fit_model(
        data_window.copy(), 
        main_vars=m_feats, 
        response_var = target[0],
        cat_features = drivers + constructors,
        save_feature_coeffs = True, 
        resample_data = False, 
        model_type = 'LASSO', 
        main_features_only = False,
        dest_file = "{}/lasso_coeffs_race.csv".format(predictions_folder)
    )
    _, quali_features = _fit_model(
        data_window.copy(),
        main_vars = m_feats,
        response_var = target[1],
        cat_features = drivers + constructors,
        save_feature_coeffs = True, 
        resample_data = False, 
        model_type = 'LASSO', 
        main_features_only = False,
        dest_file = "{}/lasso_coeffs_quali.csv".format(predictions_folder)
    )
    
    if std_errors == True:
        q_results = pd.DataFrame({key:[] for key in drivers})
        r_results = pd.DataFrame({key:[] for key in drivers})

        # if lasso regression - use full feature set
        if model_type == 'LASSO':
            race_features = m_feats
            quali_features = m_feats

        for i in tqdm(range(boot_trials), ncols=100,
                  desc='processing trials', dynamic_ncols=True, leave=True):
            
            model1 = _fit_model(
                data_window.copy(),
                main_vars = race_features,
                response_var=target[0],
                cat_features= drivers + constructors,
                save_feature_coeffs=False,
                resample_data=True,
                model_type = model_type,
                main_features_only = True,
                dest_file="../results/round{}_grid_lasso-coefs.csv".format(pred_round)
            )[0] # take the first output
            model2 = _fit_model(
                data_window.copy(),
                main_vars = quali_features,
                response_var=target[1],
                cat_features= drivers + constructors,
                save_feature_coeffs=False,
                resample_data=True,
                model_type = model_type,
                main_features_only = True,
                dest_file="../results/round{}_position-order_lasso-coefs.csv".format(pred_round)
            )[0]

            X2['prev_driver_position'] = X2['prev_driver_position'].fillna(20)
            X2 = X2.fillna(0)

            y1 = model1.predict(X2[race_features])
            y2 = model2.predict(X2[quali_features])
            
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
    
    # make predictions for input race year=2025, round=3 using a random forest model
    model1 = _fit_model(
        data_window.copy(),
        main_vars = race_features,
        response_var=target[0],
        model_type = model_type,
        cat_features= drivers + constructors,
        save_feature_coeffs=False,
        main_features_only = True,
        resample_data=False,
        # dest_file="{}/round{}_grid_lasso-coefs.csv".format(predictions_folder, pred_round)
    )[0]

    if output_dt_vis:
        # generate decision tree for visualization
        model1_b = _fit_model(
            data_window,
            main_vars = race_features,
            response_var=target[0],
            model_type = 'DT',
            cat_features= drivers + constructors,
            save_feature_coeffs=False,
            main_features_only = True,
            resample_data=False,
            show_tree=True,
            results_folder=predictions_folder,
            output_tree_file_name='race_model.pdf'
        )[0]
        
    # qualifying models
    model2 = _fit_model(
        data_window.copy(),
        main_vars = quali_features,
        response_var=target[1],
        model_type = model_type,
        cat_features= drivers + constructors,
        save_feature_coeffs=False,
        main_features_only = True,
        resample_data=False,
    )[0]
    
    if output_dt_vis:
        model2_b = _fit_model(
            data_window,
            main_vars = quali_features,
            response_var=target[1],
            model_type = 'DT',
            cat_features= drivers + constructors,
            save_feature_coeffs=False,
            main_features_only = True,
            resample_data=False,
            show_tree=True,
            results_folder=predictions_folder,
            output_tree_file_name='quali_model.pdf'
        )[0]

    if output_feature_report and model_type == 'RF':
        race_pred_dict = dict()
        quali_pred_dict = dict()

        # find driver matches and do preds 1-by-1
        for d in drivers: # iterate through list of driver cat feature names
            id_val = float(d.split("_")[-1])
            sample = X2.loc[X2[d] == 1.0]
            driver_name = drivers_db.loc[drivers_db['driverId']==id_val, 'code'].values[0]
            
            # make predictions
            pred_r, bias_r, contrib_r = ti.predict(model1, sample[race_features])
            pred_q, bias_q, contrib_q = ti.predict(model2, sample[quali_features])

            # output display dfs
            r_out = pd.DataFrame(
                {'Input Value': sample[race_features].to_numpy()[0], 'Feature Score': contrib_r[0]}, 
                index=race_features)
            r_out = r_out.loc[r_out['Feature Score']!=0].sort_values(by='Feature Score')

            q_out = pd.DataFrame(
                {'Input Value': sample[quali_features].to_numpy()[0], 'Feature Score': contrib_q[0]},
                index=quali_features,
            )
            q_out = q_out.loc[q_out['Feature Score']!=0].sort_values(by='Feature Score')

            # save for output
            race_pred_dict[driver_name] = [pred_r, r_out, bias_r]
            quali_pred_dict[driver_name] = [pred_q, q_out, bias_q]
        
        print_report(race_pred_dict, f'{predictions_folder}/race_report.txt')
        print_report(quali_pred_dict, f'{predictions_folder}/quali_report.txt')

    y1 = model1.predict(X2[race_features])
    y2 = model2.predict(X2[quali_features])
    
    X2[target[0]] = y1
    X2[target[1]] = y2
    
    # the predictions general output requires that standard errors be given
    if print_preds == True and std_errors == True:
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
            constructor_name = constructors_db.loc[constructors_db['constructorId']==id_val, 'constructorRef'].values[0]
            X2.loc[X2[c]==1.0, 'Constructor']=constructor_name
        
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

        if adjustment_session == None:
            fp2_adjust = False 

        if fp2_adjust == True:
            # use standard errors and fp2 data to update predictions
            ranks = get_driver_session_ranks(round=pred_round, session_type=adjustment_session, base_predictions=preds, approach='cluster')
            z = pd.merge(preds, ranks, on='Driver')
            z['adj_pred_order2'] = z.apply(get_new_pred_alt,axis=1)

            # replace existing predictiong order value
            z['fp'] = z['adj_pred_order2'].rank(method='min', ascending=True)
            z = recalc_fantasy_score(z, dq_scores, dr_scores)

            z = z.sort_values(by='fp')
            z.to_csv(f'{predictions_folder}/predictions.csv', index=False)


    if plot_model_coef == True:
        quali_coeffs_df = pd.read_csv("{}/lasso_coeffs_quali.csv".format(predictions_folder))
        race_coeffs_df = pd.read_csv("{}/lasso_coeffs_race.csv".format(predictions_folder))

        # plot quali coefficients
        quali_coeffs_df = plot_coeffs(
            coefficients_df = quali_coeffs_df, 
            drivers_df = drivers_db,
            out_path = "{}/race_coeff_plot.jpg".format(predictions_folder),
            plot_title = "Race Model Coefficients Plot",
        )

        # plot race coefficients
        race_coeffs_df = plot_coeffs(
            coefficients_df = race_coeffs_df, 
            drivers_df = drivers_db, 
            out_path = "{}/quali_coeff_plot.jpg".format(predictions_folder),
            plot_title = "Quali Model Coefficients Plot"
        )
        
        # save new coefficients
        quali_coeffs_df.to_csv("{}/lasso_coeffs_quali.csv".format(predictions_folder), index=False)
        race_coeffs_df.to_csv("{}/lasso_coeffs_race.csv".format(predictions_folder), index=False)

    if eval_mode == True:
        round_subset = all_data.loc[all_data['round']==pred_round, ['driverId', 'positionOrder', 'grid']].rename(
            columns={'positionOrder': 'positionOrder_true', 'grid': 'grid_true'}
        )  

        for d in drivers:
            id_val = float(d.split("_")[-1])
            driver_id = drivers_db.loc[drivers_db['driverId']==id_val, 'driverId'].values[0]
            X2.loc[X2[d]==1.0, 'driverId'] = driver_id

        output_vals = X2[['driverId', target[0], target[1]]].rename(
            columns={target[0]: 'positionOrder_pred', target[1]: 'grid_pred'}
        )
        output_vals['sp_pred'] = output_vals['grid_pred'].rank(method='min').astype(int)
        output_vals['fp_pred'] = output_vals['positionOrder_pred'].rank(method='min').astype(int)

        ret_df = pd.merge(
            left=output_vals, 
            right=round_subset, 
            how='left', 
            on='driverId', 
        )

        return ret_df
    
    elif eval_mode == False and (std_errors==True and print_preds==True):
        return preds # should be given for the conditions above
    
    else:
        return None
    
    
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


def main2(
    main_features = [
        # 'prev_driver_points',
        'prev_driver_position',
        'prev_driver_wins',
        # 'prev_construct_points',
        'prev_construct_position',
        'prev_construct_wins',
    ],
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
        'num_med_corners'
        # 'num_corners',
        # 'circuit_len'
    ],
    start_data = '../data/clean_model_data2.csv',
    drivers_data = '../data/drivers.csv',
    dest_file = '../results/lasso_coeffs.csv',
    constructors_data = '../data/constructors.csv',
    predictions_folder = '../results/hungary',
    pred_round = 14,
    k = 5,
    year = 2025,
    std_errors = True,
    boot_trials = 100,
    output_feature_report = True,
    adjust_session = 'FP1',
    model_type = 'LASSO'
):
    '''
    Some stuff
    '''
    # NOTE: code currently set up to run from main.py in 
    # code. Do not run this file directly
    if not os.path.exists(predictions_folder):
        os.mkdir(predictions_folder)

    fit_eval_window_model(
        main_features=main_features,
        vars=vars,
        k=k,
        round=pred_round, # fits model up to the round before the prediction round
        year=year,
        target=['grid','positionOrder'],
        predictions_folder=predictions_folder,
        start_data=start_data,
        drivers_data=drivers_data,
        dest_file=f'{predictions_folder}/lasso_coeff.csv',
        constructors_data=constructors_data,
        pred_round=pred_round,
        std_errors=std_errors,
        boot_trials=boot_trials,
        output_feature_report=output_feature_report,
        adjustment_session=adjust_session,
        model_type=model_type
    )


def eval_model(
    main_features=[
        'prev_driver_position', 
        'prev_driver_wins', 
        'prev_construct_position', 
        'prev_construct_wins'
    ],
    vars = [
        'strt_len_median', 'strt_len_min',
        'avg_track_spd', 'corner_spd_median', 
        'corner_spd_max', 'corner_spd_min',
        'num_slow_corners', 'num_fast_corners', 
        'num_med_corners'
    ],
    start_data = '../data/clean_model_data2.csv',
    drivers_data = '../data/drivers.csv', 
    constructors_data = '../data/constructors.csv',
    k = 5,
    year = 2025,
    result_folder = '../results/model_metrics',
    output_feature_report = False
):
    '''
    Runs model evaluation for the configured modeling approach
    with fit_eval_model()

    Args:
    - main_features (list[str]): list of main feature names + drivers
      to use
    - vars (list[str]): list of track feature names to interact with 
      drivers and constructors
    - start_data (path[str]): the path to the data to use for model fitting
    - k (int): the number of previous rounds to use for fitting
    - year (int): the year over which to do the predictions
    - result_folder (path[str]): path to the results output folder to save
      model metrics to
    Returns:
    - None
    '''
    tmp_read = pd.read_csv(start_data)
    max_round = tmp_read['round'].max()
    del tmp_read

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    # init to None
    all_preds = None
    start_round = k+1
    for i in tqdm(range(max_round - k)):
        cur_round = start_round + i
        preds = fit_eval_window_model(
            main_features = main_features, 
            vars = vars, 
            year = year, 
            k = k, 
            std_errors=False,
            round = cur_round, 
            target = ['positionOrder', 'grid'], 
            predictions_folder = result_folder, 
            start_data = start_data, 
            constructors_data = constructors_data, 
            drivers_data = drivers_data, 
            pred_round = cur_round,
            dest_file = f"{result_folder}/lasso_coeff.csv",
            eval_mode = True,
            output_feature_report= output_feature_report,
            fp2_adjust=True
        )
        if all_preds is None:
            all_preds = preds
        else:
            all_preds = pd.concat(
                [all_preds, preds], axis=0, ignore_index=True
            )
    
    print(Colors.YELLOW)
    print(all_preds)
    print(Colors.ENDC)

    all_preds = all_preds.dropna()
    
    r2_race = r2_score(all_preds['positionOrder_true'], all_preds['positionOrder_pred'])
    r2_quali = r2_score(all_preds['grid_true'], all_preds['grid_pred'])

    print("[R2-Cont.  RESULTS]: Race: {} | Quali: {}".format(r2_race, r2_quali))

    r2_race_b = r2_score(all_preds['positionOrder_true'], all_preds['fp_pred'])
    r2_quali_b = r2_score(all_preds['grid_pred'], all_preds['sp_pred'])

    print("[R2-Ranked RESULTS]: Race: {} | Quali: {}".format(r2_race_b, r2_quali_b))

    with open(f"{result_folder}/metrics.txt", 'w') as f:
        f.write("[R2-Cont.  RESULTS]: Race: {} | Quali: {}\n".format(r2_race, r2_quali))
        f.write("[R2-Ranked RESULTS]: Race: {} | Quali: {}".format(r2_race_b, r2_quali_b))


if __name__ == "__main__":
    print("[ERROR]: DO NOT RUN MODULE DIRECTLY")
    # main2()
