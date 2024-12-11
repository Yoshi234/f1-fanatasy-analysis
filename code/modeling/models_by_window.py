import pandas as pd
from ISLP.models import (ModelSpec as MS, summarize)
import statsmodels.api as sm
from ISLP import confusion_table
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
try:
    from param_train import (
        logistic_fit, 
        xgb_fit
    )
    from selection import (
        get_data_in_window, 
        get_features, 
        get_encoded_data, 
        add_interaction,
        add_podiums
    )
    from mod_point_distrib import std_pt_distrib
except: # import as module
    from .param_train import (
        logistic_fit,
        xgb_fit
    )
    from .selection import (
        get_data_in_window, 
        get_features, 
        get_encoded_data, 
        add_interaction,
        add_podiums
    )
    from .mod_point_distrib import std_pt_distrib
import warnings
warnings.filterwarnings('ignore')

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

if __name__ == "__main__":
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