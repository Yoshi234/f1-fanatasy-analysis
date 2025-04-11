import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import pickle

def Josh():
    pass

# git pull
# make changes
# git add .
# git commit -m "message"
# git push

def get_data_in_window(k, yr, r_val, track_dat=None, info=False):  
    '''
    Returns the k records up to and including round 'r_val' from 
    season 'yr' - this will look back to previous seasons for data
    if possible to fill all k records. However, if there is no 
    previous season in the data, this will be ignored

    Args:
    - k --- the number of rounds to include in the training window
    - yr --- the year / season the race to predict on comes from
    - r_val --- the round before the round to predict on 
    (r_val = predict_round - 1)

    Returns:
    - x --- a dataframe with the selected window of rounds for training
    data
    '''
    if track_dat is None: 
        track_dat = pd.read_feather("../../data/clean_model_data.feather")
    if r_val >= k:
        xa = track_dat.loc[(track_dat['year']==yr) & 
                        (track_dat['round'].isin({i for i in range(r_val-k+1,r_val+1)}))]
        x=xa
    elif r_val < k:
        xa = track_dat.loc[(track_dat['year']==yr) & 
                        (track_dat['round'].isin({i for i in range(r_val+1)}))]
        try:
            n_rnds = track_dat.loc[track_dat['year']==yr-1]['round'].max()
            dif = k - r_val
            xb = track_dat.loc[(track_dat['year']==yr-1) & 
                            (track_dat['round'].isin({i for i in range(n_rnds-dif+1, n_rnds+1)}))]
        except:
            xb = None
        if info: print(f"[INFO]: xb.shape = {xb['round'].nunique()}")
        x = pd.concat([xa,xb],axis=0)
    
    if info: print(f"[INFO]: xa.shape = {x['round'].nunique()}")
    return x

def get_features(data:pd.DataFrame, features=[], select=False, debug=False):
    if len(features) == 0: features = ['driverId', 'constructorId']
    encoder = OneHotEncoder()
    one_hot = encoder.fit_transform(data[features])
    # if debug: print("[DEBUG]: one_hot data = {}".format(one_hot))
    feature_names = encoder.get_feature_names_out()
    df_features = pd.DataFrame(one_hot.toarray(), columns=feature_names)
    
    if debug: print(df_features)
    
    if select==True:
        df_study = pd.concat([data[['positionOrder','quali_position']], df_features], axis=1)
    else:
        df_study = pd.concat([data, df_features], axis=1)

    return df_study

def get_encoded_data(dat:pd.DataFrame):
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

def add_interaction(
    data, vars=[], drivers=[], constructors=[],
    ret_term_names=False
):
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
    
    interaction_features = []
    for i in range(len(drivers)):
        # skip max verstappen and red bull
        # if drivers[i] == "driverId_830": continue
        print("\t[INFO]: drivers[i] = {}".format(drivers[i]))
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
            interaction_features.append(v_string)
            data2 = pd.concat([data2, df], axis=1)
            # # add interaction to the dataframe
            # data[v_string] = interact
            # print(v_string)
    if ret_term_names:
        return data2, interaction_features
    else:
        return data2

def add_podiums(n, data = None, year = 2021, round = 12):
    '''
    Adds the number of podiums over the last n races as a feature 
    for use in predicting podium finish outcomes
    '''
    train_df = get_data_in_window(n, year, round, track_dat=data)
    # test_df = get_data_in_window(1, year, round, track_dat=data)
    
    d_ids_train = train_df['driverId'].unique()
    # d_ids_test = test_df['driverId']

    if round - n - 10 <= 0:
        n_l_ps = round - n - 1
    else:
        n_l_ps = 10

    for j in range(1, n_l_ps + 1):
        var_string = 'Last_'+str(j)+'_Podiums'
        train_df[var_string] = [0 for i in train_df['driverId']]
        # test_df[var_string] = [0 for i in d_ids_test]
    
    for r in train_df['round'].unique():
        last_10 = get_data_in_window(n_l_ps, year, r-1)
        
        #print(np.sort(last_10['round'].unique())[::-1])
        
        for d in d_ids_train:
            d_df = last_10.loc[last_10['driverId'] == d]
            podiums = 0
            num_prev_races = 0
            for r2 in np.sort(last_10['round'].unique())[::-1]:
                p = d_df.loc[d_df['round'] == r2, 'positionOrder'].item()
                #print(type(p))
                
                if p <= 3:
                    podiums += 1
                num_prev_races += 1
                var_string_2 = 'Last_'+str(num_prev_races)+'_Podiums'
                
                #print(podiums)
                
                train_df.loc[(train_df['driverId'] == d) & (train_df['round'] == r), var_string_2] = podiums

    # for r in test_df['round'].unique():
    #     last_10 = get_data_in_window(n_l_ps, year, r-1)

    #     for d in d_ids_test:
    #         d_df = last_10.loc[last_10['driverId'] == d]
    #         podiums = 0
    #         num_prev_races = 0
    #         for r2 in np.sort(last_10['round'].unique())[::-1]:
                
    #             p = d_df.loc[d_df['round'] == r2, 'positionOrder']
    #             #print(type(p))
    #             p = p.item()
                
    #             if p <= 3:
    #                 podiums += 1
    #             num_prev_races += 1
    #             var_string_2 = 'Last_'+str(num_prev_races)+'_Podiums'
                
    #             #print(podiums)
                
    #             test_df.loc[(test_df['driverId'] == d) & (test_df['round'] == r), var_string_2] = podiums

    return train_df, ['Last_'+str(i)+'_Podiums' for i in range(1, n_l_ps + 1)]

if __name__ == "__main__":
    print(add_interaction)
    print(get_features)
    print(get_encoded_data)
    print(get_data_in_window)