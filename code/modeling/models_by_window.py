import pandas as pd
from selection import get_data_in_window, get_features, get_encoded_data, add_interaction
from ISLP.models import (ModelSpec as MS, summarize)
import statsmodels.api as sm
from ISLP import confusion_table
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

def models_by_window_2(
    start_yr, 
    start_r, 
    n, 
    data='../../data/clean_model_data.feather', 
    max_year=2024, 
    debug=False
):
    """
    Summarizes accuracy and f1 score for each window across all races in span
        
    Args:
    - start_yr - the first year of the span on which to test
    - start_r -- the first round to be tested on in start_yr
    - n -------- the biggest window size to test; windows from size 1 to n will be tested
    - data ----- data file to choose

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
    if data.split(".")[-1] == "feather":
        all_data = pd.read_feather(data)
    elif data.split(".")[-1] == 'csv':
        all_data = pd.read_csv(data)

    results_df = pd.DataFrame(columns=['n', 'Accuracy', 'F1 Score'])

    # For each model of window size 1 to n
    for i in tqdm(range(1,n+1), ncols=100, total=n, 
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
                train_window = get_data_in_window(k = i+1, yr = year, r_val = round, track_dat=all_data)
                train_window['Podium Finish'] = ['Yes' if position <= 3 else 'No' for position in train_window['positionOrder']]
                train_window = train_window.reset_index().drop(['index'],axis=1)
                
                drivers_train, constructors_train, _, df_train = get_encoded_data(train_window)

                train_window = add_interaction(df_train, vars=['corner_spd_min'], drivers=drivers_train)

                test_window = train_window[(train_window['round'] == round) & (train_window['year'] == year)]

                train_window = train_window.drop(train_window[(train_window['round'] == round) & (train_window['year'] == year)].index)

                #print(train_window['round'].unique()) # another test

                # select features using the data window
                if debug: 
                    print("[DEBUG]: train_window.keys() = \n{}".format(train_window.keys()))
                
                train_features = train_window[['Podium Finish', 'constructorId_9']]

                features = train_features.columns.drop(['Podium Finish'])

                design = MS(features)
                X = design.fit_transform(train_features)
                y = train_features['Podium Finish'] == 'Yes'
                lr = sm.GLM(y,
                            X,
                            family = sm.families.Binomial())
                lr_results = lr.fit()

                # get predicted probabilities for current race
                test = MS(features).fit_transform(test_window)
                probabilities = lr_results.predict(test)
                
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