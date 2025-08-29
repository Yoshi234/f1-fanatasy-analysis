import pandas as pd
import os
import time
from tqdm import tqdm
import itertools
import pickle
import math
import numpy as np
import sys

# local modules
from scrape_price_data import copilot_price_data_output


def score_func(x):
    '''
    Calculates the score for a given driver based on the average
    race simulation lap time relative to the fastest driver's
    average lap time. 

    Given as time/fastest_time in seconds

    NEW method: just return the ranking and associated score
    (1-20) for the testing results
    '''
    scaler = 100
    val = x - 100.0
    if val == 0:
        return scaler
    else:
        return max(100 - math.e**(val*5),0) 


def gen_combs():
    '''
    generate combinations from the number of values
    creates a list of frozen (immutable) sets to pass to a 
    pandas dataframe
    '''
    data = ['NOR', 'PIA', 'ALB', 'VER', 'LAW', 
            'OCO', 'SAI', 'HAD', 'TSU', 'RUS', 
            'ANT', 'DOO', 'HAM', 'HUL', 'BOR',
            'GAS', 'STR', 'ALO', 'BEA', 'LEC', 
            'COL']
    teams = ['ALP', 'AST', 'FER', 'HAA', 'KCK', 'MCL', 
             'MER', 'VRB', 'RED', 'WIL']
    r=5
    t=2
    driver_combinations = list(itertools.combinations(data, r))
    team_combinations = list(itertools.combinations(teams, t))
    
    # generate cartesian product of driver combs and team combs
    full_combs = []
    for i in driver_combinations:
        for j in team_combinations:
            full_combs.append(frozenset(i + j)) 
    
    # list of full combinations of teams and drivers
    return full_combs


def get_cost(team_comp, vals:pd.DataFrame):
    '''
    depreciated - use frozen set inclusion computation
    '''
    cost = 0
    score = 0
    for item in team_comp:
        # print("[DEBUG]: asset = {}".format(item))
        x = vals.loc[vals['asset']==item]
        if len(x) == 0:
            print("[INFO]: item = {}".format(item))
        cost += x['price'].item()
        score += x['points'].item()
    else:
        return cost, score 


def get_diff(set1, set2):
    return len(set1 - set2)


def check_in(set_val, item):
    return item in set_val


def get_scores(
    predictions_file="../results/saudi-arabia_predictions.csv", 
    values_file="../results/saudi-arabia/josh_fantasytools_assets.csv",
    score_field='fantasy_pts'
):
    '''
    Takes as input the predictions file with the pandas table of 
    outcome predictions for an upcoming weekend, and the pandas 
    table of scores, prices, and so forth
    '''
    con_map = {
        'red_bull': 'RED',
        'mclaren': 'MCL',
        'ferrari': 'FER', 
        'mercedes': 'MER', 
        'aston_martin': 'AST',
        'rb': 'RB', 
        'haas': 'HAA', 
        'alpine': 'ALP', 
        'sauber': 'KCK', 
        'williams': 'WIL'
    }
    preds = pd.read_csv(predictions_file)
    values = pd.read_csv(values_file)
    
    for driver in preds['Driver'].unique():
        score = preds.loc[preds['Driver']==driver, score_field].values[0]
        values.loc[values['asset']==driver, 'pred_score']= score
    for constructor in preds['Constructor'].unique():
        score = preds.loc[preds['Constructor']==constructor, score_field].sum()
        values.loc[values['asset']==con_map[constructor], 'pred_score']=score
    
    values.to_csv(values_file, index=False)


def team_analysis(
    values_table, 
    output, 
    weight=False,
    starting_team=None,
    surplus=0,
    max_dif=2,
    save_all=False
):
    '''
    new implementation for optimal team construction with pandas

    Args:
    - values_table --- containing all of the values for assets, their
    - output --------- file name for the output .csv file for the 
                       results of this analysis
    - weight --------- boolean indicator - if true, then uses the
                       weights information corresponding to the 
                       max driver to determine their additional
                       score contribution
    - starting_team -- The initial team composition. If this is None, 
                       the optimal team will be constructed from 
                       input prices and scores without further restriction.
                       If a composition is input, then teams will be 
                       evaluated based on (1) affordability is price of 
                       team 1 greater than or equal to that of the new 
                       team? (2) Are there no more than 2 asset class 
                       changes from team 1 to team 2?
    '''
    vals = pd.read_csv(values_table)
    s1 = time.time()
    combs = gen_combs() # return list of frozen sets
    s2 = time.time()
    print("[INFO]: generating team combinations = {} s".format(s2 - s1))
    full_df = pd.DataFrame({
        "team_comps": combs
    })
    full_df['price'] = 0.0
    full_df['score'] = 0.0
    full_df['max_score'] = 0.0

    if 'projected_gain' in vals.keys():
        full_df['projected_gain'] = 0.0
    if 'pred_score' in vals.keys():
        full_df['pred_score'] = 0.0
    
    # set weight for driver drs boost (average)
    if weight == True:
        full_df['weight'] = 0.0

    for idx, row in tqdm(vals.iterrows(), dynamic_ncols=True, 
                         total=vals.shape[0], leave=True):
        asset = row['asset']
        contains = full_df['team_comps'].apply(check_in, args=(asset,))
        try:
            full_df.loc[contains, "price"] += row['price']
            if 'projected_gain' in vals.keys():
                full_df.loc[contains, 'projected_gain'] += row['projected_gain']
            if 'pred_score' in vals.keys():
                full_df.loc[contains, 'pred_score'] += row['pred_score']
        except:
            print("[DEBUG]: row['price'] = \n{}".format(row['price']))
            print("[DEBUG]: full_df.loc[contains, 'price'] = \n{}".format(full_df.loc[contains,'price']))
            exit()
        full_df.loc[contains, "score"] += row['score']
        if row['type'] == 'driver':
            greatest = full_df.loc[contains, 'max_score'] <= row['score']

            # use the 'greatest' index to subindex itself
            full_df.loc[greatest[greatest].index, 'max_score'] = row['score']
            full_df.loc[greatest[greatest].index, 'weight'] = row['weight']

        # for drivers, apply a check to see if it is the greatest 
        # driver score achievable for a given group 
        # then, apply the multiplier associated with that driver
        # and sum to the score

    check_price = 100.0
    if starting_team is not None:
        start_cost = 0
        for idx, row in vals.iterrows():
            if row['asset'] in starting_team:
                start_cost += row['price']
        check_price = start_cost + surplus
        print("[INFO]: Max Team Budget = {}".format(check_price))
    
    # drop invalid team compositions
    full_df = full_df.loc[full_df['price'] <= check_price]
    full_df['score'] += full_df['max_score'] * full_df['weight']
    if 'pred_score' in vals.keys():
        full_df['ensemble_score'] = (full_df['score'] + full_df['pred_score'])/2

    if 'projected_gain' in full_df.keys():
        full_df['centered_pred'] = (full_df['pred_score'] - full_df['pred_score'].mean())/(full_df['pred_score'].std())
        full_df['centered_score'] = (full_df['score'] - full_df['score'].mean())/(full_df['score'].std())
        full_df['centered_gain'] = (full_df['projected_gain'] - full_df['projected_gain'].mean())/(full_df['projected_gain'].std())
        full_df['avg_value'] = 0.4 * full_df['centered_score'] + 0.6 * full_df['centered_gain']
        full_df['pred_avg_value'] = 0.4 * full_df['centered_pred'] + 0.6*full_df['centered_gain']

    full_df = full_df.sort_values(by='pred_avg_value', ascending=False)
    
    if starting_team is None:
        if save_all: full_df.to_csv(output, index=False)
    else:
        # check set difference
        full_df['team_diffs'] = full_df['team_comps'].apply(get_diff, args=(starting_team,))
        # subset teams with a difference <= 2
        full_df = full_df.loc[full_df['team_diffs'] <= max_dif]
        if save_all: full_df.to_csv(output, index=False)

        print(20*"-","Best Predicted Average Value",20*"-")
        best = full_df.iloc[0]
        for k in full_df.keys():
            print("{}:\t\t{}".format(k, best[k]))
        
        print()
        print(20*"-","Best Predicted Score Value",20*"-")
        x_df = full_df.sort_values(by="pred_score", ascending=False)
        best = x_df.iloc[0]
        for k in full_df.keys():
            print("{}:\t\t{}".format(k, best[k]))

        print()
        print(20*"-","Best Avg Fantasy Value",20*"-")
        z_df = full_df.sort_values(by='avg_value', ascending=False)
        best = z_df.iloc[0]
        for k in full_df.keys():
            print("{}:\t\t{}".format(k, best[k]))

        print()
        print(20*"-","Best Avg Fantasy Score",20*"-")
        y_df = full_df.sort_values(by='score', ascending=False)
        best = y_df.iloc[0]
        for k in full_df.keys():
            print("{}:\t\t{}".format(k, best[k]))


def update_price_values(
    values_file, 
    A_drivers,
    B_drivers,
    A_constructors,
    B_constructors,
    std_cols = ['driver', 'price/$', 'R13 Pts', 'R14 Pts', 'constructor']
):
    '''
    Updates price data based on the outputs from f1fantasytools

    Parameters:
    - values_file (path[str]): The path to the assets file used for the 
      driver pricing data and points values (team optimization)
    - A_drivers (pd.DataFrame): Dataframe with the A rank drivers pricing
      and points requirements data
    - B_drivers (Pd.DataFrame): Dataframe with the B rank drivers pricing
      and points requirements data
    - A_constructors (pd.DataFrame): Dataframe with the A rank constructors pricing
      and points requirements data
    - B_constructors (pd.DataFrame): Dataframe with the B rank constructors pricing
      and points requirements data

    Returns:
    - None: just pushes data to the output
    '''
    main_df = pd.read_csv(values_file)

    # iterate through all pricing dfs
    # for each of the drivers in that df
    asset_dfs = {
        'driver': [A_drivers, B_drivers], 
        'constructor': [A_constructors, B_constructors]
    }
    for a_type in asset_dfs.keys(): # for each asset dataframe (driver A, B, constructor A, B)
        for adf in asset_dfs[a_type]:            
            # print("=======")
            price_thresh_cols = adf.drop(std_cols, axis=1, errors='ignore').columns

            proj_vals = pd.Series(price_thresh_cols).str.strip("+").astype(float)
            min_proj = proj_vals.min()
            # print('minimum projected gain', min_proj)
            
            new_adf = pd.merge(
                adf, main_df[['asset', 'pred_score']], 
                left_on=a_type, right_on='asset'
            ).drop('asset',axis=1)
            # rename price column for merging
            

            # print(adf.columns)
            new_adf = new_adf.rename(columns={'price/$': 'price'})
            # print('ok')

            new_adf['pred_score'] = new_adf['pred_score'].astype(float)

            # print(new_adf.dtypes)
            
            x = new_adf.copy()
            x[price_thresh_cols] = x[price_thresh_cols].sub(x['pred_score'], axis=0)
            # set any differences less than 0 as 
              
            # print("[INFO]: Data After Differencing Predicted Score v. Thresholds")
            # print(x)

            for col in price_thresh_cols: # find greatest difference <= 0
                x.loc[x[col] > 0, col] = np.nan

            # print(x)

            x['projected_gain'] = x[price_thresh_cols].idxmax(axis=1).\
                                                       str.\
                                                       strip("+").\
                                                       astype(float)
            x['projected_gain'] = x['projected_gain'].fillna(min_proj)

            # if projected gain < 0 (-0.1 or something) but 
            # pred_score > -0.1 threshold (the max(price_thresh) < 0), 
            # then set as 0.0
            x.loc[
                (x['projected_gain'] < 0) & 
                (x[price_thresh_cols].max(axis=1) < 0),
            'projected_gain'] = 0.0

            # print('Price DataFrame')
            # print(x)
            # print()
            
            # use asset type (a_type) for the x dataframe, but not for the 
            # main dataframe - just use 'asset'
            price_gain_mapping = dict(zip(x[a_type], x['projected_gain']))
            cur_price_mapping = dict(zip(x[a_type], x['price']))

            # print('current price map',cur_price_mapping)

            main_df.loc[
                main_df['asset'].isin(price_gain_mapping.keys()),
                'projected_gain'
            ] = main_df['asset'].map(price_gain_mapping)

            main_df.loc[
                main_df['asset'].isin(cur_price_mapping.keys()),
                'price'
            ] = main_df['asset'].map(cur_price_mapping)

    
    # correct price mismatch with projected gain
    main_df.loc[
        (main_df['price']==4.5) &
        (main_df['projected_gain'] < 0),
    'projected_gain'] = 0

    # save final results
    main_df.to_csv(values_file, index=False)


if __name__ == "__main__":
    # please update this
    driversA, driversB, constructorsA, constructorsB = copilot_price_data_output()

    location = "zandfort_rf"
    results_folder = "../results"
    assets_file = "josh_fantasytools_assets.csv"
    team = 'team2'

    # set team gen parameters
    run_get_scores = True
    update_prices = True

    if not assets_file in os.listdir(f"{results_folder}/{location}"):
        dest = f"{results_folder}/{location}/{assets_file}"
        os.system(f"cp {results_folder}/assets_template.csv {dest}")

    if run_get_scores:
        # update scores from the predictions file
        get_scores(
            predictions_file=f"{results_folder}/{location}/predictions.csv", 
            values_file=f"{results_folder}/{location}/{assets_file}",
            score_field='fantasy_pts'
        )

    if update_prices:
        # update the predicted price values and actual driver prices
        update_price_values(
            values_file=f"{results_folder}/{location}/{assets_file}",
            A_drivers=driversA,
            B_drivers=driversB,
            A_constructors=constructorsA,
            B_constructors=constructorsB
        )

    teams_info = {
        'team1': {
            'cur_team': frozenset(['HUL', 'ALO', 'STR', 'PIA', 'BEA', 'MCL', 'KCK']),
            'surplus': 15.9,
            'max_dif': 3
        },
        'team2': {
            'cur_team': frozenset(['BOR', 'HAD', 'LAW', 'PIA', 'ALO', 'FER', 'MCL']),
            'surplus': 100, 
            'max_dif': 10
        }
    }

    # # run team analysis based on metrics
    selected_team = teams_info[team]
    team_analysis(
        values_table = f"{results_folder}/{location}/{assets_file}", 
        output = f"{results_folder}/{location}/josh_fantasytools_results1.csv", 
        weight=False,
        starting_team=selected_team['cur_team'],
        surplus=selected_team['surplus'],
        max_dif=selected_team['max_dif']
    )
