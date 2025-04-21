import pandas as pd
import os
import time
from tqdm import tqdm
import itertools
import pickle
import math

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
            'GAS', 'STR', 'ALO', 'BEA', 'LEC']
    teams = ['ALP', 'AST', 'FER', 'HAA', 'KCK', 'MCL', 
             'MER', 'RB', 'RED', 'WIL']
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
    values_file="../results/saudi-arabia/josh_fantasytools_assets.csv"
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
        score = preds.loc[preds['Driver']==driver, 'fantasy_pts'].values[0]
        values.loc[values['asset']==driver, 'pred_score']= score
    for constructor in preds['Constructor'].unique():
        score = preds.loc[preds['Constructor']==constructor, 'fantasy_pts'].sum()
        values.loc[values['asset']==con_map[constructor], 'pred_score']=score
    
    values.to_csv(values_file, index=False)

def team_analysis(
        values_table, 
        output, 
        weight=False,
        starting_team=None,
        surplus=0,
        max_dif=2
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
        full_df['avg_value'] = 0.7 * full_df['centered_score'] + 0.3 * full_df['centered_gain']
        full_df['pred_avg_value'] = 0.7 * full_df['centered_pred'] + 0.3*full_df['centered_gain']

    full_df = full_df.sort_values(by='pred_avg_value', ascending=False)
    
    if starting_team is None:
        full_df.to_csv(output, index=False)
    else:
        # check set difference
        full_df['team_diffs'] = full_df['team_comps'].apply(get_diff, args=(starting_team,))
        # subset teams with a difference <= 2
        full_df = full_df.loc[full_df['team_diffs'] <= max_dif]
        full_df.to_csv(output, index=False)

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


if __name__ == "__main__":
    cur_team = frozenset(['PIA', 'BEA', 'DOO', 'OCO', 'HAD', 'MCL', 'HAA'])
    get_scores(
        predictions_file="../results/miami/predictions.csv", 
        values_file="../results/miami/josh_fantasytools_assets.csv"
    )
    team_analysis(
        values_table = "../results/miami/josh_fantasytools_assets.csv", 
        output = "../results/miami/josh_fantasytools_results1.csv", 
        weight=False,
        starting_team=cur_team,
        surplus=9.5,
        max_dif=3
    ) 
