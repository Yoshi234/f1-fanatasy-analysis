import pandas as pd
import os
import time
from tqdm import tqdm
import itertools
import pickle

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
    # print("[INFO]: number of combinations = {}".format(len(driver_combinations)))
    # print("[INFO]: number of team combinations = {}".format(len(team_combinations)))

    # print(team_combinations)
    
    # generate cartesian product of driver combs and team combs
    full_combs = []
    for i in driver_combinations:
        for j in team_combinations:
            full_combs.append(frozenset(i + j)) 
    # print("[INFO]: number of full combinations = {}".format(len(full_combs)))
    
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

def check_in(set_val, item):
    return item in set_val

def team_analysis(values_table, output):
    '''
    new implementation for optimal team construction with pandas

    Args:
    - values_table --- containing all of the values for assets, their
    - output --------- file name for the output .csv file for the 
                       results of this analysis
    '''
    # cache the results of running this function
    s1 = time.time()
    if "team_comps.pkl" not in os.listdir():
        vals = gen_combs()
        with open("team_comps.pkl", 'wb') as file:
            pickle.dump(vals, file)
    else:
        with open("team_comps.pkl", 'rb') as file:
            vals = pickle.load(file)
    s2 = time.time()
    print("[INFO]: pickling = {} s.".format(s2 - s1))

    vals = pd.read_csv(values_table)
    combs = gen_combs() # return list of frozen sets
    full_df = pd.DataFrame({
        "team_comps": combs
    })
    full_df['price'] = 0.0
    full_df['score'] = 0.0
    # get first 10 for testing
    for idx, row in tqdm(vals.iterrows(), dynamic_ncols=True, 
                         total=vals.shape[0], leave=True):
        # print("[DEBUG]: row = {}".format(row))
        asset = row['asset']
        contains = full_df['team_comps'].apply(check_in, args=(asset,))
        # print("[DEBUG]: contains = {}".format(contains))
        # print("[DEBUG]: test_df.keys = {}".format(test_df.keys()))
        full_df.loc[contains, "price"] += row['price']
        full_df.loc[contains, "score"] += row['score']
    
    # drop invalid team compositions
    full_df = full_df.loc[full_df['price'] <= 100.0]
    full_df = full_df.sort_values(by='score', ascending=False)
    full_df.to_csv(output, index=False)

if __name__ == "__main__":
   team_analysis("nicks_assets.csv", "nicks_results.csv") 
