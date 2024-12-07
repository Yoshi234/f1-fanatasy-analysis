import pandas as pd

def std_pt_distrib(df):
    '''
    Takes as input a dataframe containing results for each driver for each race 
    in a specified range. The previous point values are taken for each race, and 
    the output value is given as the proportion of points belonging to a given 
    driver divided by the total prev points for all other records (may have trouble
    with dealing who have missing records for previous points)

    Args
    - df ----- dataframe containing the original data
    - save --- boolean indicator - whether or not to save the data 
    - file --- if save is true, please list a file name to save data to with folder
      path attached
    Return 
    - mod_df - return the modified data frame with point distributions encoded 
      as a proportion of total points held
    '''
    new_names = []
    if ('prev_driver_points' in df.keys()) and ('prev_construct_points' in df.keys()) and ('round' in df.keys()) and ('year' in df.keys()):
        for yr in df['year'].unique():
            for rd in df['round'].unique():
                tmp = df.loc[(df['year']==yr) & (df['round']==rd)]
                df.loc[(df['year']==yr) & (df['round']==rd), 
                       'prev_driver_points_prop'] = tmp['prev_driver_points']/tmp['prev_driver_points'].sum()
                df.loc[(df['year']==yr) & (df['round']==rd),
                       # divide by 2 to avoid double counting constructor points for each result record
                       'prev_construct_points_prop'] = tmp['prev_construct_points']/(tmp['prev_construct_points'].sum()/2)
        new_names = ['prev_construct_points_prop','prev_driver_points_prop']
    return df, new_names