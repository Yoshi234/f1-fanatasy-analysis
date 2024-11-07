import pandas as pd

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
                        (track_dat['round'].isin({i for i in range(r_val+1)}))]
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