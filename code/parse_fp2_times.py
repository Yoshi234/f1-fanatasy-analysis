import pandas as pd
import math
from modeling.models_by_window import dq_scores, dr_scores

def get_ranks(input_file='', output_file=''):
    if input_file =='':
        print("[ERROR]: No rank input file name provided")
        exit()
    if output_file == '':
        print("[ERROR]: No rank output file name provided")
        exit()

    x = pd.read_csv(input_file)
    time_cols = x.columns[1:]
    missing_drivers = []
    for col in time_cols:
        try:
            x[col] = pd.to_timedelta("00:" + x[col]).dt.total_seconds()
        except:
            print('[ERROR]: {}'.format(col))
            missing_drivers.append(col)
            continue
    x[missing_drivers] = x[time_cols].mean().mean()
    y = x.describe()
    z = y.drop('lap', axis=1).loc['mean']
    z = z.sort_values(ascending=True)
    ranks = z.rank(method='min').astype(int)
    df = pd.DataFrame({
        "LapTime": z,
        "Rank": ranks
    })
    df = df.reset_index()
    print(df.keys())
    df['Driver'] = df['index']
    df.to_csv(output_file, index=False)

def get_new_pred_alt(row):
    ratio = (row['positionOrder']-row['Rank'])/10
    new_ratio = (1 + math.fabs(row['lap_seconds_norm'])) * ratio
    prod = new_ratio * 1.96
    if prod > 2.576: prod = 2.576
    elif prod < -2.576: prod = -2.576
    return row['positionOrder'] - prod*row['std_err_r']

def get_new_pred(row):
    ratio = (row['positionOrder'] - row['Rank'])/10
    if ratio * 1.96 < -2.576:
        prod = -2.576
    elif ratio * 1.96 > 2.576:
        prod = 2.576
    else:
        prod = 1.96 * ratio
    
    return row['positionOrder'] - prod*row['std_err_r']

def main(fp2_file='', preds_file='', out_folder=''):
    if fp2_file == '':
        print("[ERROR]: No input file")
        return
    if preds_file == '':
        print("[ERROR]: No predictions file")
        return
    if out_folder == '':
        print("[ERROR]: No output folder listed")

    x = pd.read_csv(fp2_file)
    y = pd.read_csv(preds_file)
    z = pd.merge(y,x,on='Driver')
    z['LapTime'] = pd.to_timedelta(z['LapTime']) # conver laptime to timedelta
    z['lap_seconds'] = z['LapTime'].dt.total_seconds()

    # center the lap time in seconds (normalize to z-scores)
    z['lap_seconds_norm'] = (z['lap_seconds'] - z['lap_seconds'].mean())/z['lap_seconds'].std()
    z['adj_pred_order1'] = z.apply(get_new_pred, axis=1)
    z['adj_pred_order2'] = z.apply(get_new_pred_alt, axis=1)
    z['new_fp_1'] = z['adj_pred_order1'].rank(method='min', ascending=True).astype(int)
    z['new_fp_2'] = z['adj_pred_order2'].rank(method='min', ascending=True).astype(int)
    z = z.sort_values(by='adj_pred_order2')
    z[['Driver', 'new_fp_1', 'new_fp_2', 'positionOrder', 'adj_pred_order1', 'adj_pred_order2']].to_csv(
            f"{out_folder}/new_predictions.csv", index=False
    )

def recalc_fantasy_score(old_preds, new_preds):
    x = pd.read_csv(old_preds)
    y = pd.read_csv(new_preds)

    if 'new_fp_1' in x.keys():
        x = x.drop(['new_fp_1', 'new_fp_2'], axis=1)

    z = pd.merge(x, y[['Driver', 'new_fp_1', 'new_fp_2']], left_on='Driver', right_on='Driver', how='left')
    
    z['position_change_new'] = z['sp'] - z['new_fp_1']
    # print(z)

    z['fant_pts_new'] = z['position_change_new']
    # print(z)
    for idx, pred in z.iterrows():
        # print('\t[INFO]: {} --> {} + {} + {}'.format(
        #     pred['Driver'], dq_scores[pred['sp']], dr_scores[pred['new_fp_1']],
        #     pred['position_change_new'])
        # )
        z.loc[idx, 'fant_pts_new'] += dq_scores[pred['sp']]
        z.loc[idx, 'fant_pts_new'] += dr_scores[pred['new_fp_1']]
        # print('\t\tFinal Score = {}'.format(z.loc[idx, 'fant_pts_new']))

    z.to_csv(old_preds, index=False)

if __name__ == "__main__":
    get_ranks(
            input_file='../results/hungary/hungary_fp2.csv', # raw pace file
            output_file='../results/hungary/fp2_ranks.csv' # output ranks file for fp2
    )
    main(
            fp2_file='../results/hungary/fp2_ranks.csv', # fp2 rank file
            preds_file='../results/hungary/predictions.csv', # predictions file
            out_folder='../results/hungary' # folder to place new predictions in
    )
    recalc_fantasy_score(
            old_preds='../results/hungary/predictions.csv',
            new_preds='../results/hungary/new_predictions.csv'
    )
