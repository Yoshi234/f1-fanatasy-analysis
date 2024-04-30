import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTENC
import pickle

def get_smot_data(X, y, cat_indices):
    # get the normal data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # get the oversampled training data
    oversample = SMOTENC(categorical_features=cat_indices, random_state=0)
    x_train_smote, y_train_smote = oversample.fit_resample(x_train, y_train)
    return x_train_smote, y_train_smote, x_test, y_test, x_train, y_train

def get_features(data:pd.DataFrame, features=[], select=False):
    if len(features) == 0: features = ['driverId', 'constructorId']
    encoder = OneHotEncoder()
    one_hot = encoder.fit_transform(data[features])
    feature_names = encoder.get_feature_names_out()
    df_features = pd.DataFrame(one_hot.toarray(), columns=feature_names)
    
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

def add_interaction(data, vars=[], drivers=[], constructors=[]):
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
    for i in range(len(drivers)):
        # skip max verstappen and red bull
        # if drivers[i] == "driverId_830": continue
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
            data2 = pd.concat([data2, df], axis=1)
            # # add interaction to the dataframe
            # data[v_string] = interact
            # print(v_string)
    return data2

# save model to file
def save_model(location, model, mod_name, input_features):
    filename = "{}/{}.pkl".format(location, mod_name)
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    with open("{}/{}_features.pkl".format(location, mod_name), "wb") as f2:
        pickle.dump(input_features, f2)

def train_eval(name:str, method:str, 
               features:list, 
               x_test, y_test,
               x_train_smote, y_train_smote,
               x_train, y_train):
    '''
    Assumes global access to the training and test data
    '''
    x_trn = None
    y_trn = None
    x_tst = x_test[features]
    y_tst = y_test

    if method=="smote":
      x_trn = x_train_smote[features]
      y_trn = y_train_smote
    elif method == "imbalanced":
      x_trn = x_train
      y_trn = y_train

    model = LogisticRegression(penalty='l2')
    model.fit(x_trn, y_trn)
    pred = model.predict(x_tst)
    f1 = f1_score(y_tst, pred)
    acc = accuracy_score(y_tst, pred)
    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_tst, pred)
    return f1, acc, precision, recall, fbeta_score, model

def main():
    # restrict data to 2022 and 2023
    data = pd.read_feather("../data/clean_model_data.feather")

    # need to reset the index so category fitting works properly
    data2 = data.loc[data['year'] >= 2022]
    data2 = data2.reset_index()

    '''
    we know that constructor will be interacted with everything, 
    so just feed the same list every time. Leave drivers empty
    '''
    interactions = [
        ['aero_reg'],
        ['years_since_major_cycle'],
        ['years_since_major_cycle','round'],
        ['corner_spd_min','aero_reg'],
        ['corner_spd_max','engine_reg'],
        ['corner_spd_max'],
        ['corner_spd_min'],
        ['round'],
        ['round', 'years_since_major_cycle'],
        ['windspeed'],
        ['strt_len_median'],
        ['strt_len_max'],
        ['avg_track_spd'],
        ['max_track_spd'],
        ['num_fast_corners'],
        ['num_slow_corners'],
        ['num_corners'],
        ['circuit_len'],
    ]
    driver_vars, construct_vars, _, encoded_dat = get_encoded_data(data2)

    for interaction in interactions:
        encoded_dat = add_interaction(encoded_dat, constructors=construct_vars, vars=interaction)
    for interaction in interactions:
        encoded_dat = add_interaction(encoded_dat, drivers=driver_vars, constructors=[], vars=interaction)

    encoded_dat = encoded_dat.drop(['event_name', 'preciptype',
                                'fastestLap', 'rank',
                                'fastestLapTime', 'fastestLapSpeed',
                                'ref_name', 'q1', 'q2', 'q3'],
                                axis=1)

    # drop null values from miscellaneous columns (missing race tracks)
    encoded_dat = encoded_dat.dropna()

    # create the actual target variables of interest:
    encoded_dat.loc[(encoded_dat['positionOrder'] <= 3), 'top3r'] = 1
    encoded_dat.loc[(encoded_dat['positionOrder'] > 3), 'top3r'] = 0

    encoded_dat.loc[(encoded_dat['quali_position'] <= 3), 'top3q'] = 1
    encoded_dat.loc[(encoded_dat['quali_position'] > 3), 'top3q'] = 0

    # get the output variables
    y_qual = encoded_dat['quali_position']
    y_race = encoded_dat['positionOrder']
    y_top3_finish = encoded_dat['top3r']
    y_top3_quali = encoded_dat['top3q']

    X = encoded_dat.drop(['quali_position','points','top3r','top3q', 'positionOrder',
                        'statusId', 'grid', 'driverId', 'constructorId',
                        'laps', 'resultId', 'regulation_id', 'year','raceId'], axis=1)
    
    for col in X.columns: print(col)
    
    driv_cons_cycles = []
    for column in X.columns:
        if not (("constructorId" in column) or ("driverId" in column) or ('cycle' in column)):
            continue
        if len(column.split("-")) == 1:
            driv_cons_cycles.append(column)

    cat_indices = []
    for col in driv_cons_cycles:
        cat_indices.append(X.columns.get_loc(col))

    # # DEBUG Statements
    # print(X)
    # print(cat_indices)

    # pick the feature sets
    oversample = SMOTENC(categorical_features=cat_indices, random_state=0)
    X2, y_top3_finish2 = oversample.fit_resample(X, y_top3_finish)

    mut_info_score = mutual_info_classif(X2, y_top3_finish2)
    f_score, f_p_value = f_classif(X2, y_top3_finish2)

    scores = pd.DataFrame(
        {"Feature": X.columns,
         "F_score": f_score,
         "P_value": f_p_value,
         "Mutual Information": mut_info_score}
    )

    f_score2, f_p_value2 = f_classif(X, y_top3_finish)
    mut_info2 = mutual_info_classif(X, y_top3_finish)

    imbalanced_scores = pd.DataFrame({
        "Feature": X.columns,
        "F_score": f_score2,
        "P_value": f_p_value2,
        "mut_info": mut_info2
    })

    # get the training data
    x_train_smote, y_train_smote, x_test, y_test, x_train, y_train = get_smot_data(X, 
                                                                        y_top3_finish, 
                                                                        cat_indices)
    
    # set models and features to explore
    smot_mut_scores = scores.sort_values(by='Mutual Information', ascending=False)
    smot_mut_scores = smot_mut_scores.reset_index()

    smot_f_scores = scores.sort_values(by="F_score", ascending=False)
    smot_f_scores = smot_f_scores.reset_index()

    norm_mut_scores = imbalanced_scores.sort_values(by='mut_info', ascending=False)
    norm_mut_scores = norm_mut_scores.reset_index()

    norm_f_scores = imbalanced_scores.sort_values(by='F_score', ascending=False)
    norm_f_scores = norm_f_scores.reset_index()

    models = {
        "smot_mut_30_smot":["smote", smot_mut_scores.loc[:30, 'Feature']],
        "smot_mut_20_smot":["smote", smot_mut_scores.loc[:20, 'Feature']],
        "smot_mut_10_smot":["smote", smot_mut_scores.loc[:10, 'Feature']],
        "smot_mut_30_norm":["smote", norm_mut_scores.loc[:30, 'Feature']],
        "smot_mut_20_norm":["smote", norm_mut_scores.loc[:20, 'Feature']],
        "smot_mut_10_norm":["smote", norm_mut_scores.loc[:10, 'Feature']],
        "smot_f_30_smot": ["smote", smot_f_scores.loc[:30, 'Feature']],
        "smot_f_20_smot": ["smote", smot_f_scores.loc[:20, 'Feature']],
        "smot_f_10_smot": ["smote", smot_f_scores.loc[:10, 'Feature']],
        "smot_f_30_norm": ["smote", norm_f_scores.loc[:30, 'Feature']],
        "smot_f_20_norm": ["smote", norm_f_scores.loc[:20, 'Feature']],
        "smot_f_10_norm": ["smote", norm_f_scores.loc[:10, 'Feature']]
    }

    # optimal values to save at the end
    opt_score = 0
    opt_features = None
    opt_model = None
    opt_name = ""

    # initialize a dictionary to store the results associate with each model
    model_results = {
    "model name":[],
    "accuracy score": [],
    "f1 score": [],
    "precision": [],
    "recall": [],
    "F beta score": []
    }

    # these need to be cross-validated (k-folds)
    # perform smote k times!!!
    for mod in models:
        f1, acc, precision, recall, fbeta_score, model = train_eval(
                                                            mod, 
                                                            models[mod][0], 
                                                            models[mod][1],
                                                            x_test, y_test,
                                                            x_train_smote, y_train_smote, 
                                                            x_train, y_train)
        model_results['model name'].append(mod)
        model_results['accuracy score'].append(acc)
        model_results['f1 score'].append(f1)
        model_results['precision'].append(precision)
        model_results['recall'].append(recall)
        model_results['F beta score'].append(fbeta_score)

        if f1 > opt_score: 
            opt_score = f1
            opt_features = models[mod][1]
            opt_model = model
            opt_name = mod

    # save the results
    results = pd.DataFrame(model_results)
    results.to_csv("2022-2023_results.csv", index=False)

    # get the best model coefficients for saving
    coefficients = opt_model.coef_
    print(coefficients)

    # get the probabilities and coefficients and set 
    # them as a dataframe
    probs = []

    for coefficient in coefficients[0]:
        if coefficient < 0: 
            val = -(1 - np.exp(coefficient))
        elif coefficient > 0:
            val = np.exp(coefficient) - 1 
        probs.append(val)
    
    # print(len(opt_features), len(coefficients[0]), len(probs))

    features = pd.DataFrame({
        "features": opt_features,
        "coefficients": coefficients[0],
        "odds increase / decrease": probs
    })
    features.to_csv("2023-2023_opt_model_features.csv", index=False)

    # save the best model to pretrained
    # save_model("pretrained", opt_model, opt_name, opt_features)

if __name__ == "__main__":
    main()