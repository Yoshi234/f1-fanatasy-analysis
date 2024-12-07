### Custom functionalities ###
# try:
#     from selection import (
#         get_data_in_window,
#         get_features,
#         get_encoded_data, 
#         add_interaction
#     )
#     from models_by_window import models_by_window_2
# except:
#     from .selection import (
#         get_data_in_window,
#         get_features,
#         get_encoded_data, 
#         add_interaction
#     )
#     from models_by_window import models_by_window_2
### base packages ###
import statsmodels.api as sm
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from ISLP.models import (
    ModelSpec as MS, 
    summarize
)
from ISLP import confusion_table
from sklearn.model_selection import (
    train_test_split,
    KFold
)
from sklearn.metrics import(
    accuracy_score,
    f1_score
)
from imblearn.over_sampling import (
    SMOTENC, 
    SMOTE
)

def get_train_metrics(n_outs):
    '''
    fetch training errors for evaluation
    '''
    pass

def feature_select(model_type, x, y, splits=3, debug=True):
    '''
    perform a minimal feature selection process over data
    with smote-nc applied
    '''
    # if debug: 
    #     print('[DEBUG]: y = {}'.format(y))
    #     exit()

    best_features = [] # set empty list, append to after each iteration
    kf = KFold(n_splits=splits)
    features = x.keys().copy()

    iter_perf = None

    prev_best_key = None
    prev_perf = None # use F1 score as perf metric

    # print("--- NEW RACE ---")
    while True:
        # print("[INFO]: iter = {}".format(len(best_features)))
        # print("[INFO]: best_features = {}".format(best_features))
        # use train performance to measure improvement of fit
        for f in features: # remove a features one-by-one
            f_f1 = 0
            tmp_fs = best_features.copy()
            tmp_fs.append(f) # add feature f to the best features list
            x1 = x[tmp_fs] # subset the x data
            # for i, (train_idx, test_idx) in enumerate(kf.split(x)):
            # x1 = x.iloc[train_idx] # train data
            # y1 = y.iloc[train_idx]

            # x2 = x.iloc[test_idx] # test data 
            # y2 = y.iloc[test_idx]

            # set cat indices of the data
            cat_indices = []
            # print('[DEBUG]: x1.keys = {}'.format(x1.keys()))
            for col in x1.keys(): 
                if x1[col].dtype == 'category' or x1[col].dtype == 'int64': 
                    cat_indices.append(x1.columns.get_loc(col))
            
            # apply smote encoding to the data
            if len(cat_indices) > 0:
                oversample = SMOTENC(
                                categorical_features=cat_indices, 
                                k_neighbors=y.sum()-1, 
                                random_state=0
                              )
                x2, y1 = oversample.fit_resample(x1, y)
            else:
                oversample = SMOTE(
                                k_neighbors = y.sum()-1,
                                random_state = 0
                              )
                x2, y1 = oversample.fit_resample(x1, y)
            # fit the model
            xgb_mod = xgb.XGBRegressor(
                tree_method='hist',
                enable_categorical=True,
                # max_depth=3,
                # gamma=1
            )
            xgb_mod.fit(x2, 
                        y1)
            
            # predict over test x original data
            probs = xgb_mod.predict(x1)
            n = y.sum()
            sorted_indices = np.argsort(probs)
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(len(probs))
            # preds = np.empty_like(probs)
            # top_indices = np.argsort(preds)[-n:]
            # preds[top_indices]=1
            preds = np.empty_like(probs)
            preds[ranks >= y.shape[0] - y.sum()] = 1
            preds[ranks < y.shape[0] - y.sum()] = 0

            # if debug: 
            #     print("[DEBUG]: features = {}".format(tmp_fs))
            #     print("[DEBUG]: y = {}".format(y))
            #     print("[DEBUG]: probs = {}".format(probs))
            #     print("[DEBUG]: preds = {}".format(preds))
            #     print("[DEBUG]: f1_score = {}".format(f1_score(preds, y)))
            #     # exit() 

            # n_outs = y.shape[0]
            # labels = np.array(['Yes'] * n_outs) # match num yesses to num outputs
            # no_indices = np.argpartition(probs, n_outs-y.sum())[:n_outs-y.sum()] # where n_outs = 20, index is 17
            # labels[no_indices] = 'No'
            test_f1 = f1_score(preds, y)

            if prev_perf is None: 
                prev_perf = test_f1
                prev_best_key = f
            elif f_f1/splits > prev_perf:
                prev_perf = test_f1
                prev_best_key = f
        
        if iter_perf is None: 
            # add prev_best_key to best_features
            best_features.append(prev_best_key)
            # print("[DEBUG]: \tbest_features = {}".format(best_features))
            features = features.drop(prev_best_key) # remove added key
            iter_perf = prev_perf
            # reset iterator values
            prev_perf = None
            prev_best_key = None
        elif prev_perf > iter_perf:
            best_features.append(prev_best_key)
            # print("[DEBUG]: \tbest_features = {}".format(best_features))
            features = features.drop(prev_best_key)
            iter_perf = prev_perf
            # reset iterator values
            prev_perf = None
            prev_best_key = None
        else:
            break # break if perf does not improve across iterations
    return best_features, iter_perf          
                

def xgb_fit(train_features, test_window, info=True, smote=False, f_select=False):
    '''
    fits an xgboost model over the input training data and returns
    predicted output probabilities over the test_window.
    NOTE: na values should be checked before passing data for fitting
    here!

    Args:
    - train_features --- training data with selected features and the 
      response (Podium Finish)
    - test_window ------ full data frame with features not yet selected
      (to be same as training) and response still included for the 
      test data
    Returns:
    - probabilities ---- the probability of a podium finish for each 
      entry in test_window
    '''
    features = train_features.drop(['Podium Finish'], axis=1)
    x_train = features

    # extract response value and convert to an integer
    y_train = train_features['Podium Finish'] == 'Yes'
    # print('[DEBUG]: y_train vals = ', y_train)
    # print('[DEBUG]: y_train.name =',y_train.name)
    y_train = y_train.astype(int)

    if f_select:
        f_vals, best_train_perf = feature_select('xgb', x_train, y_train)
        x_train = x_train[f_vals] # subset the x_train data from best features

    # apply forward feature selection
    # select_features = feature_select('xgb', x_train, y_train)    

    if info: print("[DEBUG]: y_train.sum() = {}".format(y_train.sum()))

    # apply SMOTE to boost performance?
    cat_indices = []
    for col in x_train.keys(): 
        if x_train[col].dtype == 'category' or x_train[col].dtype == 'int64': 
            cat_indices.append(x_train.columns.get_loc(col))
        elif info: 
            print("[INFO]: var = {} type = {}".format(col, x_train[col].dtype))

    if info: print("[DEBUG]: cat_indices = {}".format(cat_indices))

    if info: print("[DEBUG]: shape x = {} shape y = {}".format(x_train.shape, y_train.shape))
    if smote:
      try: 
        oversample = SMOTENC(categorical_features=cat_indices, k_neighbors=y_train.sum()-1, random_state=0)
        x_train, y_train = oversample.fit_resample(x_train, y_train)
      except:
        oversample = SMOTE(k_neighbors=y_train.sum()-1)
        x_train, y_train = oversample.fit_resample(x_train, y_train)

    x_test = test_window[x_train.keys()]
    y_test = test_window[y_train.name]

    # set xgboost classifier model
    xgb_mod = xgb.XGBRegressor(
        tree_method='hist',
        enable_categorical=True,
        # max_depth=3,
        # gamma=1
    )
    xgb_mod.fit(x_train, 
                y_train)
    
    # evals_result = xgb_mod.evals_result()
    # if info: 
    #     for e_name, e_mtrs in evals_result.items():
    #       print("[INFO] - {}".format(e_name))
    #       for e_mtr_name, e_mtr_vals in e_mtrs.items():
    #           print("\t[INFO] - {}".format(e_mtr_name))
    #           print("\t\t[INFO] - {}".format(e_mtr_vals))

    probs = xgb_mod.predict(x_test)

    return probs

def logistic_fit(train_features, test_window):
    '''
    performs a logistic regression fit on the data and 
    outputs probabilities over the test window

    Args:
    - train_features --- dataset containing training data with 
      selected training features to use. this includes the response
      feature by default 
    - test_window ------ full dataset including non-training features
      for the test data. Will automatically be dealt with by the helper
      function.
    Returns:
    - probabilities ---- array of predicted podium finish probabilities
      for each driver in the race
    '''
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

    return probabilities

