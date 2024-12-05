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
    train_test_split
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

def xgb_fit(train_features, test_window, info=True, smote=False):
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
        oversample = SMOTENC(categorical_features=cat_indices, random_state=0)
        x_train, y_train = oversample.fit_resample(x_train, y_train)
      except:
        oversample = SMOTENC(categorical_features=cat_indices, k_neighbors=y_train.sum()-1)
        x_train, y_train = oversample.fit_resample(x_train, y_train)

    x_test = test_window[x_train.keys()]
    y_test = test_window[y_train.name]

    # set xgboost classifier model
    xgb_mod = xgb.XGBRegressor(
        tree_method='hist',
        enable_categorical=True,
        max_depth=3,
        gamma=1
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

