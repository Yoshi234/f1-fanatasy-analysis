import numpy as np
import pandas as pd

def summarize_na(dat):
  '''
  take as input a datframe and summarize missingness for all 
  columns in said dataframe
  '''
  count_na = dat[[col for col in dat.columns]].isna().sum()
  count_na = count_na.to_frame(name="na count")

  print(count_na.shape)
  if count_na.sum().item() == 0:
    print("[INFO] NO MISSING VALUES")
    return None

  # get na counts as a proportion of total records
  count_na["prop na"] = count_na["na count"]/dat.shape[0]

  # create a separate column for the variables (columns)
  count_na = count_na.reset_index()

  no_na = count_na.loc[count_na["na count"] == 0, "index"]
  count_na = count_na.drop(count_na[count_na["na count"] == 0].index)

  # get the 1st, 2nd, and 3rd quantile values
  quantiles = np.array(count_na["na count"].quantile([0.25, 0.5, 0.75]))

  # separate data into categories according to their quantile
  count_na.loc[count_na["na count"] < quantiles[0], "na amount"] = "low"
  count_na.loc[(count_na["na count"] >= quantiles[0]) & 
               (count_na["na count"] < quantiles[1]),"na amount"] = "low-mid"
  count_na.loc[(count_na["na count"] >= quantiles[1]) & 
               (count_na["na count"] < quantiles[2]),"na amount"] = "high-mid"
  count_na.loc[(count_na["na count"] >= quantiles[2]),"na amount"] = "high"
  
  # return the missing summary dataframe
  return count_na