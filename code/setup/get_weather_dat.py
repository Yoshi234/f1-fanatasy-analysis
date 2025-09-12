'''
This python file contains code for loading weather data 
from the visual crossing api. I borrow from the work done
by Kike Franssen as part of his master's thesis work. His 
work also incorporates weather data as a means to improve the predictive
capability of his models

In order to use this script, please go to www.visualcrossing.com, open 
an account, and replace `api_key` in the code below with your own 
personal api_key

With a free version of visual crossing, up to 1000 records can be 
obtained before your account is debited. As such, please make sure
to limit the number of records obtained via the API within a 24 hr 
period.
'''
# load necessary packages to interface with the API
import urllib.request
import urllib.error
import json
import requests
import pandas as pd
import numpy as np


def format_weather_dat(w_dat):
    '''
    reformat the weather data (should be run after get_weather_dat and
    get_locations_dat)

    w_dat --- dataframe with weather information in it
    '''
    for idx, row in w_dat.iterrows():
        if row['preciptype'] is None:
            continue
        else:
            w_dat.loc[idx, 'preciptype'] = row['preciptype'][0]

    # return the update weather data
    return w_dat


def get_locations_dat(folder:str = "../data"):
    '''
    preprocess function: organize a dataset with location and time
    data for each race event. Then use this data in order to obtain
    accurate information about the weather for each event
    '''
    # read relevant data
    circuits = pd.read_csv("{}/circuits.csv".format(folder))
    circuits.rename(columns = {"name":"circuit_name"}, inplace=True)

    races = pd.read_csv("{}/races.csv".format(folder))
    races.rename(columns={"name":"event_name"}, inplace=True)

    # merge data together to get the lat and long data from circuits
    merged_dat = pd.merge(circuits, races, on='circuitId', how='inner')

    if merged_dat.shape[0] == max(circuits.shape[0], races.shape[0]):
        return merged_dat
    else:
        print("Error: erroneous rows created. please check merging operation")
        return merged_dat
    

def get_weather_dat(loc_dat:pd.DataFrame, api_key=None, debug=False):
    '''
    takes as input the locations data frame (circuits merged with races) and 
    outputs the weather information associated with the following attributes
    required to be held within it:
    + latitude `lat` - case sensitive
    + longitude `lng` - case sensitive
    + date `date` - case sensitive

    See https://gitlab.com/kikefranssen/thesis_kf_f1/-/blob/main/1_load_weather.py?ref_type=heads
    for the code which is referenced here
    '''
    search_keys = ['tempmax','tempmin','temp','dew','humidity','precip','precipcover','preciptype','windspeed','winddir']
    APIkey = api_key
    BaseURL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
    EndUrl = f"?unitGroup=metric&elements=datetime%2Ctempmax%2Ctempmin%2Ctemp%2Cdew%2Chumidity%2Cprecip%2Cprecipprob%2Cprecipcover%2Cpreciptype%2Cwindspeed%2Cwinddir%2Cvisibility&include=days&key={APIkey}&contentType=json"
    
    if api_key is None:
        print("[ERROR]: Missing weather data for event {}".format(loc_dat['event_name'].unique()[0]))
        loc_dat[search_keys] = np.nan
        return loc_dat
    
    # iterate through loc_dat dataframe and obtain weather data
    n_qry = 0
    for idx, row in loc_dat[['lat', 'lng','date']].drop_duplicates().iterrows():
        lat = str(row['lat'])
        lng = str(row['lng'])
        date = str(row['date'])
        query = BaseURL + lat + "%2C" + lng + "/" + date + "/" + date + EndUrl
        response = requests.get(query)
        n_qry += 1
        if debug: 
            print(f"[DEBUG]: {query}")
            print(f"[DEBUG]: {response}")
        try:
            w_dat = json.loads(response.text)
        except:
            print("[ERROR]: Missing data for event {}".format(loc_dat['event_name'].unique()[0]))
            loc_dat[search_keys] = np.nan
            return loc_dat
        w_dat = w_dat['days'] 
        w_dict = {key:value for entry in w_dat for (key,value) in entry.items()}
        print(w_dict)

        for key in w_dict.keys():
            if key in search_keys: 
                if key == 'preciptype': 
                    if w_dict[key] is None:
                        w_dict[key] = 'no_precipitation'
                    else:
                        w_dict[key] = w_dict[key][0]
                        
                loc_dat.loc[loc_dat['date']==date, key] = w_dict[key]
    
    print("[INFO]: N_QUERIES = {}".format(n_qry))
    
    # return location dataframe with added weather information
    return loc_dat


def main(full_process=True):
    key = ''
    with open("private.txt", "r") as f:
        key = f.readlines()[0].strip('\n')
        
    if full_process == True:
        folder = "../../data"
        data = get_locations_dat(folder)
        data = data.loc[data['year'] >= 2000]
        data = get_weather_dat(data, key)
        data = format_weather_dat(data)
        # save data as feather
        data.to_feather("{}/races_circuits_weather.feather".format(folder))
    else:
        folder = "../../data"
        data = pd.read_feather("{}/races_circuits_weather.feather".format(folder))
        data = format_weather_dat(data)
        data.to_feather("{}/races_circuits_weather.feather".format(folder))


if __name__ == "__main__":
    main(False)



    





