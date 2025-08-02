# f1-fanatasy-analysis

Repository for selection of F1 drivers and teams for selection of fantasy teams and race weekend predictions

## Data Warnings

When running `fetch_new.py` for the first time, you may 
encounter a series of warnings. 

1. `[INFO]: add <driver_name>` - this means that you need 
   to add an entry for the specified driver in 
   'data/drivers.csv'
2. `[ERROR]: no circuits found for <location>`: add the 
   specified location to the 'circuits.csv' file and 
   create a new entry if the circuit is just totally new, 
   or copy the existing entry for the circuit, and change 
   the corresponding `location` column entry - keep all other fields the same

## Data Processing and Setup

1. run fetch new data 
2. then run the mod point distrib update functions and add podium 
   values (post hoc analysis)

## TODO

1. Update the `fetch_new` function so that it can run even without
   the availability of `clean_model_data2.csv`
   + NICK: go through `fetch_new` and clean up the model / iterate 
     until the functionality works so that we can run the data aggregation
     without needing valid previous records.

2. set `weather=True` when fetching data from fastf1 api and use these updated features to improve model performance

3. Update the number of medium-speed corners in the track data output (see `get_track_speeds.py`)

<center>

```python
full_list_of_speeds = []
for track in all_tracks:
   get_speeds = list(track_speeds_at_corners)
   full_list_of_speeds += get_speeds

x = np.array([
   200, 300, 100, 150, 200, 600,
   100, 200, 300, 200, 300, 200,
])
```

</center>

Get the quantiles from this full data set. 

