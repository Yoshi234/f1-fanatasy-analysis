# f1-fanatasy-analysis

Repository for selection of F1 drivers and teams for selection of fantasy teams and race weekend predictions

## Data Processing and Setup

1. run fetch new data 
2. then run the mod point distrib update functions and add podium 
   values (post hoc analysis)

## TODO

1. Update the `fetch_new` function so that it can run even without
   the availability of `clean_model_data2.csv`

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

