## 9/14/2025

Errors with model evaluation:
- imola gp: driverId 865 is printed as missing for each trial
  - somehow, the `drivers` list returned by `get_encoded_data()` is 
    obtaining colapinto iformation
  - Then, at the spanish grand prix, doohan is missing from the 
    drivers pool. This is expected since he did not have a driver
    after the spanish grand prix.
- actually, the main reason errors seem to be popping up is because
  somehow, 'Doohan' is retained as part of the `drivers` list, and this 
  causes errors to show up

## 5/4/2025

1. I discovered a very strange bug today. I loaded up the data from 
   the Miami GP, and then it made predictions based on the same 
   data set, but when I adjusted the values, predictions remained 
   the same for round 7 based on rounds 1 - 6 (rounds 1 - 6). 
   However, the miami data was invalid. When I opted to remove 
   this data from the data set, things went back to normal, but 
   I am very puzzled as to why. This is a potentially serious 
   bug with my program and requires further investigation. 

## 4/21/2025

1. currently corner-speed cutoffs are hard-coded. 
Update these values to use dynamically encoded information instead.
Build a database of values for all tracks instead of just using 
hard-coded values. Maybe want to pivot to SQLite instead of a bunch 
of .csv files?

TODO: add q1, q2, q3, and q4 corner-speed classes for the 
"num_fast_corners" and "num_slow_corners" counts. It seems 
that medium speed corners are also an important consideration, 
aside from just "slow corners". For example, HAAS seems to do 
better at slower-speed circuits, but in reality they are more of a
medium-speed circuit dominant car. Likewise, Williams seems stronger
at higher-speed tracks (similar to the racing bull car).

2. update mechanism for pushing predictions to the website hosted 
   via github

3. Add feature importance visualization, and driver performance 
   profiles based on coefficient weightings for track feature 
   interactions

4. Think about developing a weighting mechanism which accounts for 
   DNFs with more nuance - for example, instead of duplicating results
   from a given race at the same weight as every other driver, 
   if a DNF occurs, downweight that performance to a half or quarter 
   rate relative to the standardized weighting. Of course, if the number
   of samples for a given race is 1, then no weighting should be applied. 
   This shouldn't be the case for more important races though. 

## 4/02/2025

+ come up with a method to measure performance characteristics
  of cars and drivers over sectors of the track. For example, 
  taking the speed of the driver's qualifying lap over each 
  corner of the track. 

## 3/31/2025

data processing seems to be fixed completely. future updates - add data
sourcing for weather directly from fastf1 instead of using the separate
weather API - unless the queries are fixed - I can do 1000 per day it 
seems using the free version. 

next steps - train a model based on the previous 7 results. Weight the 
past two - three results exponentially more heavily than those from 
last season. 

Desirable weighting:
+ China 60% - should be the overwhelming weight
+ Australia 20%
+ 20% the previous 5 races from last year

Apply SMOTE and use l1-lasso regression to do the feature selection 
over the training data. 

## 3/30/2025

1. fix NaN grid order and driver and team ID's
   I believe this is due to a lack of ID's for some teams 
   with new names that have not been accounted for in the data. 
   Will need to print the new values so that they can match 
   in the data. 
   - this isuse is fixed now - was a data sourcing issue related 
     to the version of fastf1 being used. Data fixed now. 
2. Another issue popped up though:

```python
Traceback (most recent call last):
  File "/home/jjl20011/snap/snapd-desktop-integration/253/Lab/Projects/sports-analysis/f1-fanatasy-analysis/code/setup/fetch_new_dat.py", line 433, in <module>
    fetch_new(debug=True)
  File "/home/jjl20011/snap/snapd-desktop-integration/253/Lab/Projects/sports-analysis/f1-fanatasy-analysis/code/setup/fetch_new_dat.py", line 334, in fetch_new
    for driver in base[fastf1_dkey].unique():
  File "/home/jjl20011/miniconda3/envs/R_env/lib/python3.9/site-packages/pandas/core/generic.py", line 6299, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'SessionResults' object has no attribute 'unique'
```

## 3/29/2025

1. fix this bug

```python
Traceback (most recent call last):
  File "/home/jjl20011/snap/snapd-desktop-integration/253/Lab/Projects/sports-analysis/f1-fanatasy-analysis/code/setup/fetch_new_dat.py", line 401, in <module>
    fetch_new()
  File "/home/jjl20011/snap/snapd-desktop-integration/253/Lab/Projects/sports-analysis/f1-fanatasy-analysis/code/setup/fetch_new_dat.py", line 391, in fetch_new
    result = pd.concat([full_dat[og_dat.keys()].reset_index(drop=True),
  File "/home/jjl20011/miniconda3/envs/R_env/lib/python3.9/site-packages/pandas/core/frame.py", line 4108, in __getitem__
    indexer = self.columns._get_indexer_strict(key, "columns")[1]
  File "/home/jjl20011/miniconda3/envs/R_env/lib/python3.9/site-packages/pandas/core/indexes/base.py", line 6200, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
  File "/home/jjl20011/miniconda3/envs/R_env/lib/python3.9/site-packages/pandas/core/indexes/base.py", line 6252, in _raise_if_missing
    raise KeyError(f"{not_found} not in index")
KeyError: "['constructorId'] not in index"
```

It seems that somehow the constructorId key is being dropped from the data.
This is probably happening somewhere in the set_standings_data function. 
Or, set prev_round data function. Please examine these to see where 
the issues are. 

## 3/26/2025

1. update the standings data to use the same driver key selection
   based on the availability of "DriverId" or "FullName"
2. Rerun the data sourcing so that the data is complete enough for 
   analysis to be performed. 
   
## 3/25/2025

1. update the data sourcing so that the key availability is 
   checked. It seems that driver ids are not available for 
   all races, but abbreviations are - we may want to have an 
   option to check if the number of unique driver Id's is 0, 
   and then access data based on that 
   - the value they provide is equivalent to the "code" value in 
     the drivers.csv data set
2. Need to update the onedrive folder with the most up to 
   date data so that I don't need to keep re-running the same 
   code over and over again

## PROJECT TASK LIST 

> set [done] after you are finished with a given task 

1. Get Nick onboarded [DONE]

## JOSH

1. **Build procedure for automatic (multi-process) forward selection of features without 
   explicitly encoding everything ahead of time (not enough memory in the dataframe to do this
   unfortunately)**
   + step 1: subset $n$ races
   + step 2: duplicate race samples to weight data differently.
   + step 3: for each round, apply smote to balance the data and concatenate all subsets to form
     full feature set
   + step 4: train model on the feature set
   + step 5: evaluate performance on subsequent races (race $n+1, n+2, \dots, n+j$)
1. Figure out how to source and structure data for variable race training 
   data windows
2. Build system to parameterize the data sourcing
4. Modeling weights (IDEA):
   Resample the data by duplicating the number of samples present as a function of 
   their distance from the prediction sample(s): 
   We can tune the parameter for this weighting as a function of test performance.
   Namely, if we set weights as $w = mx + 1$ where $x=0$ indicates the furthest
   of the samples from the prediction sample, and the closest one as $x=n$ sample. 
   In my opinion, it makes sense to use $m\in[0,1]$ since we want the number of 
   samples to grow at a rate less than $1$ for each round which is closer to the 
   prediction sample. A starting parameter might be $0.5$ or something.
   Namely, the number of duplicates would be $w = \lfloor mx \rceil + 1$
   Suppose we had $k=5$ samples and $m=0.5$ for our weight parameter, then we
   should have the following duplications:
5. Format 2024 race data into the same form as the `clean_model_data.feather` file

<center>

round | $n$ duplicates
---|---
$0$ | $1 = \lfloor 0.5\cdot 0\rceil + 1$
$1$ | $2 = \lfloor 0.5\cdot 1\rceil + 1$
$2$ | $2 = \lfloor 0.5\cdot 2\rceil + 1$
$3$ | $3 = \lfloor 0.5\cdot 3\rceil + 1$
$4$ | $3 = \lfloor 0.5\cdot 4\rceil + 1$

</center>

5. alternatively, we could let $w = \lfloor (m + qx)x \rceil + 1$ or $w=\lfloor mx + qx^{2}\rceil + 1$
   where $m\in[0,1]$ and $q\in[0,0.1]$. Here, my idea is that the slope should change as a function of
   the value of $x$, like the slope should be increasing as $x$ increases (very marginally though).
   As a demonstration, if we let $m=0.5$ and $q=0.015$, the following duplicates are generated for each of $k$ 
   rounds. The only issue with this approach is that for $x \gt 31.2$, $w$ grows faster than $1$ additional 
   duplicate per round. Perhaps this is desirable behavior in the long term though. For example, if we wanted
   to train using a weighted sampling method over the entirety of data from 2018 to 2024 (which includes 115
   rounds of competition)

<center>

round | $n$ duplicates
---|---
$0$ | $1$
$1$ | $2$
$2$ | $2$
$3$ | $3$
$4$ | $3$

</center>

## NICK

3. **Test variable window on logistic models using the base data**
2. Perform forward feature selection for the base variable dataset
3. Conduct testing of 
    + decision trees
    + random forest (pruning)
    + xgboost 

## BOTH

1. Perform parameterization testing for data windows on the 
   models developed by Nick
2. Update website and figure out how to set the chron job for automatic updates