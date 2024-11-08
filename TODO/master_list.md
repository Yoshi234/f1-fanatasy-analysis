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