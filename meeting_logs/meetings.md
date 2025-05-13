# Meeting 5/13/25

+ Meeting Purpose: Met with mOlt to discuss potential collaboration 
  on AI-based competitor for F1-predictions
+ Date / Time: 2025-05-13 14:00 - 14:30?
  + member 1: Josh
  + member 2: Nick
  + member 3: Tom
  
## Agenda Items

+ item 1: Discuss Tom's idea for an AI-predictor collaboration with 
  GridLock. 

## Summary for GridLock

+ Summary of Model and Training Approach
  + Training data from the previous 6 races is obtained, 
    (we conducted a previous study which showed that 
    data from the previous 6-7 races is optimal for 
    predicting the next race outcome in sequence.)
  + These features are weighted at a ratio of 
    - race 1: 11%
    - race 2: 11%
    - race 3: 11%
    - race 4: 22%
    - race 5: 22%
    - race 6: 22%
  + Weightings are accomplished by resampling records 
    at the required rate over the corresponding races. 
  + We can also give more aggressive weightings. 
  + This approach is better than training over all 
    F1 data available since trends in F1 are very short-lived, 
    and historical data quickly becomes irrelevant. 
  + **Features we use:** 
    + Driver / Team
    + Previous driver and constructor points
    + Previous driver and constructor position in the 
      standings
    + Track Features: 
      + Mean, median, Q1, Q3, minimum, maximum and 
        standard deviation of
        straight length for each track
      + Mean, minimum, maximum, and standard deviation 
        of track speed
      + Mean, Q1, Q2 (Median), Q3, Max, and Minimum of 
        track corner speeds
      + Number of fast corners (corners taken at a speed 
        greater than 235 km/h)
      + Number of slow corners (corners taken at a speed 
        slower than 136 km/h)
      + Number of corners (overall)
      + Circuit Length
  + Points features are standardized so that they indicate 
    the proportion of all points. For example, instead of 
    saying Oscar has 131 points, we give Oscar a score of
    0.19
  + **Modeling Approach**
    + Based on the input features, we normalize the data, 
      and apply lasso regression to fit coefficients to each 
      feature. We expect that most features will be irrelevant, 
      and so we avoid model fitting issues

## Goals for the Predictor

**Short Term Improvements**

1. Improve the granularity of features
   - additional divisions of some track features, especially 
     the number of "fast" and "slow" corners at each 
     track would be good. These features seem to quite useful
     at differentiating predictions across different tracks, 
     and adding a "medium" speed, or even adding "q1", "q2", 
     and "q3" features might help to improve things. 
   - reincorporating weather features would also help to 
     improve the accuracy of our predictions going forward. 
2. Add weather predictions back into the data - 
   - based on the results from Australia, weather can be a significant
     determinant of outcomes, especially for rookies. It would
     be good if we could somehow incorporate this information into 
     our modeling. Our previous approach did include this information, 
     but the API we were using limited our queries. 
   - The FastF1 API may include this information directly, but we 
     can also use the alternative API to access this information 
     separately (limited to 1000 queries per day)

**Long Term Improvements**

1. Update our website to include 
    - driver attributes and performance scores for each 
      weekend based on the coefficients obtained through
      our model
2. Validate an optimal weighting strategy based on historical data using 
   Lasso-regression. 
3. Experiment with alternative (potentially AI-based methods) for 
   improved modeling performance
4. Incorporate results from Free Practice to better estimate 
   performance changes (weekend-on-weekend) - especially for 
   lower-ranked teams. A more nuanced analysis of the input data
   is required than just using the results though. 

## Questions for GridLock

1. How can we be of assistance to GridLock?
2. What is the monetization strategy of GridLock?
3. Promotion methods? 

# Meeting 11/21/24

+ Meeting Purpose: Met to discuss modleing methodolgy
+ Date and Time: 2024-11-21 17:20-18:20
+ Team members in attendance:
  + member 1: Nick
  + member 2: Josh

## Agenda Items

+ item 1: decision tree regression /             
  classification task
+ item 2: next steps
  + SMOTE encoding for training
  + balance data
+ item 3: 
  + get points over past x races which meet criteria y,z,m 

## Key Responses on Agenda Items

+ response 1: 
+ response 2

## Questions Discussed

> identify areas of high importance which require follow-up

+ question 1
+ question 2

## Action Items and Next Steps

+ Action item 1
    + [JOSH]: work on finishing the modeling and parameterization 
    for XGBoosting, integrating SMOTENC and determining optimal 
    race data window
+ Action item 2
    + [NICK]: update features for prev-driver points to be balanced 
    divide points over the total at each round
    + [NICK]: work on getting new features (developing procedure to get
    new features) by adding the points over previous rounds, points over
    races with x track speed, etc. (choose different track chracteristics
    of interest and try to manually encode some things)
+ Action item 3

# Quick Meeting Recap

> should be sent to other members after the meeting?

# Meeting 11/15/24

+ Meeting Purpose: Discuss progress 11/15/24
+ Date and Time: 13:00:00-14:00:00 2024-11-15
+ Team members in attendance:
  + member 1: Nick Pfeifer
  + member 2: Josh Lee

## Agenda Items

+ item 1: Data window training parameterization
  + discussed interesting features of training over variable windows
  + Nick completed construction of the parameterized function with F1 and acc. 
    results to report for each value of $n$ (averaged over all races for a 
    given window)

## Key Responses on Agenda Items

+ response 1: 
  + Window functionality works well - need to update to optimize for 
    feature selection - whole point of our method is that the best features
    to choose change over different data windows and so we want to 
    include different features over time
+ response 2
  + decision trees may be better suited to handle interactions than 
    logistic regression models (implicit interaction encoding)

## Questions Discussed

> identify areas of high importance which require follow-up

+ question 1
+ question 2

## Action Items and Next Steps

+ Action item 1:
  + JOSH: finish getting 2024 data together [DONE]
+ Action item 2:
  + JOSH: forward selection procedure (automatic feature selection) for 
    (make it to return a list of features - input should be the entire 
    training data window of interest)
+ Action item 3:
  + NICK: decision tree, xgboosting, random forest
+ Action item 4:
  + JOSH: Add a prev points feature to indicate the proportion of all
    points allocated to a given driver

# Quick Meeting Recap

> should be sent to other members after the meeting?