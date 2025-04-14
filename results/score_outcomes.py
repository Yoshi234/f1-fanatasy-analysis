from sklearn.metrics import r2_score 
import pandas as pd

def score_predictions(preds, truth):
    '''
    Provides an R2 score of predictions made for drivers
    over input data
    '''
    score_val = r2_score(truth, preds)
    print("[INFO]: R2 coefficient of predictions = {}".format(score_val))
    
def main():
    preds = pd.read_csv("bahrain_predictions.csv")
    truth = pd.read_csv("bahrain_true.csv")
    
    preds = preds['grid']
    truth = truth['sp']
    score_predictions(preds, truth)
    
if __name__ == "__main__":
    main()