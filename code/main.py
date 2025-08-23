from modeling.models_by_window import (
    fit_eval_window_model, 
    main2,
    eval_model
)
from setup.fetch_new_dat import fetch_new
import sys

if __name__ == "__main__":
    # set options
    if len(sys.argv) < 3:
        print("Usage: python3 {} fetch_true fit_true")
        print("\tfit_true or fit_false can be used")
        print("\tfetch_true or fetch_false can be used")
        exit()

    if sys.argv[1] == 'fetch_true':
        # update the data
        fetch_new(
            current_date = None, 
            key = None, 
            test = False, 
            no_key = False, 
            debug = False, 
            base_data_file = '../data/clean_model_data2.csv'
        )
        
    elif sys.argv[1] == 'eval_true': 
        eval_model(
            k=2
        )

    if sys.argv[2] == 'fit_true':
        # run modeling
        main2(
            main_features=[
                'prev_driver_position', 
                'prev_driver_wins', 
                'prev_construct_position', 
                'prev_construct_wins'
            ],
            vars = [
                'strt_len_median', 'strt_len_min',
                'avg_track_spd', 'corner_spd_median', 
                'corner_spd_max', 'corner_spd_min',
                'num_slow_corners', 'num_fast_corners', 
                'num_med_corners'
            ],
            start_data = '../data/clean_model_data2.csv',
            pred_round = 15,
            k = 3,
            year = 2025,
            std_errors = True, 
            boot_trials = 100,
            predictions_folder = "../results/zandfort"
        )
