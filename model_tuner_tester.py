from model_tuner import ModelTuner
import numpy as np
# TODO: dodac close and vol var, jako feature
PARAMS = {
    'features_file': 'multitrader/features.csv',
    'split_method': 'random', # random | date | ticker
    'split_seed': 3, # random split only ; not used if tested in loop
    'test_size_pct': 0.25, # random split only
    'test_split_date': '2022-10-01', # date split only
    'test_tickers': ['AAPL'], # ticker split only

    'columns_x': [
        'eps_est', 
        'eps_actual', 
        'eps_diff',
        # 'eps_diff_pct', 
       'on_earnings_close', 
       'on_earnings_volume', 
    #    'log_cap',
       'market_cap',
       'price_var'
       ],
    'columns_y': ['tgt'],
    'columns_info': ['ticker', 'earnings_date','history_start_date', 'history_end_date', 'price_min', 'price_max', 'price_var', 'tgt'],

    'classifier_params': {
        'learning_rate': 0.1, # wrazliwy parametr! im wyzszy, zmniejsza sie precicion a zwieksza total_pos
        'n_estimators': 30, # n_est influences the threshold, the larger, the higher threshold needed...
        'subsample': .8,

        # default:
        'max_depth': 2, # default=3
        'loss': 'exponential',
        'criterion': 'squared_error'
    },
    'classifier_seeds': [1,2,3,4,5,6,7,8,9,10],
    'threshold': 0.78, # this should be automatically figured out... TODO
    'target_recall': 0.05, #2/m-c=24/year=72/1830~=0.04 TODO this is not used yet! OR 72/4000~=0.018
}

# RANDOM: best so far 84.1/4.16 przy depth=1, th=0.7, sample=0.8, n_est=30, lr=0.1 albo 85/3 przy depth=2 i th=0.78
# DATE/2022-10-01 : best 92/4 above with depth=2, th=0.7

## SINGLE TEST ##
# MT = ModelTuner(**PARAMS)
# MT.feature_importance()

## LOOP TEST - more reliable, makes sense only when random spliting ## 
results = []
for split_seed in range(10):
    PARAMS['split_seed'] = split_seed
    MT = ModelTuner(**PARAMS)
    results.append(MT.out)
MT.feature_importance()
print("AVG PRECISION:",np.mean([o['precision'] for o in results]))
print("AVG RECALL:   ",np.mean([o['recall'] for o in results]))