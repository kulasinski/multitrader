from model_tuner import ModelTuner
# TODO: dodac close and vol var, jako feature
PARAMS = {
    'features_file': 'multitrader/features.csv',
    'split_method': 'random', # random | date | ticker
    'split_seed': 5, # random split only
    'test_size_pct': 0.25, # random split only
    'test_split_date': '2022-10-01', # date split only
    'test_tickers': ['AAPL'], # ticker split only

    'columns_x': ['eps_est', 'eps_actual', 'eps_diff',
       'eps_diff_pct', 
    #    'tot_assets', 'common_stock', 'tot_cap', 'tot_debt',
    #    'net_debt', 'ord_shares', 'tot_reve', 'gross_profit', 'op_expense',
    #    'op_income', 'avg_shares', 'prev_eps_actual', 'prev_eps_diff',
    #    'expense_to_income',
    #    'eps_q_change', 'eps_q_change_pct',
    #    'assets_to_profit', 'profit_to_reve',
       'debt_to_debt', 'debt_to_cap', 
       'on_earnings_close', 'on_earnings_volume', 
       'market_cap'
       ],
    'columns_y': ['tgt'],
    'columns_info': ['ticker', 'earnings_date','history_start_date', 'history_end_date', 'price_min', 'price_max','tgt'],

    'classifier_params': {
        'learning_rate': 0.1, # wrazliwy parametr! im wyzszy, zmniejsza sie precicion a zwieksza total_pos
        'n_estimators': 30, # n_est influences the threshold, the larger, the higher threshold needed...
        'subsample': .8,
        
        #'random_state': 5, #7,

        # default:
        'max_depth': 2, # default=3
        'loss': 'exponential',
        'criterion': 'squared_error'
    },
    'classifier_seeds': [1,2,3,4,5,6,7,8,9,10],
    'threshold': 0.75, # this should be automatically figured out... TODO
    'target_recall': 0.05, #2/m-c=24/year=72/1830~=0.04
}

MT = ModelTuner(**PARAMS)
# print(MT.y_preds)
# print(MT.F[MT.F.ticker=='MSFT'])
# print(MT.F.earnings_date.min())