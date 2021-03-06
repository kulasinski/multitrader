import numpy as np
import pandas as pd

def load_ticker(ticker, default_dir='.'):
    df = pd.read_csv(default_dir+'/'+ticker+'.csv')
    df.sort_values('Date',inplace=True)
    return df