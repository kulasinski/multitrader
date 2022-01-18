import numpy as np
import pandas as pd

class Strategy():
    
    def __init__(self, 
                name='Strategy',
                params = {} 
                ):
        self.name = name
        self.indicators = ['MA20']
        self.params = params
    
    def get_indicators(self):
        return self.indicators
    
    def check(self, data, indicators, curr_shares):
        """
            Takes in the ochl data, indicators, current amount of shares
            Returns:
            * amount of shares to buy, 0 if no move, negative if sell
            * 0 if at market, else limit price
            * [0..1] of how certain a trade is
        """
        shares_change = None
        limit = None
        quality = 1.
        return shares_change, limit, quality
        
class MA20Strat(Strategy):
    
    def __init__(self, 
                name='MA20Strat',
                params = {} 
                ):
        self.name = name
        self.indicators = ['MA20']
        self.params = params
    
    def check(self, data, indicators, curr_shares):
        """
            Takes in the ochl data, indicators, current amount of shares
            Returns:
            * amount of shares to buy, 0 if no move, negative if sell
            * 0 if at market, else limit price
            * [0..1] of how certain a trade is
        """
        shares_change = None
        limit = None
        quality = 1.
        
        curr_close = data.Close.iloc[-1]
        curr_ma = indicators.MA20.iloc[-1]
        try:
            last_close = data.Close.iloc[-2]
        except:
            last_close = None
        
        if curr_shares==0: # check if buy
            if curr_close < 0.95 * curr_ma and last_close<=curr_close:
                # BUY
                shares_change = +1
        else: # check if sell
            if curr_close > 1.05 * curr_ma and last_close>curr_close:
                # SELL
                shares_change = -curr_shares
        
        return shares_change, limit, quality