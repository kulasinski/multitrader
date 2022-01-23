import numpy as np
import pandas as pd
import numpy as np

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
    
    def check(self, data, indicators, curr_shares, cash_avail, ticker, buy_price=None):
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
                shares_change = np.floor(cash_avail/curr_close)
        else: # check if sell
            if curr_close > 1.05 * curr_ma and last_close>curr_close:
                # SELL
                shares_change = -curr_shares
        
        return shares_change, limit, quality

class RSIStrat(Strategy):
    
    def __init__(self, 
                name='RSIStrat',
                params = {} 
                ):
        self.name = name
        self.indicators = ['RSI']
        self.params = params
    
    def check(self, data, indicators, curr_shares, cash_avail, ticker, buy_price=None):
        """
            Buys if RSI falls below 30, sells if RSI is above 70
        """
        shares_change = None
        limit = None
        quality = 1.
        
        curr_close = data.Close.iloc[-1]
        curr_RSI = indicators.RSI.iloc[-1]

        if curr_RSI is None:
            return shares_change, limit, quality
        
        # print(f"curr RSI for {ticker} is {curr_RSI}")
        if curr_RSI<40 and curr_shares==0: # check if buy
            # BUY
            shares_change = np.floor(cash_avail/curr_close)
        elif curr_RSI>60 and curr_shares>0: # check if sell
            # SELL
            shares_change = -curr_shares
        
        return shares_change, limit, quality

class SLBStrat(Strategy):
    """
        Based solely on the SLB indicator
        https://atas.net/atas-possibilities/squeeze-momentum-indicator/
    """
    
    def __init__(self, 
                name='SLBStrat',
                params = {} 
                ):
        self.name = name
        self.indicators = ['SLBval','SLBtrend','STDEV20']
        self.params = params
    
    def check(self, data, indicators, curr_shares, cash_avail, ticker, buy_price=None):
        """
            Buys if squeeze off and val>0 and val_diff > 0
            Sells if squeeze on or val_diff < 0
        """
        shares_change = None
        limit = None
        quality = 1.
        
        curr_close = data.Close.iloc[-1]
        curr_val   = indicators.SLBval.iloc[-1]
        val_diff   = indicators.SLBval.diff()
        curr_val_diff = val_diff[-1]
        squeeze_off   = indicators.SLBtrend.iloc[-1]
        squeeze_on    = not squeeze_off

        try:
            last_val = indicators.SLBval.iloc[-2]
        except:
            last_val = None

        if curr_val is None or last_val is None or curr_val==np.nan or last_val==np.nan:
            return shares_change, limit, quality

        if curr_shares==0: # no shares yet, could buy
            if squeeze_off and curr_val<0 and curr_val_diff>0:
                # BUY
                print(f'buying: squeeze_off {squeeze_off} val {curr_val} diff {curr_val_diff}')
                shares_change = np.floor(cash_avail/curr_close)
        else: # already has shares, could sell
            if squeeze_on or curr_val_diff<0:
                print(f'selling: squeeze on {squeeze_on} val {curr_val} diff {curr_val_diff}')
                # SELL
                shares_change = -curr_shares
        
        return shares_change, limit, quality