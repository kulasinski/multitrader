from multitrader.indicators import Indicators
from multitrader.order import Order, Trade

import pandas as pd

class Commission():
    
    def __init__(self, 
                 min_cash=5., 
                 pct=0.29, 
                 buy_only=True):
        
        self.min_cash = min_cash
        self.pct = pct 
        self.buy_only = buy_only
        
    def get(self, shares, price, is_buy):
        
        if is_buy:
            return max(self.min_cash, shares*price*self.pct/100.)
        else:
            if self.buy_only:
                return 0.
            else:
                return max(self.min_cash, np.abs(shares)*price*self.pct/100.) 

class Account():
    
    def __init__(self, 
                 cash=0, 
                 commission=None,
                 verbose=True
                ):
        
        self.set_cash(cash)
        self.set_commission(commission)

        self.verbose = verbose
        self.strategy = None
        self.datadict = None        
        
    def set_cash(self, cash):
        self.cash = float(cash)
    
    def get_cash(self):
        return self.cash;
    
    def set_commission(self, commission):
        self.commission = commission
    
    def get_commission(self, shares, price, is_buy):
        return 0 if commission is None else self.commission.get(shares, price, is_buy);
    
    def add_data(self, ticker, df):
        """
            Add a single ticker with pandas dataframe
        """
        datadict = {} if self.datadict is None else self.datadict
        datadict[ticker] = df
        self.datadict = datadict
    
    def set_data(self, datadict):
        """
            Set data in the form {ticker: pandas df}
        """
        self.datadict = datadict
        
    def set_strategy(self, strategy):
        self.strategy = strategy
    
    def warm_up(self, tickers, start, end):
        """
            Checks if everything is ready for start and prefetches the data.
            Returns data as a dict of ticker:dataframe
                and indicators as a dict of ticker:indicator
        """
        if self.strategy is None:
            raise Exception("No strategy, exiting.")
        
        self.log(f"Running strategy: {self.strategy.name}\n")
            
        if self.datadict is None or self.datadict==[]:
            raise Exception("No data, exiting.")
            
        if tickers is None:
            tickers = list(self.datadict.keys())
            
        """
            Preparing raw data
        """
            
        self.data = {}
        self.shares = {}
        
        for t in tickers:
            d = self.datadict[t]
            if start is not None:
                d = d[d.Date >= start]
            if end is not None:
                d = d[d.Date <= end  ]
                
            self.data[t] = d.set_index('Date')
            
            self.shares[t] = 0
            
        """
            Calculating required indicators (by strategy)
        """
            
        self.indicators = {}
        
        for t in tickers:
            
            I = Indicators(self.data[t])
            i_dict = {}
            
            for i in self.strategy.get_indicators(): 
                i_dict[i] = I.get(i)
                
            self.indicators[t] = pd.DataFrame(i_dict)
        
        return tickers
            
    def benchmarking(self):
        pass
    
    def observers(self):
        pass
    
    def order_mngmt(self):
        pass
    
    def log(self, msg):
        if self.verbose:
            print(msg)
        
    def run(self, tickers=None, start=None, end=None):
        tickers = self.warm_up(tickers, start, end)
        
        # TBD tu trzeba polaczayc indexy zeby miec max zasieg
        dates = list(self.data['AAPL'].index)
        start = min(dates)
        end = max(dates)
        self.duration = len(dates)

        self.trades = []
        
        for date in dates:
            self.log(f"=== {date} ===")
            # TBD: store it! as well as cash and other observers...
            wallet = self.cash + sum([self.shares[t]*self.data[t].loc[date].Close for t in tickers])              
            # self.log(f"    cash: ${round(self.cash,2)} wallet: {round(wallet,2)}")
            
            for t in tickers:
                
                curr_close = round(self.data[t].loc[date].Close,2)

                # self.log(f"    {t} price: {curr_close} shares: {self.shares[t]}")
                
                shares_change, limit, quality = self.strategy.check(
                    self.data[t].loc[:date], 
                    self.indicators[t].loc[:date], 
                    self.shares[t]
                )

                if shares_change is None:
                    continue
                
                # TBD: change to order management...
                order = Order( # this is for market only, i.e. instant execution!
                    ticker=t,
                    shares=shares_change,
                    limit=limit,
                    quality=quality,
                    on_create_date=date,
                    on_create_price=curr_close,
                    valid_until=None,
                    memo=None,
                )

                if order.is_buy:
                    trade = Trade( order )
                    self.trades.append( trade )
                else: # if order is_close, search for matching trade that is open
                    matching_trades = [trade for trade in self.trades if (trade.is_open and trade.open_order.ticker==t)]
                    if len(matching_trades) > 1:
                        raise Exception("Too many matching trades!")
                    elif len(matching_trades) == 0:
                        self.log(f"    Warning: trying to close a non-existing position ({t})!")
                    else:
                        trade = matching_trades[0]
                        trade.set_close( order )


                # edit below based on trades and orders
                    
                self.shares[t] += shares_change
                self.cash -= shares_change * curr_close
        
        self.log("===    END     ===\n")
        
    def summary(self):

        print("=== SUMMARY ===")
        print(f"    DURATION: {self.duration} trading days")
        wallet_start = round(1, 1)
        wallet_end   = round(2, 1)
        wallet_diff  = wallet_end - wallet_start
        wallet_sign  = '+' if wallet_diff>0 else ''
        wallet_pct = round((wallet_diff / wallet_start ) *100., 1)
        print(f"    WALLET: ${wallet_start} -> ${wallet_end} ({wallet_sign}${wallet_diff}) {wallet_sign}{wallet_pct}%")
        
        pos_trades = len( [ trade for trade in self.trades if trade.gain>0. ] )
        neg_trades = len( [ trade for trade in self.trades if trade.gain<0. ] )
        err_trades = len( [ trade for trade in self.trades if trade.gain is None ] )
        tot_trades = pos_trades + neg_trades + err_trades
        assert len(self.trades)==tot_trades 
        print(f"    TRADES: POS {pos_trades} + NEG {neg_trades} + ERR {err_trades} = TOT {tot_trades}")

        trades_sorted = sorted([trade for trade in self.trades if trade.is_open == False], key=lambda x: x.gain)   
        if len(trades_sorted) > 0:
            best_trade = trades_sorted[0]
            worst_trade = trades_sorted[-1]
            print(f"    BEST TRADE:  ${best_trade.open_order.executed_price} -> ${best_trade.close_order.executed_price} (${best_trade.gain}) {best_trade.gain_pct}% "+\
                f"| {best_trade.open_order.ticker} {best_trade.open_order.executed_date} -> {best_trade.close_order.executed_date}")
            print(f"    WORST TRADE:  ${worst_trade.open_order.executed_price} -> ${worst_trade.close_order.executed_price} (${worst_trade.gain}) {worst_trade.gain_pct}% "+\
                f"| {worst_trade.open_order.ticker} {worst_trade.open_order.executed_date} -> {worst_trade.close_order.executed_date}")