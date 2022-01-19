from multitrader.indicators import Indicators
from multitrader.order import Order, Trade

import pandas as pd
import numpy as np

class Commission():
    
    def __init__(self, 
                 min_cash=5., 
                 pct=0.29, 
                 buy_only=True):
        
        self.min_cash = min_cash
        self.pct = pct 
        self.buy_only = buy_only
        
    def get(self, shares, price):
        shares = np.abs(shares)
        if shares > 0:
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
                 cash_per_trade = 0,
                 verbose=True
                ):
        
        self.set_cash(cash)
        self.set_commission(commission)
        self.cash_per_trade = cash_per_trade

        self.verbose = verbose
        self.strategy = None
        self.datadict = None     
        self.SP500 = None   
        
    def set_cash(self, cash):
        self.cash = float(cash)
    
    def get_cash(self):
        return self.cash;
    
    def set_cash_per_trade(self, cash_per_trade):
        self.cash_per_trade = cash_per_trade

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
            Initializing observers, benchmarks, and commission
        """

        self.observers = None
        self.benchmarks = None
        self.commission_total = 0

        """
            Init trades
        """
        self.trades = []

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
                
            self.indicators[t] = pd.DataFrame(i_dict, index=self.data[t].index)
        
        return tickers

    def set_SP500(self, fpath, index='Date'):
        self.SP500 = df = pd.read_csv(fpath) \
                .sort_values('Date') \
                .set_index(index, drop=True)

    def do_benchmark(self,date,tickers,start_date):
        values = {
            'SP500': None if self.SP500 is None else self.SP500.loc[date].Close / self.SP500.loc[start_date].Close
        }

        for t in tickers:
            pct = self.data[t].loc[date].Close / self.data[t].loc[start_date].Close
            values[t] = [pct]

        if self.benchmarks is None:
            self.benchmarks = pd.DataFrame(values, index=[date])
        else:
            self.benchmarks = self.benchmarks.append(pd.DataFrame(values, index=[date]))
    
    def observe(self,date,tickers):
        values = {
            "cash": [self.cash]
        }

        in_stocks = 0
        for t in tickers:
            values[f"#{t}"] = [self.shares[t]]
            values[f"${t}"] = [self.shares[t] * round(self.data[t].loc[date].Close, 2)]

            in_stocks += values[f"${t}"][0]
            values['in_stocks'] = [in_stocks]

        values['wallet'] = [self.cash + in_stocks]

        if self.observers is None:
            values['max_drowdown_pct'] = 0.
        else:
            values['max_drowdown_pct'] = round(self.observers['wallet'].min() / self.observers['wallet'].max() *100. - 100., 1)

        if self.observers is None:
            self.observers = pd.DataFrame(values, index=[date])
        else:
            self.observers = self.observers.append(pd.DataFrame(values, index=[date]))

    def order_mngmt(self):
        pass
    
    def log(self, msg):
        if self.verbose:
            print(msg)
        
    def run(self, tickers=None, start=None, end=None):
        tickers = self.warm_up(tickers, start, end)
        
        """
            Init dates list
        """
        # TBD tu trzeba polaczayc indexy zeby miec max zasieg
        dates = list(self.data['AAPL'].index)
        start = min(dates)
        end = max(dates)
        self.duration = len(dates)

        
        
        for date in dates:
            self.log(f"=== {date} ===")
                        
            for t in tickers:
                # self.log(f"{self.indicators[t].RSI.loc[date]}")    
                curr_close = round(self.data[t].loc[date].Close,2)

                """
                    Setting cash available for trade
                    if cash_per_trade is 0 then divide cash proportionally per ticker
                    Otherwise, attribute the cash_per_trade value times cash
                """

                cash_avail = self.cash/len(tickers) if self.cash_per_trade==0 else self.cash*self.cash_per_trade

                """
                    Asking strategy
                """
                
                shares_change, limit, quality = self.strategy.check(
                    self.data[t].loc[:date], 
                    self.indicators[t].loc[:date], 
                    self.shares[t],
                    cash_avail,
                    t,
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


                # edit below based on trades and orders? 
                # tbd no cash check
                if order.is_market:
                    if (shares_change * curr_close) <= self.cash:
                        self.shares[t] += shares_change
                        self.cash -= shares_change * curr_close
                    else:
                        raise Exception("Not enough cash!")

                """
                    Handle commission
                """
                curr_commission = self.commission.get(shares_change, curr_close)
                self.cash -= curr_commission
                self.commission_total += curr_commission

            self.observe(date,tickers)
            self.do_benchmark(date,tickers,start)
            self.log(f"    cash: ${round(self.cash,2)} wallet: {round(self.observers.loc[date]['wallet'],2)}")
        
        self.log("===    END     ===\n")
        
    def summary(self):

        print("=== SUMMARY ===")
        
        wallet_start = self.observers.iloc[0]['wallet']
        wallet_end   = self.observers.iloc[-1]['wallet']
        wallet_diff  = round(wallet_end - wallet_start, 2)
        wallet_sign  = '+' if wallet_diff>0 else ''
        wallet_pct = round((wallet_diff / wallet_start ) *100., 1)
        print(f"    WALLET: ${wallet_start} -> ${wallet_end} ({wallet_sign}${wallet_diff}) {wallet_sign}{wallet_pct}%")
        
        print(f"    MAX DROWDOWN: {self.observers.iloc[-1]['max_drowdown_pct']}%")

        sp500_benchmark = round(self.benchmarks.iloc[-1]['SP500'] * 100.-100.,1) if self.SP500 is not None else None
        avg_stock_benchmark = round(self.benchmarks[[c for c in self.benchmarks.columns if c!='SP500']]\
                                .iloc[-1].mean() * 100. - 100. ,1)
        print(f"    BENCHMARK: SP500: {sp500_benchmark}% | AVG STOCK {avg_stock_benchmark}%")

        print(f"\n    DURATION: {self.duration} trading days")

        pos_trades = len( [ trade for trade in self.trades if trade.is_pos()==True ] )
        neg_trades = len( [ trade for trade in self.trades if trade.is_pos()==False ] )
        err_trades = len( [ trade for trade in self.trades if trade.is_pos() is None ] )
        tot_trades = pos_trades + neg_trades + err_trades
        print(f"    TRADES: POS {pos_trades} + NEG {neg_trades} + ERR {err_trades} = TOT {tot_trades}")
        assert len(self.trades)==tot_trades

        trades_sorted = sorted([trade for trade in self.trades if trade.is_open == False], key=lambda x: x.gain)   
        if len(trades_sorted) > 0:
            best_trade = trades_sorted[0]
            worst_trade = trades_sorted[-1]
            avg_trade = round(sum([x.gain_pct for x in trades_sorted]) / len(trades_sorted), 1)
            print(f"    AVG TRADE: {avg_trade}%")
            print(f"    BEST TRADE:  ${best_trade.open_order.executed_price} -> ${best_trade.close_order.executed_price} (${best_trade.gain}) {best_trade.gain_pct}% "+\
                f"| {best_trade.open_order.ticker} {best_trade.open_order.executed_date} -> {best_trade.close_order.executed_date}")
            print(f"    WORST TRADE:  ${worst_trade.open_order.executed_price} -> ${worst_trade.close_order.executed_price} (${worst_trade.gain}) {worst_trade.gain_pct}% "+\
                f"| {worst_trade.open_order.ticker} {worst_trade.open_order.executed_date} -> {worst_trade.close_order.executed_date}")

        print(f"    COMMISSION TOTAL: ${round(self.commission_total,2)}")