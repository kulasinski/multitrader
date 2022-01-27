from multitrader.indicators import Indicators
from multitrader.order import Order, Trade

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

class Commission():
    """
        Commission calculator
    """
    
    def __init__(self, 
                 min_cash=5., 
                 pct=0.29, 
                 buy_only=True):
        
        self.min_cash = min_cash
        self.pct = pct 
        self.buy_only = buy_only
        
    def get(self, shares, price):
        """
            Returns commission value based on shares and price
        """
        shares = np.abs(shares)
        if shares > 0:
            return max(self.min_cash, shares*price*self.pct/100.)
        else:
            if self.buy_only:
                return 0.
            else:
                return max(self.min_cash, np.abs(shares)*price*self.pct/100.) 

class Account():
    """
        Main running engine
    """
    
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

        matplotlib.use('Agg')
        
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
        
        self.log(f"=== Running strategy: {self.strategy.name} ===")
            
        if self.datadict is None or self.datadict==[]:
            raise Exception("No data, exiting.")
            
        if tickers is None:
            tickers = list(self.datadict.keys())
            self.tickers = tickers
            
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
            Resetting SP500 dates
        """
        if self.SP500 is not None:
            self.SP500 = self.SP500.loc[start:end]
            
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
        """
            Sets SP500 benchmark
        """
        self.SP500 = df = pd.read_csv(fpath) \
                .sort_values('Date') \
                .set_index(index, drop=True)

    def do_benchmark(self,date,tickers,start_date):
        """
            Calculates buy-and-hold benchmark for each ticker
        """
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
        """
            Add various observables as a time series
        """

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
        values['shares'] = [sum([self.shares[t] for t in tickers])]
        values['fill_rate'] = [1. - self.cash / (self.cash + in_stocks)]

        if self.observers is None:
            values['drawdown_pct'] = [0.]
            values['max_drawdown_pct'] = [0.]
        else:
            values['drawdown_pct'] = [ round(values['wallet'][0] / self.observers['wallet'].max() *100. - 100., 1) ]
            values['max_drawdown_pct'] = self.observers['drawdown_pct'].min()

        if self.observers is None:
            self.observers = pd.DataFrame(values, index=[date])
        else:
            self.observers = self.observers.append(pd.DataFrame(values, index=[date]))

    def order_mngmt(self, t, date, new_order):
        """
            This is the broker heart...
            If no order, try to close open order
            Otherwise, replace with new order
        """
        price, shares_change = 0, 0
        matching_trades = [trade for trade in self.trades if (trade.is_open and trade.open_order.ticker==t)]
        if len(matching_trades) > 1:
            raise Exception("Too many matching trades!")

        curr_close = self.data[t].Close.loc[date]
        curr_open  = self.data[t].Open.loc[date]
        curr_high  = self.data[t].High.loc[date]
        curr_low   = self.data[t].Low.loc[date]
        
        # print("==>",date,'order mngmt')

        """
            Management of trades and orders starts here
        """
        # if new_order is None: # no new order
            # print("1 no new order")

        if len(matching_trades)>0: # try to close buy or sell
            # print(f"2 found {len(matching_trades)} matching trades that could be closed?")
            trade = matching_trades[0]

            if trade.close_order is None: # there is only BUY order

                if trade.open_order.executed_date is None: # buy order is not closed
                    if trade.open_order.is_market: # this is market order, buy at open
                        # print("3 this is market order, buy at open")
                        trade.open_order.execute(curr_open, date)
                        return curr_open, trade.open_order.shares
                    else: # this is a limit order
                        if trade.open_order.limit > curr_low: # when limit price is within the day's range
                            # print("4 executing the limit buy, because within the day's range")
                            trade.open_order.execute(trade.open_order.limit, date)
                            return trade.open_order.limit, trade.open_order.shares

                else: # buy order is closed, nothing to do
                    # print("5 nothing to do, buy order is closed and no open sell order...")
                    pass

            else: # handle pending SELL order; no need to check is is closed because the trade would be closed
                # print("6 handling a pending sell order")
                if trade.close_order.is_market: # handle a market SELL
                    # print("7 handling market SELL")
                    trade.close_order.execute(curr_open, date)
                    trade.try_close()
                    return curr_open, trade.close_order.shares
                else: # handle a limit SELL
                    # print("8 handling limit sell")
                    if trade.close_order.limit < curr_high: # when limit price is within the day's range
                        # print("9 executing the limit sell, because within the day's range")
                        trade.close_order.execute(trade.close_order.limit, date)
                        trade.try_close()
                        return trade.close_order.limit, trade.close_order.shares


            # else: # nothing to do : no new order and no market/limit calls to do
                # print("10 nothing to do: no new order and no market/limit calls to do...")
                # pass

        if new_order is not None: # some new order
            # print("11 handling new order")
            matching_trades = [trade for trade in self.trades if (trade.is_open and trade.open_order.ticker==t)]
            if new_order.is_buy: # new order is BUY
                # print("12 new order is BUY")
                if len(matching_trades)==0: # no open orders, placing new order in a new trade
                    # print("13 placing new order in a new trade")
                    trade = Trade( new_order )
                    self.trades.append( trade )
                else: # placing a new order in an old trade, replacement
                    # print("14 placing a new order in an old trade")
                    trade = matching_trades[0]
                    trade.open_order = new_order
                    trade.close_order = None # just to be sure...

            else: # new order is SELL
                # print("15 new order is SELL")
                if len(matching_trades) == 0:
                    self.log(f"    Warning: trying to close a non-existing position ({t})!")
                else: # replacing an old sell order or installing a new one
                    # print("16 replacing an old sell order or installing a new one")
                    trade = matching_trades[0]
                    trade.close_order = new_order

        return price, shares_change
    
    def log(self, msg):
        if self.verbose:
            print(msg)
        
    def run(self, tickers=None, start=None, end=None):
        """
            Market simulator: the actual run through the dates, for each ticker
        """

        tickers = self.warm_up(tickers, start, end)
        
        """
            Init dates list
        """
        indexes = [list(self.data[t].index) for t in tickers]
        dates = sorted(list(set([a for b in indexes for a in b])))
        self.dates = dates
        start = min(dates)
        end = max(dates)
        self.duration = len(dates)

        
        """
            Main loop
        """
        for date in dates:
            # self.log(f"=== {date} ===")
                        
            for t in tickers:
                
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
                    buy_price=None,
                )

                if shares_change is not None:
                    new_order = Order( # this is for market only, i.e. instant execution!
                        ticker=t,
                        shares=shares_change,
                        limit=limit,
                        quality=quality,
                        on_create_date=date,
                        on_create_price=curr_close,
                        valid_until=None,
                        memo=None,
                    )
                else:
                    new_order = None

                """
                    Handling trades and orders
                """

                price, shares_change = self.order_mngmt(t, date, new_order)

                if shares_change!=0:
                    self.shares[t] += shares_change
                    self.cash -= shares_change * price

                """
                    Handle commission
                """
                curr_commission = self.commission.get(shares_change, curr_close)
                self.cash -= curr_commission
                self.commission_total += curr_commission
                
                # TODO!
                if self.cash < 0: # this should not happen, but who knows?
                    pass
                    # print("WARNING! Negative cash!")
                    # raise Exception("Not enough cash!")

            self.observe(date,tickers)
            self.do_benchmark(date,tickers,start)
            # self.log(f"    cash: ${round(self.cash,2)} wallet: {round(self.observers.loc[date]['wallet'],2)}")
        
        self.log("===    END     ===\n")
        
    def calc_wallet_gains(self):
        """
            Wallet calculations for summary
        """
        wallet_start = self.observers.iloc[0]['wallet']
        wallet_end   = self.observers.iloc[-1]['wallet']
        wallet_diff  = round(wallet_end - wallet_start, 2)
        wallet_sign  = '+' if wallet_diff>0 else ''
        wallet_pct = round((wallet_diff / wallet_start ) *100., 1)
        return wallet_start, wallet_end, wallet_diff, wallet_sign, wallet_pct

    def calc_benchmarks(self):
        """
            Benchmarks calculations for summary
        """
        sp500_benchmark = round(self.benchmarks.iloc[-1]['SP500'] * 100.-100.,1) if self.SP500 is not None else None
        avg_stock_benchmark = round(self.benchmarks[[c for c in self.benchmarks.columns if c!='SP500']]\
                                .iloc[-1].mean() * 100. - 100. ,1)
        return sp500_benchmark, avg_stock_benchmark

    def calc_trades(self):
        """
            Trades calculations for summary
        """
        pos_trades = len( [ trade for trade in self.trades if trade.is_pos()==True ] )
        neg_trades = len( [ trade for trade in self.trades if trade.is_pos()==False ] )
        err_trades = len( [ trade for trade in self.trades if trade.is_pos() is None ] )

        return pos_trades, neg_trades, err_trades

    def summary(self):

        print("=== SUMMARY ===")
        
        wallet_start, wallet_end, wallet_diff, wallet_sign, wallet_pct = self.calc_wallet_gains()
        print(f"    WALLET: ${wallet_start} -> ${wallet_end} ({wallet_sign}${wallet_diff}) {wallet_sign}{wallet_pct}%")
        
        print(f"    MAX DROWDOWN: {self.observers.iloc[-1]['max_drawdown_pct']}%")
        print(f"    AVG FILLRATE: {round(self.observers['fill_rate'].mean()*100.,0)}%")

        sp500_benchmark, avg_stock_benchmark = self.calc_benchmarks()
        print(f"    BENCHMARK: SP500: {sp500_benchmark}% | AVG STOCK {avg_stock_benchmark}%")

        print(f"\n    DURATION: {self.duration} trading days")

        pos_trades, neg_trades, err_trades = self.calc_trades()
        tot_trades = pos_trades + neg_trades + err_trades
        print(f"    TRADES: POS {pos_trades} + NEG {neg_trades} + ERR {err_trades} = TOT {tot_trades}")
        assert len(self.trades)==tot_trades

        trades_sorted = sorted([trade for trade in self.trades if trade.is_open == False], key=lambda x: x.gain)   
        if len(trades_sorted) > 0:
            best_trade = trades_sorted[-1]
            worst_trade = trades_sorted[0]
            avg_trade = round(sum([x.gain_pct for x in trades_sorted]) / len(trades_sorted), 1)
            print(f"    AVG TRADE: {avg_trade}%")
            print(f"    BEST TRADE:  ${best_trade.open_order.executed_price} -> ${best_trade.close_order.executed_price} (${best_trade.gain}) {best_trade.gain_pct}% "+\
                f"| {best_trade.open_order.ticker} {best_trade.open_order.executed_date} -> {best_trade.close_order.executed_date}")
            print(f"    WORST TRADE:  ${worst_trade.open_order.executed_price} -> ${worst_trade.close_order.executed_price} (${worst_trade.gain}) {worst_trade.gain_pct}% "+\
                f"| {worst_trade.open_order.ticker} {worst_trade.open_order.executed_date} -> {worst_trade.close_order.executed_date}")

        print(f"    COMMISSION TOTAL: ${round(self.commission_total,2)}")

    def plot(self, output, dpi=300):
        """
            Plot timeseries to file
        """

        dates = [datetime.strptime(d, '%Y-%m-%d') for d in self.observers.wallet.index]
     
        ticker_figs = len(self.data) * (1 + len(self.strategy.indicators))
        fig, axs = plt.subplots(5+ticker_figs, 1, figsize=(7., 7.+ticker_figs*1.), constrained_layout=True, sharex=True)

        fig.suptitle(self.strategy.name)

        for ax in axs:
            # ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1))) # every year
            ax.xaxis.set_minor_locator(mdates.MonthLocator()) # every month
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            ax.grid(True, which='minor', axis='x')

        ax=axs[0]
        ax.plot(dates, self.observers.wallet, label='wallet')
        if self.SP500 is not None:
                benchmark = self.SP500.Close/self.SP500.Close.iloc[0]*self.observers.wallet.iloc[0]
                ax.plot(dates, benchmark, color='g', label='SP500')
        ax.legend()
        ax.set_ylabel('Wallet', fontsize='medium')

        ax=axs[1]
        ax.plot(dates, self.observers.cash)
        ax.set_ylabel('Cash', fontsize='medium')

        ax=axs[2]
        ax.plot(dates, self.observers.shares)
        ax.set_ylabel('Shares', fontsize='medium')

        ax=axs[3]
        ax.plot(dates, self.observers.max_drawdown_pct, color='r')
        ax.set_ylabel('Drawdown %', fontsize='medium')

        ax=axs[4]
        ax.plot(dates, self.observers.fill_rate*100.)
        ax.set_ylabel('FillRate %', fontsize='medium')

        curr_subplot = 5

        """
            for each ticker...
        """
        # width of candlestick elements
        width = .8
        width2 = .1
        # colors to use
        col1 = 'blue'
        col2 = 'black'
        
        for t in self.data.keys():
            ax=axs[curr_subplot]
            prices = self.data[t]
            prices['Dates'] = dates
            prices.set_index('Dates', inplace=True)

            """
                Plot line
            """
            ax.plot(dates, self.data[t].Close, color='k', lw=0.2)

            """
                Plot candlesticks
            """
            up = prices[prices.Close>=prices.Open]
            down = prices[prices.Close<prices.Open]
            # plot up prices
            ax.bar(up.index,up.Close-up.Open,width,bottom=up.Open,color=col1)
            ax.bar(up.index,up.High-up.Close,width2,bottom=up.Close,color=col1)
            ax.bar(up.index,up.Low-up.Open,width2,bottom=up.Open,color=col1)
            # plot down prices
            ax.bar(down.index,down.Close-down.Open,width,bottom=down.Open,color=col2)
            ax.bar(down.index,down.High-down.Open,width2,bottom=down.Open,color=col2)
            ax.bar(down.index,down.Low-down.Close,width2,bottom=down.Close,color=col2)
            
            """
                Plot orders
            """
            for trade in self.trades:
                if trade.ticker!=t:
                    continue
                if trade.open_order.executed_date is not None:
                    ax.plot(datetime.strptime(trade.open_order.executed_date, '%Y-%m-%d'),
                            trade.open_order.executed_price,
                            'g^',
                            alpha=0.5)
                if trade.close_order is not None:
                    if trade.close_order.executed_date is not None:
                        ax.plot(datetime.strptime(trade.close_order.executed_date, '%Y-%m-%d'),
                                trade.close_order.executed_price,
                                'rv',
                                alpha=0.5)

            """
                Plot benchmark
            """
            if self.SP500 is not None:
                benchmark = self.SP500.Close/self.SP500.Close.iloc[0]*self.data[t].Close.iloc[0]
                ax.plot(dates, benchmark, color='g', label='SP500')
            ax.set_ylabel(f'$ {t}', fontsize='medium')
            ax.legend()

            curr_subplot += 1

            """
                Plot indicators
            """
            for ind in self.strategy.indicators:
                ax=axs[curr_subplot]

                if ind=='SLBval': # draw special case for SLB
                    val  = self.indicators[t][ind]
                    val.index = pd.to_datetime(val.index)
                    diff = val.diff()
                    up   = diff >  0
                    down = diff <= 0
                    pos  = val >  0
                    neg  = val <= 0
                    pos_up   = pos & up
                    pos_down = pos & down
                    neg_up   = neg & up
                    neg_down = neg & down
                    ax.bar(val.index[pos_up], val[pos_up], color='lime')
                    ax.bar(val.index[pos_down], val[pos_down], color='darkgreen')
                    ax.bar(val.index[neg_up], val[neg_up], color='yellow')
                    ax.bar(val.index[neg_down], val[neg_down], color='red')
                else: # all other indicators
                    ax.plot(dates, self.indicators[t][ind])

                ax.set_ylabel(f'{ind}-{t}', fontsize='medium')
                curr_subplot += 1

        fig.savefig(output, dpi=dpi)

    def output(self, custom_str='', fname=None, append=True):
        """
            Outputs summary to console and/or a .csv file
        """
        wallet_start, wallet_end, wallet_diff, wallet_sign, wallet_pct = self.calc_wallet_gains()
        sp500_benchmark, avg_stock_benchmark = self.calc_benchmarks()
        pos_trades, neg_trades, err_trades = self.calc_trades()

        out_dict = {
            'DATE_RUN': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'DATE_START': self.dates[0],
            'DATE_END': self.dates[-1],
            'WALLET_START': wallet_start,
            'WALLET_END': wallet_end,
            'WALLET_GAIN': wallet_pct,
            'MAX_DROWDOWN': self.observers.iloc[-1]['max_drawdown_pct'],
            'SP500': sp500_benchmark,
            'STOCK_GAIN': avg_stock_benchmark,
            'TRADES': pos_trades+neg_trades+err_trades,
            'TRADES_POS': pos_trades,
            'TRADES_NEG': neg_trades,
            'TICKERS': ' '.join(self.tickers),
            'STRATEGY': self.strategy.name,
            'COMMISIONS': round(self.commission_total,2),
            'CUSTOM': custom_str,
        }

        if fname is not None:
            with open(fname, 'a' if append else 'w') as f:
                if not append:
                    f.write(','.join(out_dict.keys()))
                f.write( '\n' + ','.join( [str(v) for v in out_dict.values()] ) )
        return out_dict




