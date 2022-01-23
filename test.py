import warnings
warnings.filterwarnings("ignore")

from multitrader.account import Account, Commission
from multitrader.strategies import *
from multitrader.tools import load_ticker

### PARAMS ###

tickers = [
    'AAPL',
    # 'MSFT',
]
INITIAL_CASH = 1000
CASH_PER_TRADE = 0. # proportionally
START_DATE = '2020-11-01'
END_DATE = '2021-12-31'
STRATEGY = SLBStrat()

STOCKS_DIR = '/home/charlie/algo/stocks'
SP500_PATH = STOCKS_DIR+'/SP500.csv'
PLOT_OUTPUT = '/mnt/c/Users/kulas/Desktop/plot.png'

##############

account = Account()

account.set_cash(INITIAL_CASH)
account.set_cash_per_trade(CASH_PER_TRADE)
account.set_commission(Commission())
account.set_strategy(STRATEGY)

account.set_SP500(SP500_PATH)
for ticker in tickers:
    account.add_data(ticker, load_ticker(ticker, default_dir=STOCKS_DIR))

account.run(
        start= START_DATE,
        end  = END_DATE,
    )

account.summary()

account.plot(PLOT_OUTPUT)