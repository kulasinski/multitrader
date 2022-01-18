from multitrader.account import Account, Commission
from multitrader.strategies import MA20Strat
from multitrader.tools import load_ticker

### PARAMS ###

tickers = [
    'AAPL',
    'MSFT',
]
INITIAL_CASH = 1000
START_DATE = '2020-02-26'
END_DATE = '2020-10-15'
STRATEGY = MA20Strat()

STOCKS_DIR = '/home/charlie/algo/stocks'
SP500_PATH = STOCKS_DIR+'/SP500.csv'

##############

account = Account()

account.set_cash(INITIAL_CASH)
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