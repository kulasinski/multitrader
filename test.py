from multitrader.account import Account, Commission
from multitrader.strategies import MA20Strat
from multitrader.tools import load_ticker

### PARAMS ###

ticker = 'AAPL'
INITIAL_CASH = 1000
START_DATE = '2020-02-26'
END_DATE = '2020-10-15'
STRATEGY = MA20Strat()

##############

account = Account()

account.set_cash(INITIAL_CASH)
account.set_commission(Commission())
account.set_strategy(STRATEGY)


account.add_data(ticker, load_ticker(ticker))

account.run(
        start= START_DATE,
        end  = END_DATE,
    )

account.summary()