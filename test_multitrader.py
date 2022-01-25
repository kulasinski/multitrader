import multitrader as mt
import unittest

DFILE = 'test_data'

class TestAll(unittest.TestCase):
    def test_data(self):
        data = mt.tools.load_ticker(DFILE)
        cols = list(data.columns)
        assert('Date' in cols)
        assert('Close' in cols)
        assert('Open' in cols)
        assert('High' in cols)
        assert('Low' in cols)
        assert('Volume' in cols)
    
    def test_strategy_performance(self):
        account = mt.account.Account()
        account.set_cash(10000.)
        # account.set_cash_per_trade(CASH_PER_TRADE)
        account.set_commission(mt.account.Commission())
        account.set_strategy(mt.strategies.MA20Strat())

        # account.set_SP500(SP500_PATH)
    
        account.add_data('TEST', mt.tools.load_ticker(DFILE))

        account.run(
                start= '2020-01-01',
                end  = '2020-12-31',
            )

        out= account.output()

        assert(out['WALLET_GAIN']==5.6)
        assert(out['MAX_DROWDOWN']==-25.8) 
        assert(out['STOCK_GAIN']==76.7)
        assert(out['TRADES']==2)
        assert(out['TRADES_POS']==1)
        assert(out['TRADES_NEG']==1) 
        assert(out['COMMISIONS']==115.23)
        

if __name__ == '__main__':
    unittest.main()