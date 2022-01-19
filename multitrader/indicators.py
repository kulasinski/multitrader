import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import pandas as pd

class Indicators():
    
    def __init__(self, data):
        self.data = data
        
    def get(self, i):
        if i=='MA20':
            return self.MA(period=20)
        elif i=='MA20':
            return self.MA(period=20)
        elif i=='RSI':
            return self.RSI(period=14)
        elif i=='SLB':
            return self.SqueezeLazyBear()
        else:
            raise Exception("Unknown indicator: ",i)
            
    def MA(self, period):
        return self.data.Close.rolling(period).mean()

    def RSI(self, period):
        """
            taken from:
            https://stackoverflow.com/questions/57006437/calculate-rsi-indicator-from-pandas-dataframe/57037866
        """
        n = period
        def rma(x, n, y0):
            a = (n-1) / n
            ak = a**np.arange(len(x)-1, -1, -1)
            return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1)]

        change = self.data.Close.diff()
        gain = change.mask(change < 0, 0.0)
        loss = -change.mask(change > 0, -0.0)
        avg_gain = rma(gain[n+1:].to_numpy(), n, np.nansum(gain.to_numpy()[:n+1])/n)
        avg_loss = rma(loss[n+1:].to_numpy(), n, np.nansum(loss.to_numpy()[:n+1])/n)
        RS = avg_gain / avg_loss
        RSI = np.round(100. - 100. / ( 1. + RS ), 0)
        return RSI

    def stdev(self, period):
        return self.data.Close.rolling(period).std()

    def SqueezeLazyBear(self):

        """
            From https://www.tradingview.com/v/4IneGo8h/
            study(shorttitle = "SQZMOM_LB", title="Squeeze Momentum Indicator [LazyBear]", overlay=false)

            warning: squeeze info not passed in yet
        """

        length = 20 #input(20, title="BB Length")
        mult = 2.0 #input(2.0,title="BB MultFactor")
        lengthKC = 20 #input(20, title="KC Length")
        multKC = 1.5 #input(1.5, title="KC MultFactor")

        useTrueRange = True #input(true, title="Use TrueRange (KC)", type=bool)

        # Calculate BB
        source = self.data.Close
        basis = self.MA(length) #sma(source, length)
        dev = multKC * self.stdev(length)
        upperBB = basis + dev
        lowerBB = basis - dev

        # Calculate KC
        ma = self.MA(length) #sma(source, lengthKC)
        rng = self.data.High - self.data.Low # range = useTrueRange ? tr : (high - low)
        rangema = rng.rolling(lengthKC).mean() # sma(range, lengthKC)
        upperKC = ma + rangema * multKC
        lowerKC = ma - rangema * multKC

        sqzOn  = (lowerBB > lowerKC) & (upperBB < upperKC)
        sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
        noSqz  = (~sqzOn) & (~sqzOff)

        highest = self.data.High.rolling(lengthKC).max()
        lowest  = self.data.Low.rolling(lengthKC).min()

        correction = ( (highest+lowest)/2. + ma)/2.

        """
            calculate slope with 0 intercept, on a rolling basis
        """
        X = np.arange(lengthKC).reshape(-1,1)
        y = (source-correction).to_numpy()
        N = len(y)

        val = np.zeros(N)
        for i in range(N):
            try:
                sub_y = y[i+1-lengthKC:i+1].reshape(-1,1)
                lr = LinearRegression(fit_intercept=False)
                lr.fit(X,sub_y)
                val[i] = round(lr.coef_[0][0],3)
            except Exception as e:
                val[i] = np.nan

        return val #pd.DataFrame({'SLB':val})

        # val = linreg(
        #     source  -  correction, # avg(
        #                 # avg(
        #                 #     highest(high, lengthKC), 
        #                 #     lowest(low, lengthKC)
        #                 #     ),
        #                 # sma(close,lengthKC)
        #                 # ), 
        #     lengthKC, # length
        #     0 # offset
        # )
        

        # bcolor = None
        # if val>0:
        #     if val > nz:
        #         bcolor = 'lime' # pos and up
        #     else:
        #         bcolor = 'green' # pos and down
        # else:
        #     if val < nz:
        #         bcolor = 'red' # neg and down
        #     else:
        #         bcolor = 'maroon' # neg and up
        # bcolor = iff( val > 0, 
        #             iff( val > nz(val[1]), lime, green),
        #             iff( val < nz(val[1]), red, maroon))
        # scolor = noSqz ? blue : sqzOn ? black : gray 
        # plot(val, color=bcolor, style=histogram, linewidth=4)
        # plot(0, color=scolor, style=cross, linewidth=2)