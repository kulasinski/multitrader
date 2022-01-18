import numpy as np

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