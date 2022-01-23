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
        elif i=='STDEV20':
            return self.STDEV(period=20)
        elif i=='RSI':
            return self.RSI(period=14)
        elif i=='SLBval':
            val, inTrend = self.SqueezeLazyBear()
            return val
        elif i=='SLBtrend':
            val, inTrend = self.SqueezeLazyBear()
            return inTrend
        elif i=='ADX':
            return self.ADX()
        elif i=='VWI':
            return self.VWI()
        elif i=='BBu':
            BBu, _ = BollingerBands()
        elif i=='BBl':
            _, BBl = BollingerBands()
        elif i=='KCu':
            KCu, _ = KeltnerChannel()
        elif i=='KCl':
            _, KCl = KeltnerChannel()
        else:
            raise Exception("Unknown indicator: ",i)
            
    def MA(self, period):
        """
            (Rolling) simple moving average
        """
        return self.data.Close.rolling(period).mean()

    def STDEV(self, period):
        """
            (Rolling) standard deviation
        """
        return self.data.Close.rolling(period).std()

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

    def ATR(self, period=14):
        """
            Average True Range (ATR)
            https://www.investopedia.com/terms/a/atr.asp
        """
        return (self.data.High-self.data.Low).rolling(period).mean()

    def ADX(self, period=14):
        """
            Average Directional Movement Index (ADX)
            https://www.investopedia.com/terms/a/adx.asp

        """
        pDM = self.data.High.diff().rolling(period).mean()
        mDM =-self.data.Low.diff().rolling(period).mean()
        ATR = self.ATR(period=period)
        pDI = pDM / ATR
        mDI = mDM / ATR
        DX  = ( (pDI.abs()-mDI.abs()) / (pDI.abs()+mDI.abs()) )*100.
        ADX = DX.rolling(period).mean()
        return ADX

    def VWI(self, period=14):
        """
            Volume-Weighted Index
        """
        VDI = (self.data.Volume * (self.data.Close - self.data.Open)).rolling(period).mean()
        ATR = self.ATR()
        V   = self.data.Volume.rolling(period).mean()
        VWI = 100. * VDI / ATR / V
        return VWI.rolling(period).mean()

    def BollingerBands(self, length = 20, mult = 2.0):
        """
            https://www.investopedia.com/terms/b/bollingerbands.asp
            BOLU=MA(TP,n)+m∗σ[TP,n]
            BOLD=MA(TP,n)−m∗σ[TP,n]
            where:
            BOLU=Upper Bollinger Band
            BOLD=Lower Bollinger Band
            MA=Moving average
            TP (typical price)=(High+Low+Close)÷3
            n=Number of days in smoothing period (typically 20)
            m=Number of standard deviations (typically 2)
            σ[TP,n]=Standard Deviation over last n periods of TP
        """
        basis = self.MA(length)
        dev = mult * self.STDEV(length) # ??? czy 1.5 jak KC?
        upperBB = basis + dev
        lowerBB = basis - dev

        return upperBB, lowerBB

    def KeltnerChannel(self, lengthKC = 20, mult = 1.5, useExp = True): # was 1.5
        """
            https://www.investopedia.com/terms/k/keltnerchannel.asp
            Keltner Channel Middle Line=EMA
            Keltner Channel Upper Band=EMA+2∗ATR
            Keltner Channel Lower Band=EMA−2∗ATR
            where:
            EMA=Exponential moving average (typically over 20 periods)
            ATR=Average True Range (typically over 10 or 20 periods)
        """
        if useExp:
            ma = self.data.Close.ewm(span=20).mean()
        else:
            ma = self.MA(lengthKC)
        rng = self.data.High - self.data.Low
        rangema = rng.rolling(lengthKC).mean()
        upperKC = ma + rangema * mult
        lowerKC = ma - rangema * mult

        return upperKC, lowerKC

    def SqueezeLazyBear(self, length = 20, multBB = 2.0, multKC = 1.5):
        """
            https://atas.net/atas-possibilities/squeeze-momentum-indicator/
            Squeeze Momentum shows periods when volatility increases or decreases, 
            in other words, when the market goes from the trend into flat movement and vice versa.
        """

        # Calculate Bollinger Bands
        upperBB, lowerBB = self.BollingerBands(length = 20, mult = multBB)

        # Calculate Keltner Channel
        upperKC, lowerKC = self.KeltnerChannel(lengthKC = length, mult = multKC)

        # Are BB inside KC?
        sqzOn  = (lowerBB > lowerKC) & (upperBB < upperKC)
        sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
        noSqz  = (~sqzOn) & (~sqzOff)
        inTrend = sqzOff.apply(lambda x: 1 if x else 0).astype(int)

        """
            calculate slope with 0 intercept, on a rolling basis
        """

        highest = self.data.High.rolling(length).max()
        lowest  = self.data.Low.rolling(length).min()
        correction = ( (highest+lowest)/2. + self.MA(length) )/2.

        X = np.arange(length).reshape(-1,1)
        y = (self.data.Close-correction).to_numpy()
        N = len(y)

        val = np.zeros(N)
        for i in range(N):
            try:
                sub_y = y[i+1-length:i+1].reshape(-1,1)
                lr = LinearRegression(fit_intercept=False)
                lr.fit(X,sub_y)
                val[i] = round(lr.coef_[0][0],3)
            except Exception as e:
                val[i] = np.nan

        return val, inTrend