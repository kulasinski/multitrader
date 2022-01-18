   
class Indicators():
    
    def __init__(self, data):
        self.data = data
        
    def get(self, i):
        if i=='MA20':
            return self.MA(period=20)
        elif i=='MA20':
            return self.MA(period=20)
        else:
            raise Exception("Unknown indicator: ",i)
            
    def MA(self, period):
        return self.data.Close.rolling(period).mean()