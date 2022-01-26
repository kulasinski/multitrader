class Order():
    """
        Buy/Sell orders with market or limit mode
    """
    
    def __init__(self, 
            ticker=None,
            shares=None,
            limit=None,
            quality=None,
            on_create_date=None,
            on_create_price=None,
            valid_until=None,
            memo=None,
            verbose=False,
            ):

        self.ticker = ticker
        self.shares = shares # amount of shares to buy/sell, negative if sell
        self.limit = limit # limit price, None in market orders
        self.quality = quality
        self.on_create_price = on_create_price
        self.on_create_date = on_create_date
        self.valid_until = valid_until
        self.memo = memo  
        self.verbose = verbose

        self.is_market = (self.limit is None) #True if limit is None else False
        self.is_limit = not self.is_market
        self.is_valid = True 
        self.executed_price = None
        self.executed_date = None

        self.is_buy = (self.shares > 0)
        self.is_sell= (self.shares < 0)
        if self.shares == 0:
            raise Exception("ORDER: Cannot place an order with 0 shares!")

        self.log(f"[{on_create_date}] creating {'market' if self.is_market else 'limit'} {'BUY' if self.is_buy else 'SELL'} ({self.shares}) order at ${self.on_create_price if self.is_market else self.limit} ({self.ticker})")

            
    def log(self, msg):
        if self.verbose:
            print("ORDER: "+msg)

    def execute(self, price, date):
        self.executed_price = round(price,2)
        self.executed_date  = date
        self.is_valid = False
        self.log(f"[{date}] executing {'market' if self.is_market else 'limit'} {'BUY' if self.is_buy else 'SELL'} ({self.shares}) order at ${self.executed_price} ({self.ticker})")

    def check_validity(self, date):
        if self.is_valid and self.valid_until is not None:
            if self.valid_until < date:
                self.is_valid = False

    def is_pending(self):
        return True if self.executed_price==None else False


class Trade():
    """
        Porte-manteau for buy-sell pairs
    """

    def __init__(self, 
            open_order,
            close_order=None,
            verbose=False,
            ):

        self.open_order = open_order
        self.close_order = close_order
        self.is_open = True # deal moze miec close_order, ktory sie nie wykonal
        self.gain = None
        self.gain_pct = None
        self.verbose = verbose

        self.ticker = open_order.ticker

    def try_close(self):
        self.log(f'trying to close the trade ({self.ticker})...')
        if self.close_order is not None:
            if self.close_order.executed_date is not None and self.open_order.executed_date is not None:
                self.is_open = False
                self.gain = round(self.close_order.executed_price - self.open_order.executed_price, 2)
                self.gain_pct = round(self.gain / self.open_order.executed_price * 100. , 1)
                self.log(f"gain on position: ${self.open_order.executed_price} -> ${self.close_order.executed_price} (total ${-round(self.close_order.shares*self.gain,2)}) {self.gain_pct}%")
            else:
                self.log('warning: sell order is not yet executed!')
        else:
            self.log('warning: sell order does not exist!')

    def log(self, msg):
        if self.verbose:
            print("TRADE: "+msg)

    def is_pos(self):
        if self.gain is None:
            return None
        return self.gain>0