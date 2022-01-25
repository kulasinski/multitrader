# Multitrader

Backtesting engine for multiple tickers at once. Selects ticker(s) and run your hand-picked strategy in a backtesting manner with a minimum amount of code!

## Quick start

### Parameters setup

You can use `run.py` as an example.

First, set tickers names and make sure you downloaded the .csv files in a Yahoo format (Date, Close, Open, Low, High, Volume - as headers):

```python
tickers = [
    'AAPL',
    'MSFT',
]
```

Set up initial cash:

```python
INITIAL_CASH = 1000
```

Optionally, set up for how much to buy each ticker. 0 - proportionally, 1 - for full amount of cash, 0.5 - for half, etc.

```python
CASH_PER_TRADE = 0. # proportionally
```

Define the date range:

```python
START_DATE = '2020-11-01'
END_DATE = '2021-12-31'
```

Create you own strategy or use one of the predefined:

```python
STRATEGY = SLBStrat()
```

Define the directory containing your data, as well as SP500 benchmark.

```python
STOCKS_DIR = 'stocks'
SP500_PATH = STOCKS_DIR+'/SP500.csv'
```

Optionally, define plot output file.

```python
PLOT_OUTPUT = 'plot.png'
```

### Run the simulation

Set up the account and its parameters

```python
account = Account()
account.set_cash(INITIAL_CASH)
account.set_cash_per_trade(CASH_PER_TRADE)
account.set_commission(Commission())
account.set_strategy(STRATEGY)
```

Add benchmark (optionally) and data:

```python
account.set_SP500(SP500_PATH)
for ticker in tickers:
    account.add_data(ticker, load_ticker(ticker, default_dir=STOCKS_DIR))
```

Run the simulation

```python
account.run(
        start= START_DATE,
        end  = END_DATE,
    )
```

Show the trading summary:

```python
account.summary()
```

`=== SUMMARY ===`

    `WALLET: $10000.0 -> $11008.862833000001 (+$1008.86) +10.1%`
    
    `MAX DROWDOWN: -4.4%`
    
    `AVG FILLRATE: 16.0%`
    
    `BENCHMARK: SP500: 44.0% | AVG STOCK 64.7%`
    
    `DURATION: 294 trading days`
    
    `TRADES: POS 5 + NEG 0 + ERR 0 = TOT 5`
    
    `AVG TRADE: 5.9%`
    
    `BEST TRADE:  $283.52 -> $302.75 ($19.23) 6.8% | MSFT 2021-09-28 -> 2021-10-14`
    
    `WORST TRADE:  $126.0 -> $130.36 ($4.36) 3.5% | AAPL 2021-02-22 -> 2021-04-08`
    
    `COMMISSION TOTAL: $120.31`
    
Finally, plot the timeseries and save the key output variables to file:

```python
account.plot(PLOT_OUTPUT)

account.output(fname='log.csv')
```

![alt text](https://github.com/kulasinski/multitrader/blob/main/plot.png?raw=true)
