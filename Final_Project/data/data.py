# install the package by: pip install yfinance
import yfinance as yf
from datetime import date
import os


def generate_data(name, day, stock_period):
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    outdir = './' + day + '-' + name + '/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    ticker = yf.Ticker(name)

    hist = ticker.history(period=stock_period)
    hist.to_pickle(outdir + 'stock-' + day + '.pkl')
    hist.to_csv(outdir + 'stock-' + day + '.csv')

    for date in ticker.options:
        opt = ticker.option_chain(date=date)
        opt.calls.to_pickle(outdir + 'callop-' + date + '.pkl')
        opt.calls.to_csv(outdir + 'callop-' + date + '.csv')


if __name__ == '__main__':
    name = 'TSLA'
    # today = date.today().strftime("%Y-%m-%d")
    today = '2019-12-17'
    # 1mo, 3mo, 1y, 3y, max
    stock_period = '3y'

    generate_data(name=name, day=today, stock_period=stock_period)
    print('Finish!')
