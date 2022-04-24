# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 21:08:49 2022

@author: banik
"""

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
import pandas_datareader.data as web
import matplotlib.pyplot as plt

fred = Fred(api_key='Put your own API Key')

tic = 'V' #input any ticker from any industry

#-----------------------------------------------------------------------------#
# Stock valuation using Dividend Discount Model
#-----------------------------------------------------------------------------#

def intrinsic_value(terminalgrowth,T):
    data = yf.Ticker(tic)
    company = data.info['shortName']
    industry = data.info['industry']
    exchange = data.info['exchange']
    symb = data.info['symbol']
    website = data.info['website']
    print('\n')
    print(f'''Basic Introduction:
          Company Name: {company}
          Industry: {industry}
          Exchange: {exchange}
          Stock Symbol: {symb}
          Company Website: {website}''')
    print('\n')
    details = data.info['longBusinessSummary']
    print(f'''Company Description: 
          {details}''')
    print('\n')
    currentprice = data.history(period='5y',interval='1d').Close[-1]
    NI = data.financials.loc['Net Income'][0]
    sales=data.financials.loc['Total Revenue'][0]
    rev = data.info['revenueGrowth']
    gp = data.financials.loc['Gross Profit'][0]
    gm = gp/sales
    ebitda = data.financials.loc['Operating Income'][0]
    om = ebitda/sales
    pm = NI/data.info['totalRevenue']
    ROE = data.financials.loc['Net Income'].mean()/data.balancesheet.loc['Total Stockholder Equity'][0]
    ROA = data.financials.loc['Net Income'].mean()/data.balancesheet.loc['Total Assets'][0]
    print(f'''Profitability of {company}:
          Revenue Growth: {round(rev*100,2)}%
          Gross Profit: ${round(gp/1000000000,3)}bn
          Net Profit: ${round(NI/1000000000,3)}bn
          Operating Profit: ${round(ebitda/1000000000,3)}bn
          Gross Profit Margin: {round(gm*100,2)}%
          Operating Profit Margin: {round(om*100,2)}%
          Net Profit Margin: {round(pm*100,2)}%
          Return on Equity: {round(ROE*100,2)}%
          Return on Assets: {round(ROA*100,2)}%
          ''')
    print('\n')
    cr = data.info['currentRatio']
    qr = data.info['quickRatio']
    print(f'''Liquidity of {company}:
          Current Ratio: {cr}
          Quick Ratio: {qr}
          ''')
    print('\n')
    DE = data.info['debtToEquity']
    EM = data.info['bookValue']*data.info[
        'sharesOutstanding']/(data.info['totalDebt']+data.info['bookValue']*data.info['sharesOutstanding'])
    print(f'''Solvency of {company}:
          Debt-to-Equity: {round(DE,2)}x
          Equity Multiplier: {round(EM,2)}x
          ''')
    print('\n')
    PB = data.info['priceToBook']
    PE = data.info['trailingPE']
    PS = data.info['priceToSalesTrailing12Months']
    print(f'''Price multipliers of {company}:
          Price-to-Book: {round(PB,2)}x
          Price-to-Earnings: {round(PE,2)}x
          Price-to-Sales: {round(PS,2)}x
          ''')
    print('\n')
    div0 = data.info['trailingAnnualDividendRate']
    print(f'Current dividend of {company}: ${round(div0,3)}')
    ret_ratio = 1 - (data.info['sharesOutstanding']*div0/NI)
    g = ROE * ret_ratio
    print(f'Sustainable growth rate of {company} stock: {round(g*100,2)}%')
    rf = fred.get_series('DTB6')[-1]/100
    rm= data.info['SandP52WeekChange']
    b = data.info['beta']
    r = rf+b*(rm-rf)
    t=1
    sum = 0
    yf_data = []
    for i in range(T):
        div=float(div0)*(1+g)**(i+t)
        iv=div/(1+r)**(i+t)
        sum = sum+iv
        print (f'Expected dividend in 202{i+2}: ${round(div,2)}')
        yf_data.append([div])
    df = pd.DataFrame(yf_data,columns={'Dividends'})
    df = df.transpose()
    df.columns = ['2022E','2023E','2024E','2025E','2026E']
    df['2021A'] = div0
    df = df[['2021A','2022E','2023E','2024E','2025E','2026E']]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(df.columns[0],df['2021A'],width=0.7,color = 'lightcoral')
    ax.bar(df.columns[1],df['2022E'],width=0.7,color = 'dodgerblue')
    ax.bar(df.columns[2],df['2023E'],width=0.7,color = 'dodgerblue')
    ax.bar(df.columns[3],df['2024E'],width=0.7,color = 'dodgerblue')
    ax.bar(df.columns[4],df['2025E'],width=0.7,color = 'dodgerblue')
    ax.bar(df.columns[5],df['2026E'],width=0.7,color = 'dodgerblue')
    ax.set_title('Actual and Forecasted Dividends of Ford Motor Company')
    plt.show()
    print(f'The market consensus for the terminal growth of {company}: {round(terminalgrowth*100,2)}%.')
    terminalvalue = (div*(1+terminalgrowth)/(r-terminalgrowth))/(1+r)**T
    intrinsvalue = float(np.array(sum+terminalvalue))
    print(f'Risk-free rate (10-yr T-bond rate): {round(rf,2)*100}%')
    print(f'Stock beta: {round(b,2)}')
    print(f'Market risk premium: {round(rm-rf,2)*100}%')
    print(f'Required rate of return for the stock: {round(r,2)*100}%')
    print(f'Intrinsic value of {company}: ${round(intrinsvalue,2)}')
    print(f'Current price of {company}: ${round(currentprice,2)}')
    print('\x1b[47m'+'Investment Decision:'+'\x1b[47m')
    print(f"""          BUY: More than 5% upside potential
          HOLD: Within 5% risk of loss or gain potential
          SELL: More than 5% downside risk""")
    print('\n')
    if intrinsvalue<currentprice*(1-0.05):
        
        print('\x1b[41m'+f'Downside risk of loss: {round((1-intrinsvalue/currentprice)*100,2)}%'+'\x1b[41m')
        print('\x1b[41m'+f'{company} is overvalued!!! So, i give SELL recommendation for the stock.'+'\x1b[41m')
    elif intrinsvalue==currentprice*(1+-0.05):
        print(f'{company} is fairly valued')
    else:
        
        print('\x1b[42m'+f'Upside potential: {round((intrinsvalue/currentprice-1)*100,2)}%'+'\x1b[42m')
        print('\x1b[42m'+f'{company} is undervalued!!! So, i give BUY recommendation for the stock.'+'\x1b[42m')
intrinsic_value(0.0165,5)
