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
from dateutil.relativedelta import relativedelta
from pandas_datareader import data
import datetime
from datetime import date
from yahoofinancials import YahooFinancials

fred = Fred(api_key='0bbc318d3ba2efaf9d4e56708954067d')
d1 = date.today()
d0 = d1 - relativedelta(years=5)

all = ('ATVI', 'ALL','BAC','CBRE','CVX','CVS','DG','LLY','HSY','HUN',
       'INTC','LULU','MSFT','NOC','PXD','PLD','PAYX','TMUS','TRTN','WMT','WRK','UNH')

# Read wikipedia to get S&P 500 info
source = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

# Sort S&P 500 info to get only tickers
df = pd.DataFrame(source[0])
df = df.transpose()
df = df.iloc[0:4]
df = df.transpose()
df = df.sort_values('Symbol').reset_index()
df = df.drop(['index'], axis=1)

# Read specific tickers from all tickers
specific_asset = df[df['Symbol'].isin(all)]

#-----------------------------------------------------------------------------#
#Get all relative valuation techniques for healthcare stocks
#-----------------------------------------------------------------------------#
# Read only health care stocks
healthcare_industry = df[df['GICS Sector']=='Health Care'].Symbol.values.tolist()
# read selected stocks from healthcare sectors
health_care = specific_asset[specific_asset['GICS Sector']=='Health Care']
health_care = health_care.set_index('Security').drop(['SEC filings'],axis=1)
health_care = health_care.Symbol.values.tolist()

health_insurance = ('UNH','ELV','CNC','HUM','CVS','MOH','CI','CAH','CLOV')

# Estimate multiples of healthcare stocks and the industry average
yf_data = []
for tic in healthcare_industry:
    ps = yf.Ticker(tic).info.get('priceToSalesTrailing12Months',{})
    pb = yf.Ticker(tic).info.get('priceToBook',{})
    pe = yf.Ticker(tic).info.get('forwardPE',{})
    evebitda = yf.Ticker(tic).info.get('enterpriseToEbitda',{})
    yf_data.append([tic,ps,pb,pe,evebitda])

industry = pd.DataFrame(yf_data, columns=['Symbol',
                    'Price-to-Sales (x)','Price-to-Book (x)','Price-to-Earnings (x)','EV-to-EBITDA']).replace(np.nan,0).mean()
industry = pd.DataFrame(industry)
industry.columns = ['Industry']

health_data = []
for hc in health_insurance:
    ps = yf.Ticker(hc).info.get('priceToSalesTrailing12Months',{})
    pb = yf.Ticker(hc).info.get('priceToBook',{})
    pe = yf.Ticker(hc).info.get('forwardPE',{})
    evebitda = yf.Ticker(hc).info.get('enterpriseToEbitda',{})
    health_data.append([hc,ps,pb,pe,evebitda])
    multipliers = pd.DataFrame(health_data, columns=['Symbol',
                    'Price-to-Sales (x)','Price-to-Book (x)','Price-to-Earnings (x)','EV-to-EBITDA']).replace(np.nan,0)
    multipliers = pd.DataFrame(multipliers).set_index('Symbol').transpose()
    # multipliers['Industry'] = industry
multipliers.transpose().plot.bar()

#-----------------------------------------------------------------------------#
#Ratio Analysis and Stock-specific information of top healthcare stocks
#-----------------------------------------------------------------------------#
top_healthcare = ('UNH','CVS','ELV','HCA','SGFY','WBA','CI','WMT','KR','RAD','ACI','CNC','LLY',)
# Ratio Analysis
hc_ratio = []
# for th in top_healthcare:
for th in health_insurance:
    RPS = yf.Ticker(th).info.get('revenuePerShare',{})
    TCPS = yf.Ticker(th).info.get('totalCashPerShare',{})
    EPS = yf.Ticker(th).info.get('forwardEps',{})
    GPM = float(yf.Ticker(th).info.get('grossMargins',{}))*100
    OPM = float(yf.Ticker(th).info.get('operatingMargins',{}))*100
    NPM = float(yf.Ticker(th).info.get('profitMargins',{}))*100
    CR = yf.Ticker(th).info.get('currentRatio',{})
    QR = yf.Ticker(th).info.get('quickRatio',{})
    # ROE = float(yf.Ticker(th).financials.loc['Net Income'][0]/yf.Ticker(th).balancesheet.loc['Total Stockholder Equity'][0])*100
    # ROA = float(yf.Ticker(th).info.get('returnOnAssets',{}))*100
    # payout = float(yf.Ticker(th).info.get('payoutRatio',{}))*100
    #DY = yf.Ticker(th).info.get('dividendRate',{})*100/float(yf.Ticker(th).history('5y').Close.iloc[-1])
    DE = yf.Ticker(th).info.get('debtToEquity',{})
    hc_ratio.append([th,RPS,TCPS,EPS,GPM,OPM,NPM,CR,QR,DE])
th_ratio = round(pd.DataFrame(hc_ratio, columns=['Stock','Revenue Per Share (USD)','Total Cash per share (USD)',
                                         'Earnings Per Share (USD)','Gross Profit Margin (%)',
                                         'Operating Profit Margin (%)','Net Profit Margin (%)',
                                         'Current Ratio (x)','Quick Ratio (x)','Debt-to-Equity (x)']).set_index('Stock').transpose(),2)
print(f'''Ratio Analysis of Top Healthcare Equities:
{th_ratio}''')

#Stock-specific information
hc_market = []
# for th in top_healthcare:
for th in health_insurance:
    O = "${:,.2f}".format(yf.Ticker(th).info.get('regularMarketOpen',{}))
    C = "${:,.2f}".format(float(yf.Ticker(th).history('5y').Close.iloc[-1]))
    H = "${:,.2f}".format(yf.Ticker(th).info.get('regularMarketDayHigh',{}))
    L = "${:,.2f}".format(yf.Ticker(th).info.get('regularMarketDayLow',{}))
    V = yf.Ticker(th).info.get('volume',{})/1000000
    CB = yf.Ticker(th).info.get('beta',{})
    SR = yf.Ticker(th).info.get('shortRatio',{})
    FS = float(yf.Ticker(th).info.get('floatShares',{}))/1000000
    MA15 = "${:,.2f}".format(yf.Ticker(th).history('5y').Close.rolling(15).mean().iloc[-1])
    MA30 = "${:,.2f}".format(yf.Ticker(th).history('5y').Close.rolling(30).mean().iloc[-1])
    MA90 = "${:,.2f}".format(yf.Ticker(th).history('5y').Close.rolling(90).mean().iloc[-1])
    MA180 = "${:,.2f}".format(yf.Ticker(th).history('5y').Close.rolling(180).mean().iloc[-1])
    MA360 = "${:,.2f}".format(yf.Ticker(th).history('5y').Close.rolling(360).mean().iloc[-1])
    hc_market.append([th,O,C,H,L,V,CB,SR,FS,MA15,MA30,MA90,MA180,MA360])
th_market = round(pd.DataFrame(hc_market, columns=['Stock','Open','Close','High','Low','Volume (mn)','beta',
                                             'Short Ratio (%)','Float Share (mn)','15-Day-MA',
                                             '30-Day-MA','90-Day-MA','180-Day-MA','52-Weeks-MA']).set_index('Stock').transpose(),2)
print(f'''Stock-specific information of Top Healthcare Equities:
{th_market}''')

with pd.ExcelWriter('Healthcare Industry Analytics.xlsx') as mydata:
    th_ratio.to_excel(mydata, sheet_name='Ratio Analysis',index=True)
    th_market.to_excel(mydata, sheet_name='Stock-specific Info',index=True)

#-----------------------------------------------------------------------------#
#Get all relative valuation techniques for real estate stocks
#-----------------------------------------------------------------------------#
# Read only health care stocks
realestate_industry = df[df['GICS Sector']=='Real Estate'].Symbol.values.tolist()
# read selected stocks from healthcare sectors
real_estate = specific_asset[specific_asset['GICS Sector']=='Real Estate']
real_estate = real_estate.set_index('Security').drop(['SEC filings'],axis=1)
real_estate = real_estate.Symbol.values.tolist()
top_realestate = ('AVB','PLD','CCI','PSA','O','SPG','SBAC','WELL','DLR','VICI','CBRE')

# Estimate multiples of healthcare stocks and the industry average
yf_data = []
for tic in realestate_industry:
    ps = yf.Ticker(tic).info.get('priceToSalesTrailing12Months',{})
    pb = yf.Ticker(tic).info.get('priceToBook',{})
    pe = yf.Ticker(tic).info.get('forwardPE',{})
    evebitda = yf.Ticker(tic).info.get('enterpriseToEbitda',{})
    yf_data.append([tic,ps,pb,pe,evebitda])

industry = pd.DataFrame(yf_data, columns=['Symbol',
                    'Price-to-Sales (x)','Price-to-Book (x)','Price-to-Earnings (x)','EV-to-EBITDA']).replace(np.nan,0).mean()
industry = pd.DataFrame(industry)
industry.columns = ['Industry']

realestate_data = []
for re in top_realestate:
    ps = yf.Ticker(re).info.get('priceToSalesTrailing12Months',{})
    pb = yf.Ticker(re).info.get('priceToBook',{})
    pe = yf.Ticker(re).info.get('forwardPE',{})
    evebitda = yf.Ticker(re).info.get('enterpriseToEbitda',{})
    realestate_data.append([re,ps,pb,pe,evebitda])
    multipliers = pd.DataFrame(realestate_data, columns=['Symbol',
                    'Price-to-Sales (x)','Price-to-Book (x)','Price-to-Earnings (x)','EV-to-EBITDA']).replace(np.nan,0)
    multipliers = pd.DataFrame(multipliers).set_index('Symbol').transpose()
    multipliers['Industry'] = industry
multipliers.transpose().plot.bar()
# sector = ['CBRE','PLD','CVS','LLY']

#-----------------------------------------------------------------------------#
#Ratio Analysis and Stock-specific information of top real estate stocks
#-----------------------------------------------------------------------------#

# Ratio Analysis
re_ratio = []
for tr in top_realestate:
    RPS = yf.Ticker(tr).info.get('revenuePerShare',{})
    TCPS = yf.Ticker(tr).info.get('totalCashPerShare',{})
    EPS = yf.Ticker(tr).info.get('forwardEps',{})
    GPM = float(yf.Ticker(tr).info.get('grossMargins',np.nan))*100
    OPM = float(yf.Ticker(tr).info.get('operatingMargins',np.nan))*100
    NPM = float(yf.Ticker(tr).info.get('profitMargins',np.nan))*100
    CR = yf.Ticker(tr).info.get('currentRatio',np.nan)
    QR = yf.Ticker(tr).info.get('quickRatio',np.nan)
    ROE = yf.Ticker(tr).info.get('returnOnEquity',{})
    ROA = yf.Ticker(tr).info.get('returnOnAssets',{})
    # payout = float(yf.Ticker(tr).info.get('payoutRatio',{}))*100
    #DY = yf.Ticker(th).info.get('dividendRate',{})*100/float(yf.Ticker(th).history('5y').Close.iloc[-1])
    DE = yf.Ticker(tr).info.get('debtToEquity',{})
    re_ratio.append([tr,RPS,TCPS,EPS,GPM,OPM,NPM,CR,QR,ROE,ROA,DE])
tr_ratio = round(pd.DataFrame(re_ratio, columns=['Stock','Revenue Per Share (USD)','Total Cash per share (USD)',
                                         'Earnings Per Share (USD)','Gross Profit Margin (%)',
                                         'Operating Profit Margin (%)','Net Profit Margin (%)',
                                         'Current Ratio (x)','Quick Ratio (x)','Return on Equity','Return on Assets','Debt-to-Equity (x)']).set_index('Stock').transpose(),2)
print(f'''Ratio Analysis of Top Real Estate Equities:
{tr_ratio}''')

#Stock-specific information
re_market = []
for tr in top_realestate:
    O = "${:,.2f}".format(yf.Ticker(tr).info.get('regularMarketOpen',{}))
    C = "${:,.2f}".format(float(yf.Ticker(tr).history('5y').Close.iloc[-1]))
    H = "${:,.2f}".format(yf.Ticker(tr).info.get('regularMarketDayHigh',{}))
    L = "${:,.2f}".format(yf.Ticker(tr).info.get('dayLow',{}))
    V = yf.Ticker(tr).info.get('volume',{})/1000000
    CB = yf.Ticker(tr).info.get('beta',{})
    SR = yf.Ticker(tr).info.get('shortRatio',{})
    MA15 = "${:,.2f}".format(yf.Ticker(tr).history('5y').Close.rolling(15).mean().iloc[-1])
    MA30 = "${:,.2f}".format(yf.Ticker(tr).history('5y').Close.rolling(30).mean().iloc[-1])
    MA90 = "${:,.2f}".format(yf.Ticker(tr).history('5y').Close.rolling(90).mean().iloc[-1])
    MA180 = "${:,.2f}".format(yf.Ticker(tr).history('5y').Close.rolling(180).mean().iloc[-1])
    MA360 = "${:,.2f}".format(yf.Ticker(tr).history('5y').Close.rolling(360).mean().iloc[-1])
    re_market.append([tr,O,C,H,L,V,CB,SR,MA15,MA30,MA90,MA180,MA360])
tr_market = round(pd.DataFrame(re_market, columns=['Stock','Open','Close','High','Low','Volume (mn)','beta',
                                             'Short Ratio (%)','15-Day-MA',
                                             '30-Day-MA','90-Day-MA','180-Day-MA','52-Weeks-MA']).set_index('Stock').transpose(),2)
print(f'''Stock-specific information of Top Real Estate Equities:
{tr_market}''')

with pd.ExcelWriter('Real Estate Industry Analytics.xlsx') as redata:
    tr_ratio.to_excel(redata, sheet_name='Ratio Analysis',index=True)
    tr_market.to_excel(redata, sheet_name='Stock-specific Info',index=True)

#-----------------------------------------------------------------------------#
# Stock valuation using Dividend Discount Model
#-----------------------------------------------------------------------------#
#def intrinsic_value(terminalgrowth,T,sector,n,g_noips,tax_rate,avg_corp_loan_rate):
def intrinsic_value(terminalgrowth,T,sector,n,tax_rate,avg_corp_loan_rate):
    
    # Get a short introduction of the company
    data = yf.Ticker(sector[n])
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
    
    # Get current income statement data
    NI = data.financials.loc['Net Income'][0]/1000000
    sales=data.financials.loc['Total Revenue'][0]/1000000
    ebitda = data.financials.loc['Operating Income'][0]/1000000
    div0 = data.info['trailingAnnualDividendRate']
    #print(f'Current dividend of {company}: ${round(div0,3)}')
    
    # Determine growth rate of sales, operating profit margin and net profit margin
    g_sales = data.info['revenueGrowth']
    print(f'Revenue growth rate in 2021: {g_sales}')
    g_sales1 = float(input(f'Revenue growth rate in 2022:'))
    g_sales2 = float(input(f'Revenue growth rate in 2023:'))
    g_sales3 = float(input(f'Revenue growth rate in 2024:'))
    g_sales4 = float(input(f'Revenue growth rate in 2025:'))
    g_sales5 = float(input(f'Revenue growth rate in 2026:'))
    g_sales6 = float(input(f'Revenue growth rate in 2027:'))
    g_sales7 = float(input(f'Revenue growth rate in 2028:'))
    g_sales8 = float(input(f'Revenue growth rate in 2029:'))
    g_sales9 = float(input(f'Revenue growth rate in 2030:'))
    g_sales10 = float(input(f'Revenue growth rate in 2031:'))
    sales1 = sales*(1+g_sales1)
    sales2 = sales1*(1+g_sales2)
    sales3 = sales2*(1+g_sales3)
    sales4 = sales3*(1+g_sales4)
    sales5 = sales4*(1+g_sales5)
    sales6 = sales5*(1+g_sales6)
    sales7 = sales6*(1+g_sales7)
    sales8 = sales7*(1+g_sales8)
    sales9 = sales8*(1+g_sales9)
    sales10 = sales9*(1+g_sales10)
        
    OPM = ebitda/sales
    print(f'Operating profit margin in 2021: {OPM}')
    OPM1 = float(input(f'Operating profit margin in 2022:'))
    OPM2 = float(input(f'Operating profit margin in 2023:'))
    OPM3 = float(input(f'Operating profit margin in 2024:'))
    OPM4 = float(input(f'Operating profit margin in 2025:'))
    OPM5 = float(input(f'Operating profit margin in 2026:'))
    OPM6 = float(input(f'Operating profit margin in 2027:'))
    OPM7 = float(input(f'Operating profit margin in 2028:'))
    OPM8 = float(input(f'Operating profit margin in 2029:'))
    OPM9 = float(input(f'Operating profit margin in 2030:'))
    OPM10 = float(input(f'Operating profit margin in 2031:'))
        
    NPM = NI/sales
    print(f'Net profit margin in 2021: {NPM}')
    NPM1 = float(input(f'Net profit margin in 2022:'))
    NPM2 = float(input(f'Net profit margin in 2023:'))
    NPM3 = float(input(f'Net profit margin in 2024:'))
    NPM4 = float(input(f'Net profit margin in 2025:'))
    NPM5 = float(input(f'Net profit margin in 2026:'))
    NPM6 = float(input(f'Net profit margin in 2027:'))
    NPM7 = float(input(f'Net profit margin in 2028:'))
    NPM8 = float(input(f'Net profit margin in 2029:'))
    NPM9 = float(input(f'Net profit margin in 2030:'))
    NPM10 = float(input(f'Net profit margin in 2031:'))
    
    payout = (data.info['sharesOutstanding']*div0/NI)/1000000
    print(f'Divident payout ratio in 2021: {payout}')
    payout1 = float(input(f'Divident payout ratio in 2022:'))
    payout2 = float(input(f'Divident payout ratio in 2023:'))
    payout3 = float(input(f'Divident payout ratio in 2024:'))
    payout4 = float(input(f'Divident payout ratio in 2025:'))
    payout5 = float(input(f'Divident payout ratio in 2026:'))
    payout6 = float(input(f'Divident payout ratio in 2027:'))
    payout7 = float(input(f'Divident payout ratio in 2028:'))
    payout8 = float(input(f'Divident payout ratio in 2029:'))
    payout9 = float(input(f'Divident payout ratio in 2030:'))
    payout10 = float(input(f'Divident payout ratio in 2031:'))
    
    # Develop a Forecasted Financial Summary
    forecast = pd.DataFrame([[sales1,sales2,sales3,sales4,sales5,sales6,sales7,sales8,sales9,sales10]])
    forecast.columns = ['2022E','2023E','2024E','2025E','2026E','2027E','2028E','2029E','2030E','2031E']
    forecast['2021A'] = sales
    forecast.loc[len(forecast.index)] = [OPM1*sales1,OPM2*sales2,OPM3*sales3,OPM4*sales4,OPM5*sales5,
                                         OPM6*sales6,OPM7*sales7,OPM8*sales8,OPM9*sales9,OPM10*sales10,ebitda]
    forecast.loc[len(forecast.index)] = [OPM1*100,OPM2*100,OPM3*100,OPM4*100,OPM5*100,
                                         OPM6*100,OPM7*100,OPM8*100,OPM9*100,OPM10*100,OPM*100]
    forecast.loc[len(forecast.index)] = [NPM1*sales1,NPM2*sales2,NPM3*sales3,NPM4*sales4,NPM5*sales5,
                                         NPM6*sales6,NPM7*sales7,NPM8*sales8,NPM9*sales9,NPM10*sales10,NI]
    forecast.loc[len(forecast.index)] = [NPM1*100,NPM2*100,NPM3*100,NPM4*100,NPM5*100,
                                         NPM6*100,NPM7*100,NPM8*100,NPM9*100,NPM10*100,NPM*100]
    forecast.loc[len(forecast.index)] = [NPM1*sales1*payout1/(data.info['sharesOutstanding']/1000000),
                                         NPM2*sales2*payout2/(data.info['sharesOutstanding']/1000000),
                                         NPM3*sales3*payout3/(data.info['sharesOutstanding']/1000000),
                                         NPM4*sales4*payout4/(data.info['sharesOutstanding']/1000000),
                                         NPM5*sales5*payout5/(data.info['sharesOutstanding']/1000000),
                                         NPM6*sales6*payout6/(data.info['sharesOutstanding']/1000000),
                                         NPM7*sales7*payout7/(data.info['sharesOutstanding']/1000000),
                                         NPM8*sales8*payout8/(data.info['sharesOutstanding']/1000000),
                                         NPM9*sales9*payout9/(data.info['sharesOutstanding']/1000000),
                                         NPM10*sales10*payout10/(data.info['sharesOutstanding']/1000000),div0]
    forecast.loc[len(forecast.index)] = [payout1,payout2,payout3,payout4,payout5,
                                         payout6,payout7,payout8,payout9,payout10,payout]
    forecast.index = ['Revenue','Operating Income','Operating Profit Margin (%)','Net Income',
                      'Net Profit Margin (%)','Dividend per share (USD)','Dividend payout (%)']
    forecast.index.name = 'Particulars (in millions USD)'
    forecast = forecast[['2021A','2022E','2023E','2024E','2025E','2026E','2027E','2028E','2029E','2030E','2031E']]
    print(round(forecast,2))
    print(forecast.to_excel('Forecast.xlsx'))
    
    #Ratio Analysis
    gp = data.financials.loc['Gross Profit'][0]
    gm = gp/sales
    ROE = data.financials.loc['Net Income'].mean()/data.balancesheet.loc['Total Stockholder Equity'][0]
    ROA = data.financials.loc['Net Income'].mean()/data.balancesheet.loc['Total Assets'][0]
    print(f'''Profitability of {company}:
          Revenue Growth: {round(g_sales*100,2)}%
          Gross Profit: ${round(gp/1000000000,3)}bn
          Net Profit: ${round(NI/1000000000,3)}bn
          Operating Profit: ${round(ebitda/1000000000,3)}bn
          Gross Profit Margin: {round(gm*100,2)}%
          Operating Profit Margin: {round(OPM*100,2)}%
          Net Profit Margin: {round(NPM*100,2)}%
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
    
    # Determine intrisic value of the real estate company
    if sector == top_realestate:
        # noips0 = float(data.info['ebitda']/data.info['sharesOutstanding'])
        # print(f'Net operating income of {company}: ${round(noips0,3)}')
        #print(f'Sustainable growth rate of {company} stock: {round(g_noips*100,2)}%')
        rf = fred.get_series('TB3MS')[-1]/100
        rm= float(web.DataReader(['sp500'],'fred',d0,d1).replace(np.nan,0).resample('1y').last().pct_change().mean())
        b = data.info['beta']
        re = rf+b*(rm-rf)
        rd = avg_corp_loan_rate
        #abs(yf.Ticker(sector[n]).financials.loc['Interest Expense'][0])/yf.Ticker(sector[n]).info['totalDebt']
        wd = yf.Ticker(sector[n]).info['debtToEquity']/(yf.Ticker(sector[n]).info['debtToEquity']+1)
        we = 1 - wd
        r = wd*(1-tax_rate)*rd+we*re
        # g_noips1 = float(input(f'NOI growth rate in 2022:'))
        # g_noips2 = float(input(f'NOI growth rate in 2023:'))
        # g_noips3 = float(input(f'NOI growth rate in 2024:'))
        # g_noips4 = float(input(f'NOI growth rate in 2025:'))
        # g_noips5 = float(input(f'NOI growth rate in 2026:'))
        # g_noips6 = float(input(f'NOI growth rate in 2027:'))
        # g_noips7 = float(input(f'NOI growth rate in 2028:'))
        # g_noips8 = float(input(f'NOI growth rate in 2029:'))
        # g_noips9 = float(input(f'NOI growth rate in 2030:'))
        # g_noips10 = float(input(f'NOI growth rate in 2031:'))
        # noips1 = noips0*(1+g_noips1)
        # noips2 = noips1*(1+g_noips2)
        # noips3 = noips2*(1+g_noips3)
        # noips4 = noips3*(1+g_noips4)
        # noips5 = noips4*(1+g_noips5)
        # noips6 = noips5*(1+g_noips6)
        # noips7 = noips6*(1+g_noips7)
        # noips8 = noips7*(1+g_noips8)
        # noips9 = noips8*(1+g_noips9)
        # noips10 = noips9*(1+g_noips10)
        # cf = (noips3/(1+r)**1) + (noips4/(1+r)**2) + (noips5/(1+r)**3) + (noips6/(1+r)**4) 
        # + (noips7/(1+r)**5) + (noips8/(1+r)**6) + (noips9/(1+r)**7) + (noips10/(1+r)**8)
        # sum = 0
        
        # yf_data = []
        # for i in range(3,T):                       
        #     sum = sum+cf
        #     if i>7:
        #         print(f'Expected net operating income per share in 203{i-8}: ${round(noips,2)}')
        #     else:
        #         print(f'Expected net operating income per share in 202{i+2}: ${round(noips,2)}')
        #     yf_data.append([noips])
        # df = pd.DataFrame(yf_data,columns={'Net operating income per share'})
        # df = df.transpose()
        # df = pd.DataFrame([[noips1,noips2,noips3,noips4,noips5,noips6,noips7,noips8,noips9,noips10]])
        # df.columns = ['2022E','2023E','2024E','2025E','2026E','2027E','2028E','2029E','2030E','2031E']
        # df['2021A'] = noips0
        # df = df[['2021A','2022E','2023E','2024E','2025E','2026E','2027E','2028E','2029E','2030E','2031E']]
        # fig, ax = plt.subplots(figsize=(10,4))
        # ax.bar(df.columns[0],df['2021A'],width=0.7,color = 'lightcoral')
        # ax.bar(df.columns[1],df['2022E'],width=0.7,color = 'lightcoral')
        # ax.bar(df.columns[2],df['2023E'],width=0.7,color = 'lightcoral')
        # ax.bar(df.columns[3],df['2024E'],width=0.7,color = 'dodgerblue')
        # ax.bar(df.columns[4],df['2025E'],width=0.7,color = 'dodgerblue')
        # ax.bar(df.columns[5],df['2026E'],width=0.7,color = 'dodgerblue')
        # ax.bar(df.columns[6],df['2027E'],width=0.7,color = 'dodgerblue')
        # ax.bar(df.columns[7],df['2028E'],width=0.7,color = 'dodgerblue')
        # ax.bar(df.columns[8],df['2029E'],width=0.7,color = 'dodgerblue')
        # ax.bar(df.columns[9],df['2030E'],width=0.7,color = 'dodgerblue')
        # ax.bar(df.columns[10],df['2031E'],width=0.7,color = 'dodgerblue')
        cf = (forecast.iloc[1,3]/(1+r)**1) + (forecast.iloc[1,4]/(1+r)**2) + (forecast.iloc[1,5]/(1+r)**3) + (forecast.iloc[1,6]/(1+r)**4) 
        + (forecast.iloc[1,7]/(1+r)**5) + (forecast.iloc[1,8]/(1+r)**6) + (forecast.iloc[1,9]/(1+r)**7) + (forecast.iloc[1,10]/(1+r)**8)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.bar(forecast.columns[0],forecast.iloc[1,0]/(data.info['sharesOutstanding']/1000000),width=0.7,color = 'lightcoral')
        ax.bar(forecast.columns[1],forecast.iloc[1,1]/(data.info['sharesOutstanding']/1000000),width=0.7,color = 'lightcoral')
        ax.bar(forecast.columns[2],forecast.iloc[1,2]/(data.info['sharesOutstanding']/1000000),width=0.7,color = 'lightcoral')
        ax.bar(forecast.columns[3],forecast.iloc[1,3]/(data.info['sharesOutstanding']/1000000),width=0.7,color = 'dodgerblue')
        ax.bar(forecast.columns[4],forecast.iloc[1,4]/(data.info['sharesOutstanding']/1000000),width=0.7,color = 'dodgerblue')
        ax.bar(forecast.columns[5],forecast.iloc[1,5]/(data.info['sharesOutstanding']/1000000),width=0.7,color = 'dodgerblue')
        ax.bar(forecast.columns[6],forecast.iloc[1,6]/(data.info['sharesOutstanding']/1000000),width=0.7,color = 'dodgerblue')
        ax.bar(forecast.columns[7],forecast.iloc[1,7]/(data.info['sharesOutstanding']/1000000),width=0.7,color = 'dodgerblue')
        ax.bar(forecast.columns[8],forecast.iloc[1,8]/(data.info['sharesOutstanding']/1000000),width=0.7,color = 'dodgerblue')
        ax.bar(forecast.columns[9],forecast.iloc[1,9]/(data.info['sharesOutstanding']/1000000),width=0.7,color = 'dodgerblue')
        ax.bar(forecast.columns[10],forecast.iloc[1,10]/(data.info['sharesOutstanding']/1000000),width=0.7,color = 'dodgerblue')
        ax.set_title(f'Actual and Forecasted net operating income per share of {company}')
        plt.show()
        print(f'The market consensus for the terminal growth of {company}: {round(terminalgrowth*100,2)}%.')
        terminalvalue = (forecast.iloc[1,10]*(1+terminalgrowth)/(r-terminalgrowth))/(1+r)**(T-2)
        intrinsvalue = (float(np.array(cf+terminalvalue))-float(yf.Ticker(sector[n]).info['totalDebt'])/1000000)/(data.info['sharesOutstanding']/1000000)
        MCPI = pd.DataFrame(fred.get_series('MEDCPIM158SFRBCLE', observation_start=d1-relativedelta(years=1), observation_end=d1), 
                           columns={'Median Consumer Price Index (%)'}).ffill().bfill().mean()
        print(f'Risk-free rate (3 month T-bond rate): {round(rf,2)*100}%')
        print(f'Stock beta: {round(b,2)}')
        print(f'Market risk premium: {round((rm-rf)*100,2)}%')
        print(f'Required rate of return for the stock: {round(r,2)*100}%')
        print(f'Intrinsic value of {company}: ${round(intrinsvalue,2)}')
        print(f'Current price of {company}: ${round(currentprice,2)}')
        print('\x1b[47m'+'Investment Decision:'+'\x1b[47m')
        print(f"""              BUY: More than {round(float(MCPI.mean()),2)+5}% upside potential
              HOLD: Within {max(round(float(MCPI.mean()),2),5)}% risk of loss and {round(float(MCPI.mean()),2)+5}% gain potential
              SELL: More than {max(round(float(MCPI.mean()),2),5)}% downside risk""")
        print('\n')
        if intrinsvalue<currentprice*(1-(max(round(float(MCPI.mean()),2),5)/100)):
            
            print('\x1b[41m'+f'Downside risk of loss: {round((1-intrinsvalue/currentprice)*100,2)}%'+'\x1b[41m')
            print('\x1b[41m'+f'{company} is overvalued!!! So, we give SELL recommendation for the stock.'+'\x1b[41m')
        elif intrinsvalue>currentprice*(1+((round(float(MCPI.mean()),2)+5)/100)):
            print('\x1b[42m'+f'Upside potential: {round((intrinsvalue/currentprice-1)*100,2)}%'+'\x1b[42m')
            print('\x1b[42m'+f'{company} is undervalued!!! So, we give BUY recommendation for the stock.'+'\x1b[42m')
        else:
            print('\x1b[47m'+f'Investment return: {round((intrinsvalue/currentprice-1)*100,2)}%'+'\x1b[47m')
            print('\x1b[47m'+f'{company} is fairly valued!! So, we should hold the stock.'+'\x1b[47m')
    
    # Determine the intrinsic value of the health care company
    else:
        ret_ratio = 1 - payout
        g = ROE * ret_ratio
        print(f'Sustainable growth rate of {company} stock: {round(g*100,2)}%')
        rf = fred.get_series('TB3MS')[-1]/100
        rm= float(web.DataReader(['sp500'],'fred',d0,d1).replace(np.nan,0).resample('1y').last().pct_change().mean())
        b = data.info['beta']
        r = rf+b*(rm-rf)
        # t=1
        # sum = 0
        # yf_data = []
        # for i in range(T):
        #     div=float(div0)*(1+g)**(i+t)
        #     div1 = div0*(1+g)
        #     div2 = div1*(1+g)
        #     div3 = div2*(1+g)
        #     cf = div3/(1+r)**(i+t)
        #     sum = sum+cf
        #     if i>7:
        #         print(f'Expected dividend per share in 203{i-8}: ${round(div,2)}')
        #     else:
        #         print(f'Expected dividend per share in 202{i+2}: ${round(div,2)}')
        #     yf_data.append([div])
        # df = pd.DataFrame(yf_data,columns={'Dividends'})
        # df = df.transpose()
        # df.columns = ['2022E','2023E','2024E','2025E','2026E','2027E','2028E','2029E','2030E','2031E']
        # df['2021A'] = div0
        # df = df[['2021A','2022E','2023E','2024E','2025E','2026E','2027E','2028E','2029E','2030E','2031E']]
        cf = (forecast.iloc[5,3]/(1+r)**1) + (forecast.iloc[5,4]/(1+r)**2) + (forecast.iloc[5,5]/(1+r)**3) + (forecast.iloc[5,6]/(1+r)**4) 
        + (forecast.iloc[5,7]/(1+r)**5) + (forecast.iloc[5,8]/(1+r)**6) + (forecast.iloc[5,9]/(1+r)**7) + (forecast.iloc[5,10]/(1+r)**8)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.bar(forecast.columns[0],forecast.iloc[5,0],width=0.7,color = 'lightcoral')
        ax.bar(forecast.columns[1],forecast.iloc[5,1],width=0.7,color = 'lightcoral')
        ax.bar(forecast.columns[2],forecast.iloc[5,2],width=0.7,color = 'lightcoral')
        ax.bar(forecast.columns[3],forecast.iloc[5,3],width=0.7,color = 'dodgerblue')
        ax.bar(forecast.columns[4],forecast.iloc[5,4],width=0.7,color = 'dodgerblue')
        ax.bar(forecast.columns[5],forecast.iloc[5,5],width=0.7,color = 'dodgerblue')
        ax.bar(forecast.columns[6],forecast.iloc[5,6],width=0.7,color = 'dodgerblue')
        ax.bar(forecast.columns[7],forecast.iloc[5,7],width=0.7,color = 'dodgerblue')
        ax.bar(forecast.columns[8],forecast.iloc[5,8],width=0.7,color = 'dodgerblue')
        ax.bar(forecast.columns[9],forecast.iloc[5,9],width=0.7,color = 'dodgerblue')
        ax.bar(forecast.columns[10],forecast.iloc[5,10],width=0.7,color = 'dodgerblue')
        ax.set_title(f'Actual and Forecasted Dividends of {company} (in USD)')
        plt.show()
        print(f'The market consensus for the terminal growth of {company}: {round(terminalgrowth*100,2)}%.')
        
        value = (forecast.iloc[5,10]*(1+terminalgrowth)/(r-terminalgrowth))/(1+r)**(T-2)
        intrinsvalue = float(np.array(cf+terminalvalue))
        MCPI = pd.DataFrame(fred.get_series('MEDCPIM158SFRBCLE', observation_start=d1-relativedelta(years=1), observation_end=d1), 
                           columns={'Median Consumer Price Index (%)'}).ffill().bfill().mean()
        print(f'Risk-free rate (3 month T-bond rate): {round(rf,2)*100}%')
        print(f'Stock beta: {round(b,2)}')
        print(f'Market risk premium: {round((rm-rf)*100,2)}%')
        print(f'Required rate of return for the stock: {round(r*100,2)}%')
        print(f'Intrinsic value of {company}: ${round(intrinsvalue,2)}')
        print(f'Current price of {company}: ${round(currentprice,2)}')
        print('\x1b[47m'+'Investment Decision:'+'\x1b[47m')
        print(f"""              BUY: More than {round(float(MCPI.mean()),2)+5}% upside potential
              HOLD: Within {max(round(float(MCPI.mean()),2),5)}% risk of loss and {round(float(MCPI.mean()),2)+5}% gain potential
              SELL: More than {max(round(float(MCPI.mean()),2),5)}% downside risk""")
        print('\n')
        if intrinsvalue<currentprice*(1-(max(round(float(MCPI.mean()),2),5)/100)):
            
            print('\x1b[41m'+f'Downside risk of loss: {round((1-intrinsvalue/currentprice)*100,2)}%'+'\x1b[41m')
            print('\x1b[41m'+f'{company} is overvalued!!! So, we give SELL recommendation for the stock.'+'\x1b[41m')
        elif intrinsvalue>currentprice*(1+((round(float(MCPI.mean()),2)+5)/100)):
            print('\x1b[42m'+f'Upside potential: {round((intrinsvalue/currentprice-1)*100,2)}%'+'\x1b[42m')
            print('\x1b[42m'+f'{company} is undervalued!!! So, we give BUY recommendation for the stock.'+'\x1b[42m')
        else:
            print('\x1b[47m'+f'Investment return: {round((intrinsvalue/currentprice-1)*100,2)}%'+'\x1b[47m')
            print('\x1b[47m'+f'{company} is fairly valued!! So, we should hold the stock.'+'\x1b[47m')


# # Run the code for CVS
# intrinsic_value(0.01,10,health_care,0,0.21,0.07)
# # Run the code for LLY
# intrinsic_value(0.01,10,health_care,1,0.21,0.07)
# # Run the code for LLY
# intrinsic_value(0.01,10,health_care,2,0.21,0.07)
# # Run the code for CBRE
# intrinsic_value(0.01,10,real_estate,0,0.21,0.07)
# Run the code for PLD
intrinsic_value(0.03,10,top_realestate,0,0.21,0.07)
# # Run the code for UNH
# intrinsic_value(0.05,10,health_insurance,0,0.21,0.07)

