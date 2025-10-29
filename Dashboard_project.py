"""CREATE A MARKET DAHSBOARD PROJECT / WITH ADVANCED MACROECONOMIC INDICATOR AND STRATEGY BACKTEST"""

# import all modules 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import yfinance as yf
import datetime as dt
import plotly.express as px

# Create a dictionnary of indexes

indices = {
    "^GSPC" : "S&P500",
    "^IXIC" : "NASDAQ",
    "^FTSE" : "FTSE100",
    "^FCHI" : "CAC40",
    "^HSI" : "HANG SENG",
    "^BVSP" : "BOVESPA",
    "^N225" : "NIKKEI225",
    "^STOXX" : "STOXX600",
    "^GDAXI" : "DAX",
    "^NYA" : "NYSE",
    "^MXX" : "IPC Mexico",
    "^SSMI" : "SMI",
    "^IBEX" : "IBEX35",
    "^AEX" : "AEX",
    "^NSEI" : "NIFTY50"
}

# Create a dictionnary of sectors

sectors = {
    "^YH310" : "Industrials",
    "^YH101" : "Materials",
    "^YH103" : "Financial",
    "^YH207" : "Utilities",
    "^YH102" : "Cons. Discr.",
    "^YH205" : "Cons. Stap.",
    "^YH309" : "Energy",
    "^YH206" : "Health Care",
    "^YH311" : "Technology",
    "^YH308" : "Communication",
    "^YH104" : "Real Estate"
}

# Create a function for SP500 stocks

def treemap(csv, title):
    data = pd.read_csv(csv)
    data["Weight"] = data["Weight"].astype(float)*100
    chart = px.treemap(data, path=[px.Constant("SP500"),"Sector", "Industry", "Shortname"], values="Weight",title=title, height=700)
    st.plotly_chart(chart)

# Create a dictionnary of forex

forex = {
    "EURUSD=X" : "EUR",
    "JPYUSD=X" : "JPY",
    "GBPUSD=X" : "GBP",
    "AUDUSD=X" : "AUD",
    "NZDUSD=X" : "NZD",
    "CADUSD=X" : "CAD",
    "CHFUSD=X" : "CHF", 
    "CNYUSD=X" : "CNY",
    "MXNUSD=X" : "MXN",
    "RUBUSD=X" : "RUB"
}

# Create a dictionnary of commodities

commodities = {
    "GC=F" : "Gold",
    "SI=F" : "Silver",
    "CL=F" : "Crude Oil",
    "PL=F" : "Platinum",
    "HG=F" : "Copper",
    "PA=F" : "Palladium",
    "NG=F" : "Natural Gas", 
    "ZC=F" : "Corn",
    "ZO=F" : "Oat",
    "ZS=F" : "Soybean",
    "LE=F" : "Live Cattle", 
    "CC=F" : "Cocoa",
    "KC=F" : "Coffee",
    "CT=F" : "Cotton", 
    "OJ=F" : "Orange Juice",
    "SB=F" : "Sugar"
}

# Create a dictionnary of bonds

USbonds = {
    "^IRX" : "3m T-Bill",
    "2YY=F" : "2y T-Bond",
    "^FVX" : "5y T-Bond",
    "^TNX" : "10y T-Bond",
    "^TYX" : "30y T-Bond"
}

# Create a dictionnary for cryptos

Crypto = {
    "BTC-USD" : "Bitcoin",
    "ETH-USD" : "Ethereum",
    "BNB-USD" : "BNB",
    "XRP-USD" : "Ripple",
    "SOL-USD" : "Solana",
    "TRX-USD" : "Tron",
    "DOGE-USD" : "Dogecoin",
    "ADA-USD" : "Cardano"
}

# Create a dictionnary of macro (cpi...)

# Create a function to import data from Fred

def fred_table(dictionnary):
    list_fred = []
    for ticker, name in dictionnary.items():
        data = yf.download(ticker, start, end)["Close"]
        data.to_csv(f"{name}.csv", index=True)
        list_fred.append(data)
    table = pd.concat(list_fred, axis=1)
    table = table.rename(columns=dictionnary)
    table = table.ffill().bfill()
    return table

# Create a plot function for bonds

def plot_bonds (dictionnary, title):
    st.subheader(title)
    st.line_chart(fred_table(dictionnary))

# Create a function that will import data from yahoo finance for dictionnaries and create a corresponding csv file

def import_data(ticker, name):
    data = yf.download(ticker,start, end)["Close"]
    data.to_csv(f"{name}.csv", index=True)
    return data

# Create a function that will create a table with the data concatenated

def table (dictionnary):
    list = []
    for ticker, name in dictionnary.items():
        df = import_data(ticker, name)
        list.append(df)
    
    table = pd.concat(list, axis=1)
    table = table.ffill().bfill()
    table = table.rename(columns=dictionnary)
    return table

# Create a function that normalize the data of the table

def normalize (dictionnary):
    returns = table(dictionnary).pct_change().fillna(0)
    normalized_table = (1+returns).cumprod()
    return normalized_table

# Create a function that plots the data table through a chart

def plot_chart (dictionnary, title):
    col1,col2 = st.columns(2)
    with col1:
        start, end = st.date_input("Period:", value=(normalize(dictionnary).index.min().date(),normalize(dictionnary).index.max().date()),key=f"period_{title}")
        frequency = st.selectbox("Select Frequency:", options=frequency_list, key=f"freq_{title}")
    with col2:
        assets = st.multiselect("Selected assets:", options=normalize(dictionnary).columns, default=normalize(dictionnary).columns, key=f"assets_{title}")
        smas = st.multiselect("Moving Averages:", options=sma_list, default=None, key=f"sma_{title}")
    selection = selected_assets(dictionnary, assets)
    filtered_data = period(selection, start, end)    
    fig_data = selected_frequency(frequency, filtered_data)
    fig_wtih_smas = selected_smas(fig_data, smas)
    st.subheader(title)
    st.line_chart(fig_wtih_smas)

# Create a function that plots the data table through a small chart

def plot_second_chart (dictionnary, title):
    col1,col2 = st.columns(2)
    with col1:
        start, end = st.date_input("Period:", value=(table(dictionnary).index.min().date(),table(dictionnary).index.max().date()),key=f"period_{title}")
        frequency = st.selectbox("Select Frequency:", options=frequency_list, key=f"freq_{title}")
    with col2:
        assets = st.selectbox("Selected assets:", options=table(dictionnary).columns, key=f"assets_{title}")
        smas = st.multiselect("Moving Averages:", options=sma_list, default=None, key=f"sma_{title}")
    selection = table(dictionnary)[assets]
    filtered_data = period(selection, start, end, "not none")    
    fig_data = selected_frequency(frequency, filtered_data)
    fig_wtih_smas = selected_smas(fig_data, smas)
    st.subheader(title)
    st.line_chart(fig_wtih_smas)

# Create a function ton filter data
    
def selected_frequency(frequency, filtered_data):
    if frequency == "Annually":
        fig = pd.DataFrame(timeframes(filtered_data,"Y"))
    elif frequency == "Monthly":
        fig = pd.DataFrame(timeframes(filtered_data,"M"))
    elif frequency == "Weekly":
        fig = pd.DataFrame(timeframes(filtered_data,"W"))
    elif frequency == "Daily":
        fig = pd.DataFrame(timeframes(filtered_data,"D"))
    return fig

# Create a function for selecting assets
    
def selected_assets(dictionnary, assets):
    selection = normalize(dictionnary)[assets]
    return selection

# Create a function for selecting SMAs

def selected_smas(data, smas):
    if not smas:
        final_set = data
    else: 
        final_set = pd.DataFrame(data.copy())
        for sma in smas:
            final_set[f"SMA_{sma}"] = data.rolling(window=sma).mean()
    return final_set

# Create different timeframes
    
def timeframes (data, period):
    return data.resample(period).last()

# Create the period selected for the analysis

def period (selection, start, end, type=None):
    full_index = pd.date_range(start=start, end=end, freq="B")
    data = selection.reindex(full_index).ffill()
    if type == None:
        data = ((data / data.iloc[0])-1)*100
    return data

# Create a correlation matrix of the indexes

def plot_correl(dictionnary, title):
    col1,col2 = st.columns(2)
    with col2:
        frequency = st.selectbox("Select Frequency:", options=frequency_list, key=f"freq_{title}")
    with col1:
        start, end = st.date_input("Period:", value=(normalize(dictionnary).index.min().date(),normalize(dictionnary).index.max().date()),key=f"period_{title}")
    selection = normalize(dictionnary)
    filtered_data = period(selection, start, end)
    if frequency == "Annually":
        correl_indices = timeframes(filtered_data,"Y").corr(method="pearson").mul(100).round(0)
    elif frequency == "Monthly":
        correl_indices = timeframes(filtered_data,"M").corr(method="pearson").mul(100).round(0)
    elif frequency == "Weekly":
        correl_indices = timeframes(filtered_data,"W").corr(method="pearson").mul(100).round(0)
    elif frequency == "Daily":
        correl_indices = timeframes(filtered_data,"D").corr(method="pearson").mul(100).round(0)
    fig = px.imshow(correl_indices, text_auto= True, width=800, height=600, title= title)
    st.plotly_chart(fig, use_container_width=True)

# Create a function to create a table of yields over different periods
    
def yield_table (dictionnary):
    data = table(dictionnary)
    last = data.iloc[-1]
    def safe(i):
        return data.iloc[-i] if len(data) > i else data.iloc[0]
    ytable = pd.DataFrame({
        "Daily": ((last/ safe(2)) - 1) * 100,
        "Weekly": ((last/ safe(7)) - 1) * 100,
        "Monthly": ((last/ safe(30)) - 1) * 100,
        "YTD": ((last/ data.loc[data.index >= f"{dt.date.today().year}-01-01"].iloc[0]) - 1) * 100,
        "Yearly": ((last/ safe(252)) - 1) * 100,
        "5Y": ((last/ safe(252*5)) - 1) * 100,
        "10Y": ((last/ safe(252*10)) - 1) * 100
    }).T.round(2).div(100)
    return ytable

# Create a function to plot the yield table

def plot_ytable(dictionnary, title):
    data = yield_table(dictionnary).transpose()
    fig = sns.light_palette("green", as_cmap=True)
    style = data.style.format("{:.1%}").text_gradient(cmap=fig)
    st.subheader(title)
    st.dataframe(style, height=(int(data.shape[0]+1) * 35))

# Create a function to plot the yield curve

def plot_ycurve(dictionnary, title):
    table = fred_table(dictionnary)
    data = table.iloc[-1]
    df_curve = pd.DataFrame({
        "Maturity":data.index,
        "Yield":data.values
    }, )
    maturity_order = list(dictionnary.values())
    df_curve['Maturity'] = pd.Categorical(
        df_curve['Maturity'], 
        categories=maturity_order, 
        ordered=True
    )
    df_curve = df_curve.sort_values("Maturity")
    st.subheader(title)
    st.line_chart(df_curve, x="Maturity", y="Yield")
    

# Create a figure to select and compare bonds

# Create a figure to select and compare sectors in different regions
# Create a figure to select and compare the forex market

# Create a figure to select and compare pairs of currency
# Create a correlation matrix of the forex market

# Create advanced macroeconomic indicators
# Create a figure to showcase them (a 4-quadrant one)

###############################################################
    #MAIN WORKFLOW
###############################################################
    
start = dt.date(year=2005, month=1, day=1)
end = dt.date.today()
pages = ["Market charts", "Bond Market", "Macro", "Backtest"]
frequency_list = ["Annually", "Monthly", "Weekly", "Daily"]
sma_list = [200, 50, 20]
st.set_page_config(layout="wide")
st.sidebar.header("SUMMARY")
selected_page = st.sidebar.selectbox("Page Selection:", options=pages, key="Page selection")

if selected_page == "Market charts":
    st.title("Market Dashboard")
    with st.container():
        plot_chart(indices, "Major Indices Normalized Returns")
        col1, col2 = st.columns(2)
        with col1:
            plot_ytable(indices, "Indices Yield Table")
        with col2:
            plot_correl(indices, "Correlation Matrix for Major Indices")
    with st.container():
        plot_chart(sectors, "SP500 Sectors Normalized Returns")
        col1, col2 = st.columns(2)
        with col1:
            treemap("/Users/mathis/Desktop/MATHIS/PRO/PROJETS/sp500_companies.csv", "Sectors Treemap")
        with col2:
            plot_correl(sectors, "SP500 Sectors Correlation Matrix")
    with st.container():
        plot_chart(commodities, "Major Commodities Normalized Returns")
        col1, col2 = st.columns(2)
        with col1:
            plot_ytable(commodities, "Commodities Yield Table")
        with col2:
            plot_correl(commodities, "Correlation Matrix for Major Commodities")
    with st.container():
        col1, col2 = st.columns([0.6,0.4])
        with col1:
            plot_chart(forex, "Major Currencies Normalized Returns")
        with col2:
            plot_second_chart(forex,"Price of currency exchange")
        col1, col2 = st.columns(2)
        with col1:
            plot_ytable(forex, "Currencies Yield Table")
        with col2:
            plot_correl(forex, "Correlation Matrix for Major Currencies")
    with st.container():
        col1, col2 = st.columns([0.6,0.4])
        with col1:
            plot_chart(Crypto, "Major Cryptos Normalized Returns")
        with col2:
            plot_second_chart(Crypto,"Price of Crypto exchange")
        col1, col2 = st.columns(2)
        with col1:
            plot_ytable(Crypto, "Cryptos Yield Table")
        with col2:
            plot_correl(Crypto, "Correlation Matrix for Major Cryptos")
elif selected_page == "Bond Market":
    st.title("Bond Market")
    with st.container():
        plot_bonds(USbonds, "US Bonds yield")
        col1, col2 = st.columns(2)
        with col1:
            plot_ytable(USbonds, "US Bonds Yield Table")
        with col2:
            plot_ycurve(USbonds, "US Yield Curve")

#elif selected_page == "Macro":

#elif selected_page == "Backtest:
