"""CREATE A MARKET DAHSBOARD PROJECT / WITH ADVANCED MACROECONOMIC INDICATOR AND STRATEGY BACKTEST"""

# import all modules 

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import yfinance as yf
import pandas_datareader.data as web
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go

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

# Create a dictionnary of US bonds

USbonds = {
    
    "DGS1MO" : "1m T-Bill",
    "DGS3MO" : "3m T-Bill",
    "DGS6MO" : "6m T-Bill",
    "DGS1" : "1y T-Bond",
    "DGS2" : "2y T-Bond",
    "DGS3" : "3y T-Bond",
    "DGS5" : "5y T-Bond",
    "DGS7" : "7y T-Bond",
    "DGS10" : "10y T-Bond",
    "DGS20" : "20y T-Bond",
    "DGS30" : "30y T-Bond"
}

# Create a dictionnary of other countries 10Y bonds

TenYbonds = {
    "IRLTLT01DEM156N": "Germany",
    "IRLTLT01FRM156N": "France",
    "IRLTLT01ITM156N": "Italy",
    "IRLTLT01GBM156N": "United Kingdom",
    "IRLTLT01JPM156N": "Japan",
    "IRLTLT01ESM156N": "Spain",
    "IRLTLT01PTM156N": "Portugal",
    "IRLTLT01GRM156N": "Greece",
    "IRLTLT01CAM156N": "Canada",
    "IRLTLT01AUM156N": "Australia",
    "IRLTLT01CHM156N": "Switzerland",
    "IRLTLT01USM156N" : "United States"
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

inflation = {
    "CPIAUCNS": "US CPI"
}

growth = {
    "GDP": "US GDP"
}

fedfund = {
    "FEDFUNDS": "Fed Fund Rate"
}

# Create a function to import data for sectors

@st.cache_data
def fred_table(dictionnary):
    list_fred = []
    for ticker, name in dictionnary.items():
        data = web.DataReader(ticker, "fred", start= "2010-1-1")
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
    data = pd.DataFrame(data)
    data.rename(columns={"Close":name}, inplace=True)
    data.to_csv(f"{name}.csv", index=True)
    return data

# Create a function that will create a table with the data concatenated

def table (dictionnary):
    list = []
    for ticker, name in dictionnary.items():
        df = import_data(ticker, name)
        list.append(df)
    table = pd.concat(list, axis=1)
    table = table.asfreq("D")
    table = table.ffill().bfill()
    table = table.rename(columns=dictionnary)
    return table

# Create a function that normalize the data of the table

def normalize (dictionnary):
    returns = table(dictionnary).pct_change().fillna(0)
    normalized_table = (((1+returns).cumprod())-1)*100
    return normalized_table

# Create a function that plots the data table through a chart

def plot_chart (dictionnary, title):
    col1,col2 = st.columns([1,3])
    with col1:
        start, end = st.date_input("Period:", value=(normalize(dictionnary).index.min().date(),normalize(dictionnary).index.max().date()),key=f"period_{title}")
        assets = st.multiselect("Selected assets:", options=normalize(dictionnary).columns, default=normalize(dictionnary).columns, key=f"assets_{title}")
        smas = st.multiselect("Moving Averages:", options=sma_list, default=None, key=f"sma_{title}")
    selection = selected_assets(dictionnary, assets)
    filtered_data = period(selection, start, end, "not none")
    fig_wtih_smas = selected_smas(filtered_data,smas)
    with col2:
        st.subheader(title)
        st.line_chart(fig_wtih_smas)

# Create a function that plots the data table through a small chart

def plot_second_chart (dictionnary, title):
    col1,col2 = st.columns([1,3])
    with col1:
        start, end = st.date_input("Period:", value=(table(dictionnary).index.min().date(),table(dictionnary).index.max().date()),key=f"period_{title}")
        assets = st.selectbox("Selected assets:", options=table(dictionnary).columns, key=f"assets_{title}")
        smas = st.multiselect("Moving Averages:", options=sma_list, default=None, key=f"sma_{title}")
    selection = table(dictionnary)[assets]
    filtered_data = period(selection, start, end, "not none")
    fig_wtih_smas = selected_smas(filtered_data, smas)
    with col2:
        st.subheader(title)
        st.line_chart(fig_wtih_smas)

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
    fig = px.imshow(correl_indices, text_auto= True, width=800, height=600, title= title, color_continuous_scale="RdYlBu", color_continuous_midpoint=0)
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
    style = data.style.format("{:.1%}").applymap(lambda x: "color: green" if x > 0 else "color: red")
    st.subheader(title)
    st.dataframe(style, height=(int(data.shape[0]+1) * 35))

# Create a function to plot the yield curve

def plot_ycurve(dictionnary, title): 
    table = fred_table(dictionnary)
    data = table.iloc[-1]
    df_curve = pd.DataFrame({
        "Maturity":data.index,
        "Yield":data.values
    })
    maturity_order = list(dictionnary.values())
    df_curve['Maturity'] = pd.Categorical(
        df_curve['Maturity'], 
        categories=maturity_order, 
        ordered=True
    )
    df_curve = df_curve.sort_values("Maturity")
    st.subheader(title)
    st.line_chart(df_curve, x="Maturity", y="Yield")
    

# Create a figure to select and compare 10Y bonds
    
def plot_10y(dictionnary, title):
    data = fred_table(dictionnary)
    data = data.iloc[-1].sort_values().to_frame(name="10Y Yields")
    data.index = pd.CategoricalIndex(data.index, categories=data.index, ordered=True)
    st.subheader(title)
    st.bar_chart(data, )

# Create a proxy for inflation
    
def prox_inflation():
    data_list = ["Crude oil","Natural Gas"]
    data = []
    for proxy in data_list:
        df = pd.read_csv(f"{proxy}.csv",parse_dates=["Date"] ,index_col="Date")
        data.append(df)
    table = pd.concat(data, axis=1)
    table = table.asfreq("D").pct_change(periods=365)*100
    table = table.ffill()
    table["Proxy"] = ((table["CL=F"]))/20
    return table["Proxy"]
    
# Create advanced macroeconomic indicators
    
def plot_macro (dictionnary1, period, title):
    data = fred_table(dictionnary1)
    data = data.pct_change(periods=period)*100
    st.subheader(title)
    st.line_chart(data)

# Create interest rate indicator
    
def plot_interest (dictionnary, title):
    data = fred_table(dictionnary)
    st.subheader(title)
    st.bar_chart(data)

# Create the backtest

def backtest(asset, sma_short, sma_long):
    df = pd.read_csv(f"{asset}.csv", parse_dates=["Date"], index_col="Date")
    df.rename(columns={df.columns[0]: asset}, inplace=True)
    df = df.pct_change()
    df = (((1+ df).cumprod())-1)*100
    infl = prox_inflation()
    infl = pd.DataFrame(infl)
    table = pd.concat([df,infl],axis=1)
    table = table.dropna()
    table[f"SMA {sma_short}"] = table[asset].rolling(sma_short).mean()
    table[f"SMA {sma_long}"] = table[asset].rolling(sma_long).mean()
    table = table.dropna()
    return table

# Create the dataframe for analysis

def table_analysis(asset, sma_short, sma_long):
    df = pd.read_csv(f"{asset}.csv", parse_dates=["Date"], index_col="Date")
    df.rename(columns={df.columns[0]: asset}, inplace=True)
    df = df.pct_change()*100
    infl = prox_inflation()
    infl = pd.DataFrame(infl)
    table = pd.concat([df,infl],axis=1)
    table = table.dropna()
    table[f"SMA {sma_short}"] = table[asset].rolling(sma_short).mean()
    table[f"SMA {sma_long}"] = table[asset].rolling(sma_long).mean()
    table = table.dropna()
    return table

# Plot chart for the backtest

def chart_backtest(asset, title, sma_short, sma_long):
    df = backtest(asset, sma_short, sma_long)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[asset], name=asset, yaxis="y1"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Proxy"], name="Proxy", yaxis="y2"))
    fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA {sma_short}"], name=f"SMA {sma_short}", yaxis="y1"))
    fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA {sma_long}"], name=f"SMA {sma_long}", yaxis="y1"))
    fig.update_layout(
        title=title,
        yaxis=dict(title=f"{asset} Returns (%)"),
        yaxis2=dict(title="Proxy Inflation (%)", overlaying="y", side="right")
    )
    st.plotly_chart(fig, use_container_width=True)
    
# Function to showcase pdf about the different periods
    
def period_analysis(asset,period,sma_short,sma_long):
    table = table_analysis(asset,sma_short,sma_long)
    if period == "whole period":
        data = table
        title = "during whole period"
    elif period == "increasing inflation":
        data = table[(table["Proxy"]>0) & (table[f"SMA {sma_short}"] > table[f"SMA {sma_long}"])]
        title = f"when inflation & SMA{sma_short} > SMA{sma_long}"
    elif period == "decreasing inflation":
        data = table[(table["Proxy"]>0) & (table[f"SMA {sma_short}"]<table[f"SMA {sma_long}"])]
        title = f"when inflation & SMA{sma_short} < SMA{sma_long}"
    elif period == "increasing deflation":
        data = table[(table["Proxy"]<0) & (table[f"SMA {sma_short}"]>table[f"SMA {sma_long}"])]
        title = f"when deflation & SMA{sma_short} > SMA{sma_long}"
    elif period == "decreasing deflation":
        data = table[(table["Proxy"]<0) & (table[f"SMA {sma_short}"]<table[f"SMA {sma_long}"])]
        title = f"when deflation & SMA{sma_short} < SMA{sma_long}"
    return data, title

# Plot analysis

def plot_anlysis(asset,period,sma_short,sma_long):
    data,title = period_analysis(asset,period,sma_short,sma_long)
    fig = px.histogram(
        data,
        x = data.iloc[:,0],
        nbins=200,
        histnorm="probability density",
        title =f"{asset} returns {title}",
        labels={asset:"Returns", "count":"Probability"}
    )
    mean = data.iloc[:,0].mean()
    std = data.iloc[:,0].std()
    x_vals = np.linspace(data.iloc[:,0].min(),data.iloc[:,0].max(),500)
    y_vals = stats.norm.pdf(x_vals,mean, std)
    fig.add_trace(go.Scatter(x=x_vals,y=y_vals,mode='lines',name="Normal fit"))
    fig.add_annotation(x=0.95,y=0.95,xref='paper',yref="paper",text=f"Mean: {mean:.4f}<br>Std: {std:.4f}")
    st.plotly_chart(fig, use_container_width=True)

# Statistics for each strategy

def statistics(asset,sma_short,sma_long):
    list_periods=["whole period","increasing inflation","decreasing inflation","increasing deflation","decreasing deflation"]
    Returns=[]
    Volatilities=[]
    Annualized_Means=[]
    Max_Drawdowns=[]
    Sharpe=[]
    Skewness=[]
    Kurtosis=[]
    Time=[]
    for period in list_periods:
        df,title = period_analysis(asset,period,sma_short,sma_long)
        df = df[asset]
        cum = (1+(df/100)).cumprod()-1
        Returns.append(f"{(cum.iloc[-1]*100).round(2)}%")
        Volatilities.append(f"{((df).std() * np.sqrt(252)).round(2)}%")
        Annualized_Means.append(f"{(df.mean()*252).round(2)}%")
        Max_Drawdowns.append(f"{df.min():.2f}%")
        Sharpe.append(f"{(((df.mean()*252)-2)/(df.std() * np.sqrt(252))).round(2)}")
        Skewness.append(f"{(stats.skew(df,axis=0)).round(2)}")
        Kurtosis.append(f"{(stats.kurtosis(df,axis=0)).round(2)}")
        Time.append(f"{len(df)/(len(table_analysis(asset,sma_short,sma_long)))*100:.2f}%")
    data = pd.DataFrame({
        "Return":Returns,
        "Volatility":Volatilities,
        "Annualized Mean":Annualized_Means,
        "Max Loss":Max_Drawdowns,
        "Sharpe":Sharpe,
        "Skewness":Skewness,
        "Kurtosis":Kurtosis,
        "Time percentage":Time
        }).T
    list_col = [
        "Whole Period",
        f"Infl. SMA{sma_short}>SMA{sma_long}",
        f"Infl. SMA{sma_short}<SMA{sma_long}",
        f"Defl. SMA{sma_short}>SMA{sma_long}",
        f"Defl. SMA{sma_short}<SMA{sma_long}"]
    data.columns = list_col
    st.subheader("Table of statistics")
    st.dataframe(data)


###############################################################
    #MAIN WORKFLOW
###############################################################
    
start = dt.date(year=2010, month=1, day=1)
end = dt.date.today()
pages = ["Market charts", "Bond Market", "Macro", "Backtest"]
frequency_list = ["Annually", "Monthly", "Weekly", "Daily"]
sma_list = [200, 50, 20]
st.set_page_config(layout="wide")
st.sidebar.header("SUMMARY")
selected_page = st.sidebar.selectbox("Page Selection:", options=pages, key="Page selection")

if selected_page == "Market charts":
    st.title("🏙️ Market Dashboard")
    with st.container():
        st.markdown("""<hr style="height:10px;border:none;color:#e0e0e0;background-color:#e0e0e0;" /> """, unsafe_allow_html=True) 
        plot_chart(indices, "Major Indices Normalized Returns")
        col1, col2 = st.columns(2)
        with col1:
            plot_ytable(indices, "Indices Yield Table")
        with col2:
            plot_correl(indices, "Correlation Matrix for Major Indices")
        st.markdown("""<hr style="height:10px;border:none;color:#e0e0e0;background-color:#e0e0e0;" /> """, unsafe_allow_html=True) 
    with st.container():
        plot_chart(sectors, "SP500 Sectors Normalized Returns")
        col1, col2 = st.columns(2)
        with col1:
            treemap("/Users/mathis/Desktop/MATHIS/PRO/PROJETS/sp500_companies.csv", "Sectors Treemap")
        with col2:
            plot_correl(sectors, "SP500 Sectors Correlation Matrix")
        st.markdown("""<hr style="height:10px;border:none;color:#e0e0e0;background-color:#e0e0e0;" /> """, unsafe_allow_html=True) 
    with st.container():
        plot_chart(commodities, "Major Commodities Normalized Returns")
        col1, col2 = st.columns(2)
        with col1:
            plot_ytable(commodities, "Commodities Yield Table")
        with col2:
            plot_correl(commodities, "Correlation Matrix for Major Commodities")
        st.markdown("""<hr style="height:10px;border:none;color:#e0e0e0;background-color:#e0e0e0;" /> """, unsafe_allow_html=True) 
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
        st.markdown("""<hr style="height:10px;border:none;color:#e0e0e0;background-color:#e0e0e0;" /> """, unsafe_allow_html=True) 
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
        st.markdown("""<hr style="height:10px;border:none;color:#e0e0e0;background-color:#e0e0e0;" /> """, unsafe_allow_html=True) 
        plot_bonds(USbonds, "US Bonds yield")
        col1, col2 = st.columns(2)
        with col1:
            plot_10y(TenYbonds, "10Y yield")
        with col2:
            plot_ycurve(USbonds, "US Yield Curve")
    with st.container():
        plot_bonds(TenYbonds,"10Y yields historical chart")
elif selected_page == "Macro":
    st.title("Macro")
    st.markdown("""<hr style="height:10px;border:none;color:#e0e0e0;background-color:#e0e0e0;" /> """, unsafe_allow_html=True) 
    col1, col2 = st.columns(2)
    with col1:
        plot_macro(inflation,12,"US CPI Index")
    with col2:
        plot_interest(fedfund,"Fed Fund Rate")
    with st.container():
        plot_macro(growth,4,"US GDP")
elif selected_page == "Backtest":
    st.title("Backtest")
    st.markdown("""<hr style="height:10px;border:none;color:#e0e0e0;background-color:#e0e0e0;" /> """, unsafe_allow_html=True) 
    with st.container():
        selection = st.selectbox("select asset for analysis:",options=["S&P500"]+list(sectors.values()),index=0)
        x = st.slider("Short SMA:",step=1,min_value=1,max_value=3650)
        y = st.slider("Long SMA:",step=1,min_value=2,max_value=3650)
        chart_backtest(selection, f"{selection} Normalized Returns",x,y)
        col1,col2 = st.columns(2)
        with col1:
            plot_anlysis(selection, "whole period",x,y)
        with col2:
            statistics(selection,x,y)
        col1,col2 = st.columns(2)
        with col1:
            plot_anlysis(selection, "increasing inflation",x,y)
            plot_anlysis(selection, "decreasing inflation",x,y)
        with col2:
            plot_anlysis(selection, "increasing deflation",x,y)
            plot_anlysis(selection, "decreasing deflation",x,y)
