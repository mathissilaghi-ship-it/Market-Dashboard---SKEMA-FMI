# import all modules 
import numpy as np
import scipy.stats as stats
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import yfinance as yf
import pandas_datareader.data as web
import datetime as dt
import plotly.express as px
import os 

# --- Global Configuration ---
st.set_page_config(layout="wide", page_title="Market Dashboard")
start = dt.date(year=2010, month=1, day=1)
end = dt.date.today()

# --- Dictionaries ---
indices = {
    "^GSPC" : "S&P500", "^IXIC" : "NASDAQ", "^FTSE" : "FTSE100", "^FCHI" : "CAC40",
    "^HSI" : "HANG SENG", "^BVSP" : "BOVESPA", "^N225" : "NIKKEI225", "^STOXX" : "STOXX600",
    "^GDAXI" : "DAX", "^NYA" : "NYSE", "^MXX" : "IPC Mexico", "^SSMI" : "SMI",
    "^IBEX" : "IBEX35", "^AEX" : "AEX", "^NSEI" : "NIFTY50"
}
sectors = {
    "^YH310" : "Industrials", "^YH101" : "Materials", "^YH103" : "Financial",
    "^YH207" : "Utilities", "^YH102" : "Cons. Discr.", "^YH205" : "Cons. Stap.",
    "^YH309" : "Energy", "^YH206" : "Health Care", "^YH311" : "Technology",
    "^YH308" : "Communication", "^YH104" : "Real Estate"
}
forex = {
    "EURUSD=X" : "EUR", "JPYUSD=X" : "JPY", "GBPUSD=X" : "GBP", "AUDUSD=X" : "AUD",
    "NZDUSD=X" : "NZD", "CADUSD=X" : "CAD", "CHFUSD=X" : "CHF", "CNYUSD=X" : "CNY",
    "MXNUSD=X" : "MXN"
}
commodities = {
    "GC=F" : "Gold", "SI=F" : "Silver", "CL=F" : "Crude Oil", "PL=F" : "Platinum",
    "ALI=F": "Aluminium", "HG=F" : "Copper", "PA=F" : "Palladium", "NG=F" : "Natural Gas", 
    "ZC=F" : "Corn", "ZO=F" : "Oat", "ZS=F" : "Soybean", "ZW=F" : "Wheat", 
    "CC=F" : "Cocoa", "KC=F" : "Coffee", "CT=F" : "Cotton", "OJ=F" : "Orange Juice",
    "SB=F" : "Sugar", "RB=F" : "Gasoline", "HO=F" : "Heating oil", "DA=F" : "Milk"
}
USbonds = {
    "DGS1MO" : "1m T-Bill", "DGS3MO" : "3m T-Bill", "DGS6MO" : "6m T-Bill",
    "DGS1" : "1y T-Bond", "DGS2" : "2y T-Bond", "DGS3" : "3y T-Bond", "DGS5" : "5y T-Bond",
    "DGS7" : "7y T-Bond", "DGS10" : "10y T-Bond", "DGS20" : "20y T-Bond", "DGS30" : "30y T-Bond"
}
TenYbonds = {
    "IRLTLT01DEM156N": "Germany", "IRLTLT01FRM156N": "France", "IRLTLT01ITM156N": "Italy",
    "IRLTLT01GBM156N": "United Kingdom", "IRLTLT01JPM156N": "Japan", "IRLTLT01ESM156N": "Spain",
    "IRLTLT01PTM156N": "Portugal", "IRLTLT01GRM156N": "Greece", "IRLTLT01CAM156N": "Canada",
    "IRLTLT01AUM156N": "Australia", "IRLTLT01CHM156N": "Switzerland", "IRLTLT01USM156N" : "United States"
}
Crypto = {
    "BTC-USD" : "Bitcoin", "ETH-USD" : "Ethereum", "BNB-USD" : "BNB", "XRP-USD" : "Ripple",
    "SOL-USD" : "Solana", "TRX-USD" : "Tron", "DOGE-USD" : "Dogecoin", "ADA-USD" : "Cardano"
}
inflation = {"CPIAUCNS": "US CPI"}
growth = {"GDP": "US GDP"}
fedfund = {"FEDFUNDS": "Fed Fund Rate"}
unr = {"UNRATE": "Unemployment Rate"}
m2 = {"M2SL": "M2 money supply"}
mortgage = {"MORTGAGE30US": "30Y Mortgage Rate"}

# --- SECTION 1: DATA LOADING ENGINE (OPTIMIZED) ---

@st.cache_data(ttl="24h")
def get_batch_data(tickers_dict):
    """Download entire dictionary at once to avoid lags."""
    if not tickers_dict: return pd.DataFrame()
    symbols = list(tickers_dict.keys())
    
    try:
        # Batch download
        df = yf.download(symbols, start=start, end=end, group_by='ticker', progress=False)
        
        # Clean DataFrame reconstruction
        clean_df = pd.DataFrame()
        for ticker, name in tickers_dict.items():
            try:
                if len(symbols) > 1:
                    if ticker in df.columns.levels[0]:
                        series = df[ticker]['Close']
                    else:
                        continue
                else:
                    series = df['Close']
                
                # Timezone cleanup for chart compatibility
                if series.index.tz is not None:
                    series.index = series.index.tz_localize(None)
                    
                clean_df[name] = series
            except KeyError:
                continue
        
        # Explicit float conversion to avoid type errors
        return clean_df.apply(pd.to_numeric, errors='coerce').ffill().bfill()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl="24h")
def get_single_data(ticker):
    """Fetch single asset for backtest (no CSV)."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty: return pd.Series(dtype=float)
        
        # Multi-column or simple handling
        if 'Close' in df.columns:
            s = df['Close']
        else:
            s = df.iloc[:, 0]
            
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return s.astype(float)
    except:
        return pd.Series(dtype=float)

@st.cache_data(ttl="24h")
def fred_table(dictionnary):
    list_fred = []
    for ticker, name in dictionnary.items():
        try:
            data = web.DataReader(ticker, "fred", start)
            data.columns = [name]
            list_fred.append(data)
        except Exception:
            pass
    if not list_fred: return pd.DataFrame()
    return pd.concat(list_fred, axis=1).ffill().bfill()

def normalize(dictionnary):
    # Using optimized loader here
    data_table = get_batch_data(dictionnary)
    if data_table.empty: return pd.DataFrame()
    
    returns = data_table.pct_change().fillna(0)
    normalized_table = (((1+returns).cumprod())-1)*100
    return normalized_table

@st.cache_data(ttl="24h")
def prox_inflation():
    """Optimized in-memory version of proxy inflation calculation."""
    weight_agri = 0.145
    weight_energy = 0.166
    weight_medic = 0.083
    weight_housing = 0.442
    weight_other = 0.164
    
    # Full ticker -> name mapping for algo
    proxy_map = {
        "ZC=F": "Corn", "ZW=F": "Wheat", "ZS=F": "Soybean", "SB=F": "Sugar",
        "CL=F": "Crude Oil", "RB=F": "Gasoline", "HO=F": "Heating Oil",
        "ALI=F": "Aluminium", "HG=F": "Copper", "NG=F": "Natural Gas",
        "CT=F": "Cotton", "PL=F": "Platinum", "PA=F": "Palladium"
    }
    
    # 1. Batch download of all necessary commodities
    raw_data = get_batch_data({k:k for k in proxy_map.keys()}) # Temporarily keep tickers as columns
    if raw_data.empty: return pd.Series(dtype=float)
    
    # 2. Rename columns to match logic (Ticker -> Name)
    raw_data.columns = [proxy_map.get(c, c) for c in raw_data.columns]
    
    # 3. Normalization
    table_proxy_norm = (raw_data / raw_data.iloc[0]) * 100
    table_proxy_norm["Proxy"] = 0.0
    
    # 4. Exact weighting logic
    proxy_config = [
        ("Corn", 0.35 * weight_agri), ("Wheat", 0.25 * weight_agri), ("Soybean", 0.20 * weight_agri), ("Sugar", 0.20 * weight_agri),
        ("Crude Oil", 0.60 * weight_energy), ("Gasoline", 0.30 * weight_energy), ("Heating Oil", 0.10 * weight_energy), 
        ("Aluminium", 0.4 * weight_medic), ("Copper", 0.4 * weight_medic), ("Crude Oil", 0.4 * weight_medic),
        ("Copper", 0.4 * weight_housing), ("Natural Gas", 0.2 * weight_housing), ("Aluminium", 0.2 * weight_housing), 
        ("Cotton", 0.1 * weight_housing), ("Heating Oil", 0.1 * weight_housing),
        ("Copper", 0.27 * weight_other), ("Natural Gas", 0.041 * weight_other), ("Aluminium", 0.153 * weight_other), 
        ("Cotton", 0.162 * weight_other), ("Crude Oil", 0.17 * weight_other),
        ("Platinum", 0.056 * weight_other), ("Palladium", 0.038 * weight_other)
    ]
    
    for name, weight in proxy_config:
        if name in table_proxy_norm.columns:
            table_proxy_norm["Proxy"] += table_proxy_norm[name] * weight
            
    return table_proxy_norm["Proxy"].dropna()

# --- SECTION 2: PLOTTING FUNCTIONS ---

def plot_bonds (dictionnary, title):
    data = fred_table(dictionnary)
    if data.empty:
        st.warning(f"No FRED data for {title}")
        return
    st.subheader(title)
    st.line_chart(data)

def plot_chart (dictionnary, title):
    norm_data = normalize(dictionnary)
    if norm_data.empty:
        st.warning(f"Loading... {title}")
        return
    col1,col2 = st.columns([1,3])
    with col1:
        start_val = norm_data.index.min().date()
        end_val = norm_data.index.max().date()
        start_date, end_date = st.date_input("Period:", value=(start_val, end_val), min_value=start_val, max_value=end_val, key=f"period_{title}")
        assets = st.multiselect("Selected assets:", options=norm_data.columns, default=norm_data.columns.tolist(), key=f"assets_{title}")
        smas = st.multiselect("Moving Averages:", options=[20, 50, 200], default=None, key=f"sma_{title}")
    if not assets: return
    
    filtered_data = period(norm_data[assets], start_date, end_date, type="not none")
    fig_wtih_smas = selected_smas(filtered_data, smas)
    with col2:
        st.subheader(title)
        st.line_chart(fig_wtih_smas)

def plot_second_chart (dictionnary, title):
    table_data = get_batch_data(dictionnary)
    if table_data.empty: return
    col1,col2 = st.columns([1,3])
    with col1:
        start_val = table_data.index.min().date()
        end_val = table_data.index.max().date()
        start_date, end_date = st.date_input("Period:", value=(start_val, end_val), min_value=start_val, max_value=end_val, key=f"period_{title}")
        assets = st.selectbox("Selected assets:", options=table_data.columns, key=f"assets_{title}")
        smas = st.multiselect("Moving Averages:", options=[20, 50, 200], default=None, key=f"sma_{title}")
    
    selection = table_data[[assets]] # Keep dataframe format
    filtered_data = period(selection, start_date, end_date, type="not none")
    fig_wtih_smas = selected_smas(filtered_data, smas)
    with col2:
        st.subheader(title)
        st.line_chart(fig_wtih_smas)

def selected_smas(data, smas):
    if not smas or data.empty: return data
    final_set = data.copy() 
    for col in final_set.columns:
        if "_SMA_" in col: continue
        for sma in smas:
            final_set[f"{col}_SMA_{sma}"] = final_set[col].rolling(window=sma).mean()
    return final_set

def period (selection, start, end, type=None):
    if selection.empty: return selection
    mask = (selection.index.date >= start) & (selection.index.date <= end)
    data = selection.loc[mask].copy()
    
    if data.empty: return data
    if type == "normalize_from_start":
        # Safe normalization
        first = data.iloc[0]
        if isinstance(first, pd.Series): first = first.replace(0, np.nan)
        elif first == 0: first = np.nan
        data = ((data / first) - 1) * 100
    return data

def plot_correl(dictionnary, title):
    # Correlation uses raw price data, not normalized
    price_data = get_batch_data(dictionnary)
    if price_data.empty: return

    col1,col2 = st.columns(2)
    with col2:
        frequency = st.selectbox("Select Frequency:", options=["Daily", "Weekly", "Monthly", "Annually"], key=f"freq_{title}")
    with col1:
        start_val = price_data.index.min().date()
        end_val = price_data.index.max().date()
        start_date, end_date = st.date_input("Period:", value=(start_val, end_val), min_value=start_val, max_value=end_val, key=f"period_{title}")
    
    filtered_prices = period(price_data, start_date, end_date, type="not none")
    if filtered_prices.empty: return

    if frequency == "Annually": returns = filtered_prices.resample("Y").last().pct_change()
    elif frequency == "Monthly": returns = filtered_prices.resample("ME").last().pct_change()
    elif frequency == "Weekly": returns = filtered_prices.resample("W").last().pct_change()
    else: returns = filtered_prices.pct_change()
    
    returns = returns.dropna(how='all')
    if returns.shape[1] < 2:
        st.warning("Not enough data.")
        return
        
    correl_indices = returns.corr(method="pearson").mul(100).round(0)
    fig = px.imshow(correl_indices, text_auto= True, width=800, height=600, title= title, color_continuous_scale="RdYlBu", color_continuous_midpoint=0)
    st.plotly_chart(fig, use_container_width=True)
    
def yield_table(dictionnary):
    data = get_batch_data(dictionnary)
    if data.empty: return pd.DataFrame()
    last = data.iloc[-1]
    
    def safe_iloc(idx):
        if len(data) > idx: return data.iloc[-(idx+1)]
        return data.iloc[0]

    # YTD Safety
    ytd_data = data[data.index.year == dt.date.today().year]
    if not ytd_data.empty:
        ytd_val = (last / ytd_data.iloc[0]) - 1
    else:
        ytd_val = 0.0

    ytable = pd.DataFrame({
        "Daily": ((last/ safe_iloc(1)) - 1),
        "Weekly": ((last/ safe_iloc(5)) - 1),
        "Monthly": ((last/ safe_iloc(21)) - 1),
        "YTD": ytd_val,
        "Yearly": ((last/ safe_iloc(252)) - 1),
        "5Y": ((last/ safe_iloc(252*5)) - 1),
        "10Y": ((last/ safe_iloc(252*10)) - 1)
    }).T.round(4)
    return ytable

def plot_ytable(dictionnary, title):
    df_yields = yield_table(dictionnary)
    if df_yields.empty: return
    
    # Transposition for display (Assets in rows)
    data_display = df_yields.transpose()
    
    st.subheader(title)
    # Simplified and secured display
    try:
        st.dataframe(data_display.style.format("{:.2%}"), height=(int(data_display.shape[0]+1) * 35))
    except Exception:
        st.dataframe(data_display)

def plot_ycurve(dictionnary, title): 
    table_data = fred_table(dictionnary)
    if table_data.empty: return
    data = table_data.iloc[-1]
    df_curve = pd.DataFrame({"Maturity":data.index, "Yield":data.values})
    
    # Filter categories that actually exist
    valid_cats = [x for x in dictionnary.values() if x in df_curve['Maturity'].values]
    
    df_curve['Maturity'] = pd.Categorical(df_curve['Maturity'], categories=valid_cats, ordered=True)
    df_curve = df_curve.sort_values("Maturity").dropna()
    st.subheader(title)
    st.line_chart(df_curve, x="Maturity", y="Yield")
    
def plot_10y(dictionnary, title):
    data = fred_table(dictionnary)
    if data.empty: return
    data = data.iloc[-1].sort_values().to_frame(name="10Y Yields")
    st.subheader(title)
    st.bar_chart(data)
    
def plot_macro (dictionnary1, period, title):
    data = fred_table(dictionnary1)
    if data.empty: return
    data = data.pct_change(periods=period)*100
    st.subheader(title)
    st.line_chart(data)

def plot_interest (dictionnary, title):
    data = fred_table(dictionnary)
    if data.empty: return
    st.subheader(title)
    st.line_chart(data)

# --- SECTION 3: BACKTEST CORE (CORRECTED & SECURED) ---

@st.cache_data(ttl="24h")
def table_analysis(asset, sma_short, sma_long):
    # 1. Identify ticker
    ticker = None
    for d in [indices, sectors, commodities]: # Search for ticker
        for k, v in d.items():
            if v == asset: ticker = k
    if not ticker: ticker = asset # Case where ticker is passed directly
        
    # 2. Fetch data
    df = get_single_data(ticker)
    growth = get_single_data("^GSPC")
    
    if df.empty or growth.empty: return pd.DataFrame()
    
    df = df.pct_change()*100
    growth = growth.pct_change()*100
    
    # 3. Fetch proxy inflation (from cache)
    infl = prox_inflation()

    # 4. SECURE DATAFRAME PREPARATION (BUG FIX)
    # Force conversion to DataFrame with CORRECT column name
    if isinstance(df, pd.Series):
        df = df.to_frame(name=asset)
    else:
        df.columns = [asset]

    if isinstance(growth, pd.Series):
        growth = growth.to_frame(name="SP500")
    else:
        growth.columns = ["SP500"]

    if isinstance(infl, pd.Series):
        infl = infl.to_frame(name="Proxy")
    else:
        infl.columns = ["Proxy"]

    # 5. Merge and Calculations
    table = pd.concat([df, infl, growth], axis=1).dropna()
    
    # Now table["SP500"] definitely exists
    table["Proxy"] = ((table["Proxy"] / table["Proxy"].iloc[0]) - 1) * 100
    table["SP500 cum"] = (((1+ table["SP500"]/100).cumprod())-1)*100
    table[f"{asset} cum"] = (((1+ table[asset]/100).cumprod())-1)*100
    
    table[f"I SMA {sma_short}"] = table["Proxy"].rolling(sma_short).mean()
    table[f"I SMA {sma_long}"] = table["Proxy"].rolling(sma_long).mean()
    table[f"G SMA {sma_short}"] = table["SP500 cum"].rolling(sma_short).mean()
    table[f"G SMA {sma_long}"] = table["SP500 cum"].rolling(sma_long).mean()
    
    table["G SMA Diff %"] = (table[f"G SMA {sma_short}"] - table[f"G SMA {sma_long}"]) 
    table["I SMA Diff %"] = (table[f"I SMA {sma_short}"] - table[f"I SMA {sma_long}"])
    
    return table.dropna()

def chart_backtest(asset, title, sma_short, sma_long):
    df = table_analysis(asset, sma_short, sma_long)
    if df.empty: 
        st.error("No data for backtest.")
        return
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[f"{asset} cum"], name=asset, yaxis="y1"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SP500 cum"], name="Growth Proxy", yaxis="y1"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Proxy"], name="Inflation Proxy", yaxis="y2"))
    fig.add_trace(go.Scatter(x=df.index, y=df[f"G SMA {sma_short}"], name=f"G SMA {sma_short}", yaxis="y1", line=dict(dash='dot', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df[f"G SMA {sma_long}"], name=f"G SMA {sma_long}", yaxis="y1", line=dict(dash='dot', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df[f"I SMA {sma_short}"], name=f"I SMA {sma_short}", yaxis="y2", line=dict(dash='dot', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df[f"I SMA {sma_long}"], name=f"I SMA {sma_long}", yaxis="y2", line=dict(dash='dot', width=1)))
    fig.update_layout(
        title=title,
        yaxis=dict(title=f"{asset} Returns (%)"),
        yaxis2=dict(title="Proxy Inflation (%)", overlaying="y", side="right"),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

def period_analysis(asset,period,sma_short,sma_long):
    table = table_analysis(asset,sma_short,sma_long) 
    if table.empty: return pd.DataFrame(), "No Data"
    
    if period == "whole period":
        data = table
        title = "during whole period"
    elif period == "increasing inflation":
        data = table[(table[f"G SMA {sma_short}"] > table[f"G SMA {sma_long}"]) & (table[f"I SMA {sma_short}"] > table[f"I SMA {sma_long}"])]
        title = f"when Inflation & Growth"
    elif period == "decreasing inflation":
        data = table[(table[f"G SMA {sma_short}"] < table[f"G SMA {sma_long}"]) & (table[f"I SMA {sma_short}"] > table[f"I SMA {sma_long}"])]
        title = f"when Inflation & Degrowth"
    elif period == "increasing deflation":
        data = table[(table[f"G SMA {sma_short}"] > table[f"G SMA {sma_long}"]) & (table[f"I SMA {sma_short}"] < table[f"I SMA {sma_long}"])]
        title = f"when Deflation & Growth"
    elif period == "decreasing deflation":
        data = table[(table[f"G SMA {sma_short}"] < table[f"G SMA {sma_long}"]) & (table[f"I SMA {sma_short}"] < table[f"I SMA {sma_long}"])]
        title = f"when Deflation & Degrowth"
    return data, title

def plot_anlysis(asset,period,sma_short,sma_long):
    data,title = period_analysis(asset,period,sma_short,sma_long)
    if data.empty or len(data) < 5: 
        st.info(f"Not enough data: {title}")
        return
    
    data[asset] = pd.to_numeric(data[asset], errors='coerce').dropna()
    fig = px.histogram(
        data,
        x = data[asset],
        nbins=200,
        histnorm="probability density",
        title =f"{asset} returns {title}",
        labels={asset:"Returns", "count":"Probability"}
    )
    mean = data[asset].mean()
    std = data[asset].std()
    try:
        x_vals = np.linspace(data[asset].min(),data[asset].max(),500)
        y_vals = stats.norm.pdf(x_vals, mean, std)
        fig.add_trace(go.Scatter(x=x_vals,y=y_vals,mode='lines',name="Normal fit"))
    except Exception: pass 
    
    st.plotly_chart(fig, use_container_width=True)

def plot_scatter(asset, period, sma_short, sma_long):
    data, title = period_analysis(asset, period, sma_short, sma_long)
    if data.empty: return
    
    fig = px.scatter(
        data, 
        x = "G SMA Diff %",
        y = "I SMA Diff %",
        color = asset, 
        color_continuous_scale= "Portland",
        color_continuous_midpoint=0,
        size = abs(data[asset]),
        title =f"{asset} returns {title}",
        height=600
    )
    fig.add_shape(type="line", x0=0, x1=0, y0=data["I SMA Diff %"].min(), y1=data["I SMA Diff %"].max(),
                  line=dict(color="white", dash="dash"))
    fig.add_shape(type="line", y0=0, y1=0, x0=data["G SMA Diff %"].min(), x1=data["G SMA Diff %"].max(),
                  line=dict(color="white", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

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
    full_table = table_analysis(asset,sma_short,sma_long)
    if full_table.empty: return
    
    len_full_table = len(full_table)
    for period in list_periods:
        df,title = period_analysis(asset,period,sma_short,sma_long)
        if df.empty:
             for l in [Returns, Volatilities, Annualized_Means, Max_Drawdowns, Sharpe, Skewness, Kurtosis]: l.append("N/A")
             Time.append("0%")
             continue
             
        df_returns = df[asset] 
        cum = (1 + (df_returns / 100)).cumprod()
        Returns.append(f"{(cum.iloc[-1] - 1)*100:.2f}%")
        vol_ann = df_returns.std() * np.sqrt(252)
        Volatilities.append(f"{vol_ann:.2f}%")
        mean_ann = df_returns.mean() * 252
        Annualized_Means.append(f"{mean_ann:.2f}%")
        Max_Drawdowns.append(f"{df_returns.min():.2f}%")
        sharpe_ratio = (mean_ann - 2) / vol_ann if vol_ann > 0 else 0
        Sharpe.append(f"{sharpe_ratio:.2f}")
        Skewness.append(f"{stats.skew(df_returns,axis=0):.2f}")
        Kurtosis.append(f"{stats.kurtosis(df_returns,axis=0):.2f}")
        time_pct = (len(df) / len_full_table) * 100 if len_full_table > 0 else 0
        Time.append(f"{time_pct:.2f}%")
        
    data = pd.DataFrame({
        "Return (Total)":Returns,
        "Volatility (Ann.)":Volatilities,
        "Annualized Mean":Annualized_Means,
        "Max Daily Loss":Max_Drawdowns,
        "Sharpe (rf=2%)":Sharpe,
        "Skewness":Skewness,
        "Kurtosis":Kurtosis,
        "Time percentage":Time
        }).T
    data.columns = ["Whole Period", "Infl & Growth", "Infl & Degrowth", "Defl & Growth", "Defl & Degrowth"]
    st.subheader("Table of Metrics")
    st.dataframe(data)

# --- SECTION 4: MAIN WORKFLOW ---
pages = ["Market charts", "Bond Market", "Macro", "Backtest"]
st.sidebar.header("SUMMARY")
selected_page = st.sidebar.selectbox("Page Selection:", options=pages, key="Page selection")

if selected_page == "Market charts":
    st.title("🏙️ Market Dashboard")

    tab_indices, tab_sectors, tab_commodities, tab_forex, tab_crypto = st.tabs([
        "Indices", "Sectors", "Commodities", "Forex", "Crypto"
    ])
    with tab_indices:
        st.header("Major Indices")
        plot_chart(indices, "Major Indices Normalized Returns")
        st.markdown("""<hr>""", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1: plot_ytable(indices, "Indices Yield Table")
        with col2: plot_correl(indices, "Indices Correlation Matrix")
    with tab_sectors:
        st.header("S&P 500 Sectors")
        plot_chart(sectors, "SP500 Sectors Normalized Returns")
        st.markdown("""<hr>""", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            plot_ytable(sectors, "Indices Yield Table")
        with col2: plot_correl(sectors, "Sectors Correlation")
    with tab_commodities:
        st.header("Major Commodities")
        plot_chart(commodities, "Major Commodities Normalized Returns")
        st.markdown("""<hr>""", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1: plot_ytable(commodities, "Yield Table")
        with col2: plot_correl(commodities, "Commodities Correlation")
    with tab_forex:
        st.header("Major Currencies")
        col1, col2 = st.columns([0.6, 0.4])
        with col1: plot_chart(forex, "Major Currencies Normalized Returns")
        with col2: plot_second_chart(forex, "Price of currency exchange")
        st.markdown("""<hr>""", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1: plot_ytable(forex, "Yield Table")
        with col2: plot_correl(forex, "Correlation")
    with tab_crypto:
        st.header("Major Cryptocurrencies")
        col1, col2 = st.columns([0.6, 0.4])
        with col1: plot_chart(Crypto, "Major Cryptos Normalized Returns")
        with col2: plot_second_chart(Crypto, "Price of Crypto exchange")
        st.markdown("""<hr>""", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1: plot_ytable(Crypto, "Yield Table")
        with col2: plot_correl(Crypto, "Crypto Correlation")

elif selected_page == "Bond Market":
    st.title("Bond Market")
    st.markdown("""<hr>""", unsafe_allow_html=True) 
    with st.container():
        plot_bonds(USbonds, "US Bonds yield")
        st.markdown("""<hr>""", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1: plot_10y(TenYbonds, "10Y yield")
        with col2: plot_ycurve(USbonds, "US Yield Curve")
    with st.container():
        st.markdown("""<hr>""", unsafe_allow_html=True)
        plot_bonds(TenYbonds,"10Y yields historical chart")

elif selected_page == "Macro":
    st.title("US Macro")
    st.markdown("""<hr>""", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        plot_macro(growth,4,"US GDP (Yearly Change %)")
        st.markdown("""<hr>""", unsafe_allow_html=True)
        plot_macro(inflation,12,"US CPI Index (Year-over-Year %)")
        st.markdown("""<hr>""", unsafe_allow_html=True)
        plot_interest(m2,"M2 money supply")
    with col2:
        plot_interest(unr,"Unemployment Rate (%)")
        st.markdown("""<hr>""", unsafe_allow_html=True)
        plot_interest(fedfund,"Fed Fund Rate (%)")
        st.markdown("""<hr>""", unsafe_allow_html=True)
        plot_interest(mortgage,"30Y Mortgage Rate (%)")

elif selected_page == "Backtest":
    st.title("Backtest")
    st.markdown("""<hr>""", unsafe_allow_html=True)
    list_assets_backtest = [indices["^GSPC"]] + list(sectors.values()) + [commodities["GC=F"]] + [commodities["CL=F"]]
    
    with st.container():
        selection = st.selectbox("Select asset for analysis:",options=list_assets_backtest, index=0)
        col1, col2 = st.columns(2)
        with col1:
            x = st.slider("Short SMA (Inflation):",step=1,min_value=1,max_value=365, value=20)
            a = st.slider("Short SMA (Growth):",step=1,min_value=1,max_value=365, value=20)
        with col2:
            y = st.slider("Long SMA (Inflation):",step=1,min_value=2,max_value=365, value=100)
            b = st.slider("Long SMA (Growth):",step=1,min_value=2,max_value=365, value=100)
            
        chart_backtest(selection, f"{selection} Normalized Returns & Proxy",x,y)
        st.markdown("""<hr>""", unsafe_allow_html=True)
        st.header("Distribution Analysis")
        plot_scatter(selection, "whole period",x,y)
        col1,col2 = st.columns(2)
        with col1: plot_anlysis(selection, "whole period",x,y)
        with col2: statistics(selection,x,y)
        st.markdown("""<hr>""", unsafe_allow_html=True)
        st.header("Regime Analysis")
        col1,col2 = st.columns(2)
        with col1:
            plot_anlysis(selection, "increasing inflation",x,y)
            plot_anlysis(selection, "decreasing inflation",x,y)
        with col2:
            plot_anlysis(selection, "increasing deflation",x,y)
            plot_anlysis(selection, "decreasing deflation",x,y)
