# 📈 Macro-Financial Market Dashboard & Strategy Backtester

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![Finance](https://img.shields.io/badge/Finance-Quantitative-green)
![Data](https://img.shields.io/badge/Data-Yahoo%20%7C%20FRED-orange)

### marketdashboard--mathissilaghi-rkfxyxasphj5n9ltbfaa2s

## 🎓 Context & Objective
This project was developed as part of the **MSc Financial Markets & Investments** curriculum at **SKEMA Business School**.

The primary objective was to build a bridge between **macroeconomic theory** and **quantitative trading strategies**. While traditional dashboards display static data, this tool integrates a proprietary **regime-based backtesting engine** to simulate how different asset classes perform under varying growth and inflation cycles.

## 🚀 Key Features

### 1. 🏙️ comprehensive Market Monitor
A unified interface to track real-time performance across:
* **Global Indices:** Normalized returns comparison (S&P500, CAC40, Nikkei 225, etc.).
* **Sector Rotation:** Tracking the 11 GICS sectors to identify defensive vs. cyclical flows.
* **Commodities & Forex:** Correlation matrices to detect inter-market dependencies.
* **Cryptocurrencies:** Volatility and price tracking for digital assets.

### 2. 🏦 Fixed Income & Yield Curve Analysis
* **Dynamic Yield Curve:** Visualizes the US Treasury term structure (1M to 30Y) to detect recessionary signals (inversions).
* **Sovereign Spreads:** Monitors 10-Year yields across G7 nations to assess global risk premiums.

### 3. 🧪 The Proprietary Backtester (Regime Analysis)
This is the core innovation of the dashboard. It moves beyond simple price analysis to **macro-regime detection**.

#### A. The Inflation Proxy Construction
Since official CPI data is released monthly with a lag, the model constructs a **high-frequency Daily Inflation Proxy** based on a weighted basket of commodities:
* **Energy (16.6%):** Crude Oil, Gasoline, Heating Oil.
* **Housing/Construction (44.2%):** Copper, Lumber (proxied), Natural Gas.
* **Agriculture (14.5%):** Corn, Wheat, Soybeans, Sugar.
* **Other Ind. Metals (24.7%):** Aluminum, Palladium, Platinum.

#### B. Signal Generation: Dual SMA Crossover
The strategy identifies regimes by comparing Short-Term (Fast) vs. Long-Term (Slow) Simple Moving Averages (SMA) on two indicators:
1.  **Growth:** S&P 500 Price Action.
2.  **Inflation:** The Proprietary Commodity Proxy.

#### C. The Four Macro Quadrants
The backtester segments time into four distinct economic environments:
1.  **Reflation (📈 Growth, 📈 Inflation):** "Goldilocks" or overheating economy.
2.  **Stagflation (📉 Growth, 📈 Inflation):** The most challenging environment for equities.
3.  **Deflationary Bust (📉 Growth, 📉 Inflation):** Recessionary crash.
4.  **Disinflationary Boom (📈 Growth, 📉 Inflation):** Productivity-led growth.

## 🛠️ Technical Architecture & Optimization

The application is built on **Python** using **Streamlit** for the frontend. To handle heavy financial datasets without latency, specific optimizations were implemented:

* **Batch Processing:** Uses `yfinance` grouped downloads to fetch entire asset classes in single API calls rather than iterative loops, reducing load time by ~90%.
* **Session State Management:** Implements "Lazy Loading" patterns. Heavy datasets (like the S&P 500 constituents treemap) are only computed when the user specifically navigates to the relevant tab.
* **Smart Caching:** Utilizes `@st.cache_data` with a 24h Time-To-Live (TTL) to store processed dataframes, preventing redundant computations during user interaction.

## ⚙️ Installation & Usage

### Prerequisites
* Python 3.8 or higher.
* An active internet connection (for Yahoo Finance & FRED APIs).

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Market-Dashboard-Project.git](https://github.com/YOUR_USERNAME/Market-Dashboard-Project.git)
    cd Market-Dashboard-Project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the Dashboard:**
    ```bash
    streamlit run MarketDashboard.py
    ```

## 🔮 Future Roadmap
To further enhance this tool, the following features are currently under consideration:
* **Portfolio Optimization:** Implementation of Markowitz Mean-Variance optimization for the "Backtest" selection.
* **Risk Metrics:** Addition of Value at Risk (VaR) and Expected Shortfall (CVaR).
* **Sentiment Analysis:** Integration of NLP on financial news to add a third dimension to the regime detection.

## ⚠️ Disclaimer
This tool is intended for educational and research purposes. The strategies backtested are theoretical and do not account for transaction costs, slippage, or liquidity constraints.

---
**Author:** Mathis Silaghi  
**Master in Financial Markets & Investments** **SKEMA Business School**
