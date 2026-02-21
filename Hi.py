import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ==============================================================================
# PAGE CONFIG & STATE
# ==============================================================================
st.set_page_config(page_title="NSE Swing Strategy Dashboard", layout="wide", page_icon="📈")
st.title("📈 NSE Swing Trading vs Buy & Hold Dashboard")

# ==============================================================================
# INDIAN MARKET TAX & COST CONSTANTS (2025-2026 REGIME)
# ==============================================================================
BROKERAGE_PER_ORDER = 20.0
STT_DELIVERY_RATE = 0.001
EXCHANGE_TXN_RATE = 0.0000325
SEBI_TURNOVER_RATE = 0.000001
STAMP_DUTY_RATE = 0.00015
GST_RATE = 0.18

def calculate_trade_economics(buy_price, sell_price, quantity):
    buy_turnover = buy_price * quantity
    sell_turnover = sell_price * quantity
    total_turnover = buy_turnover + sell_turnover
    
    brokerage_total = BROKERAGE_PER_ORDER * 2
    stt_charge = total_turnover * STT_DELIVERY_RATE
    exchange_charge = total_turnover * EXCHANGE_TXN_RATE
    sebi_charge = total_turnover * SEBI_TURNOVER_RATE
    stamp_duty = buy_turnover * STAMP_DUTY_RATE
    
    taxable_services = brokerage_total + exchange_charge + sebi_charge
    gst_charge = taxable_services * GST_RATE
    
    total_frictional_cost = brokerage_total + stt_charge + exchange_charge + sebi_charge + stamp_duty + gst_charge
    gross_profit = sell_turnover - buy_turnover
    net_profit = gross_profit - total_frictional_cost
    
    return {'gross_pnl': gross_profit, 'net_pnl': net_profit, 'total_costs': total_frictional_cost}

# ==============================================================================
# TECHNICAL INDICATOR COMPUTATION
# ==============================================================================
@st.cache_data(show_spinner=False)
def compute_technical_features(df):
    df = df.copy()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['ATR'] = tr.rolling(14).mean()
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
    
    return df

# ==============================================================================
# FUNDAMENTAL SCREENING
# ==============================================================================
@st.cache_data(show_spinner=False)
def execute_fundamental_screen(ticker):
    try:
        info = yf.Ticker(ticker).info
        
        if not info:
            return True, "Yahoo API returned empty. Bypassing fundamental screen."

        if info.get('trailingEps', 0) <= 0: return False, "Negative/Zero EPS"
        if info.get('debtToEquity', 100) > 50: return False, "High Debt-to-Equity"
        return True, "Passed"
        
    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
            return True, "Yahoo Rate Limit hit. Bypassing fundamental screen to allow backtest."
        return False, f"Data Error: {str(e)}"

# ==============================================================================
# STRATEGY & BUY-AND-HOLD ENGINE
# ==============================================================================
@st.cache_data(show_spinner=False)
def run_backtests(ticker, start_date, end_date, initial_capital):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty: return None, None, None
    
    # ---------------------------------------------------------
    # FIX FOR YFINANCE MULTI-INDEX UPDATE
    # ---------------------------------------------------------
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data = compute_technical_features(data).dropna()
    
    # Buy & Hold Calculation
    bnh_qty = int(initial_capital / data['Close'].iloc[0])
    data['Buy_Hold_Equity'] = initial_capital - (bnh_qty * data['Close'].iloc[0]) + (bnh_qty * data['Close'])
    bnh_economics = calculate_trade_economics(data['Close'].iloc[0], data['Close'].iloc[-1], bnh_qty)
    bnh_final = initial_capital + bnh_economics['net_pnl']
    
    # Swing Strategy Calculation
    capital = initial_capital
    position_qty, entry_price, stop_loss, take_profit = 0, 0, 0, 0
    trade_log, daily_equity = [], []
    
    for i in range(len(data)):
        row, prev_row = data.iloc[i], data.iloc[i-1] if i > 0 else data.iloc[i]
        
        if position_qty == 0:
            # UNCHAINED COMPARISON FIX
            if (row['Close'] > row['EMA_50']) and (row['EMA_50'] > row['EMA_200']) and \
               (row['Volume_MA_20'] > 1000000) and (prev_row['RSI'] < 45) and \
               (row['MACD'] > row['MACD_signal'] and prev_row['MACD'] <= prev_row['MACD_signal']):
               
                entry_price = row['Close']
                position_qty = int((capital * 0.95) / entry_price)
                if position_qty > 0:
                    stop_loss = entry_price - (2 * row['ATR'])
                    take_profit = entry_price + (3 * row['ATR'])
        
        elif position_qty > 0:
            exit_price = 0
            if row['Low'] <= stop_loss: exit_price = stop_loss
            elif row['High'] >= take_profit: exit_price = take_profit
            elif row['Close'] < row['EMA_50']: exit_price = row['Close']
            
            if exit_price > 0:
                eco = calculate_trade_economics(entry_price, exit_price, position_qty)
                capital += eco['net_pnl']
                trade_log.append({
                    'Exit_Date': data.index[i],
                    'Net_PnL': eco['net_pnl'],
                    'Total_Costs': eco['total_costs']
                })
                position_qty = 0
                
        # Mark-to-market daily equity tracking
        current_eq = capital
        if position_qty > 0:
            current_eq += (row['Close'] - entry_price) * position_qty
        daily_equity.append(current_eq)
        
    data['Strategy_Equity'] = daily_equity
    return data, pd.DataFrame(trade_log), bnh_final

# ==============================================================================
# UI RENDERING
# ==============================================================================
st.sidebar.header("⚙️ Strategy Parameters")
default_tickers = ["COCHINSHIP.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "TATAMOTORS.NS"]
ticker = st.sidebar.selectbox("Select or Type NSE Ticker", default_tickers, index=0, format_func=lambda x: x)
custom_ticker = st.sidebar.text_input("Or enter custom ticker (e.g., ZOMATO.NS):")
if custom_ticker: ticker = custom_ticker.upper()

start_date = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2026, 2, 21))
initial_cap = st.sidebar.number_input("Initial Capital (₹)", value=100000, step=10000)

if st.sidebar.button("Run Backtest", type="primary"):
    with st.spinner(f"Analyzing {ticker}..."):
        passed, msg = execute_fundamental_screen(ticker)
        
        if not passed:
            st.error(f"**Fundamental Screen Failed for {ticker}**: {msg}")
            st.warning("The strategy halts here to protect capital based on fundamental rules.")
        else:
            if "Bypassing" in msg:
                st.warning(f"⚠️ {msg}")
            else:
                st.success(f"**{ticker}** passed fundamental screening.")
                
            data, trades, bnh_final = run_backtests(ticker, start_date, end_date, initial_cap)
            
            if data is None or data.empty:
                st.error("No price data found. Please check the ticker symbol and dates.")
            else:
                final_strat_cap = data['Strategy_Equity'].iloc[-1]
                strat_return = ((final_strat_cap / initial_cap) - 1) * 100
                bnh_return = ((bnh_final / initial_cap) - 1) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Strategy Final Capital", f"₹{final_strat_cap:,.0f}", f"{strat_return:.1f}%")
                col2.metric("Buy & Hold Final Capital", f"₹{bnh_final:,.0f}", f"{bnh_return:.1f}%", 
                            delta_color="off" if bnh_return > strat_return else "normal")
                col3.metric("Strategy Total Trades", len(trades) if not trades.empty else 0)
                col4.metric("Strategy Win vs B&H", f"₹{(final_strat_cap - bnh_final):,.0f}")
                
                st.markdown("### 📊 Equity Curve Comparison")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Strategy_Equity'], mode='lines', name='Swing Strategy', line=dict(color='#00ff00', width=2)))
                fig.add_trace(go.Scatter(x=data.index, y=data['Buy_Hold_Equity'], mode='lines', name='Buy & Hold', line=dict(color='#888888', width=2, dash='dot')))
                fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Portfolio Value (₹)", template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                if not trades.empty:
                    st.markdown("### 💸 Frictional Cost Drag (Strategy)")
                    total_costs = trades['Total_Costs'].sum()
                    st.info(f"Total Taxes & Brokerage Paid: **₹{total_costs:,.0f}**")
                else:
                    st.info("No trades executed during this period based on your strategy rules.")
    
    total_frictional_cost = brokerage_total + stt_charge + exchange_charge + sebi_charge + stamp_duty + gst_charge
    gross_profit = sell_turnover - buy_turnover
    net_profit = gross_profit - total_frictional_cost
    
    return {'gross_pnl': gross_profit, 'net_pnl': net_profit, 'total_costs': total_frictional_cost}

# ==============================================================================
# TECHNICAL INDICATOR COMPUTATION
# ==============================================================================
@st.cache_data(show_spinner=False)
def compute_technical_features(df):
    df = df.copy()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['ATR'] = tr.rolling(14).mean()
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
    
    return df

# ==============================================================================
# FUNDAMENTAL SCREENING (With yfinance auto-handling)
# ==============================================================================
@st.cache_data(show_spinner=False)
def execute_fundamental_screen(ticker):
    try:
        # We now let yfinance handle the connection natively
        info = yf.Ticker(ticker).info
        
        if not info:
            return True, "Yahoo API returned empty. Bypassing fundamental screen."

        if info.get('trailingEps', 0) <= 0: return False, "Negative/Zero EPS"
        if info.get('debtToEquity', 100) > 50: return False, "High Debt-to-Equity"
        return True, "Passed"
        
    except Exception as e:
        error_str = str(e).lower()
        # If Yahoo still acts up, we gracefully bypass the screen so your chart still loads!
        if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
            return True, "Yahoo Rate Limit hit. Bypassing fundamental screen to allow backtest."
        return False, f"Data Error: {str(e)}"

# ==============================================================================
# STRATEGY & BUY-AND-HOLD ENGINE
# ==============================================================================
@st.cache_data(show_spinner=False)
def run_backtests(ticker, start_date, end_date, initial_capital):
    # Removed the custom session here as well
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty: return None, None, None
    
    data = compute_technical_features(data).dropna()
    
    # Buy & Hold Calculation
    bnh_qty = int(initial_capital / data['Close'].iloc[0])
    data['Buy_Hold_Equity'] = initial_capital - (bnh_qty * data['Close'].iloc[0]) + (bnh_qty * data['Close'])
    bnh_economics = calculate_trade_economics(data['Close'].iloc[0], data['Close'].iloc[-1], bnh_qty)
    bnh_final = initial_capital + bnh_economics['net_pnl']
    
    # Swing Strategy Calculation
    capital = initial_capital
    position_qty, entry_price, stop_loss, take_profit = 0, 0, 0, 0
    trade_log, daily_equity = [], []
    
    for i in range(len(data)):
        row, prev_row = data.iloc[i], data.iloc[i-1] if i > 0 else data.iloc[i]
        
        if position_qty == 0:
            if (row['Close'] > row['EMA_50'] > row['EMA_200']) and (row['Volume_MA_20'] > 1000000) and \
               (prev_row['RSI'] < 45) and (row['MACD'] > row['MACD_signal'] and prev_row['MACD'] <= prev_row['MACD_signal']):
                entry_price = row['Close']
                position_qty = int((capital * 0.95) / entry_price)
                if position_qty > 0:
                    stop_loss = entry_price - (2 * row['ATR'])
                    take_profit = entry_price + (3 * row['ATR'])
        
        elif position_qty > 0:
            exit_price = 0
            if row['Low'] <= stop_loss: exit_price = stop_loss
            elif row['High'] >= take_profit: exit_price = take_profit
            elif row['Close'] < row['EMA_50']: exit_price = row['Close']
            
            if exit_price > 0:
                eco = calculate_trade_economics(entry_price, exit_price, position_qty)
                capital += eco['net_pnl']
                trade_log.append({
                    'Exit_Date': data.index[i],
                    'Net_PnL': eco['net_pnl'],
                    'Total_Costs': eco['total_costs']
                })
                position_qty = 0
                
        # Mark-to-market daily equity tracking
        current_eq = capital
        if position_qty > 0:
            current_eq += (row['Close'] - entry_price) * position_qty
        daily_equity.append(current_eq)
        
    data['Strategy_Equity'] = daily_equity
    return data, pd.DataFrame(trade_log), bnh_final

# ==============================================================================
# UI RENDERING
# ==============================================================================
st.sidebar.header("⚙️ Strategy Parameters")
default_tickers = ["COCHINSHIP.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "TATAMOTORS.NS"]
ticker = st.sidebar.selectbox("Select or Type NSE Ticker", default_tickers, index=0, format_func=lambda x: x)
custom_ticker = st.sidebar.text_input("Or enter custom ticker (e.g., ZOMATO.NS):")
if custom_ticker: ticker = custom_ticker.upper()

start_date = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2026, 2, 21))
initial_cap = st.sidebar.number_input("Initial Capital (₹)", value=100000, step=10000)

if st.sidebar.button("Run Backtest", type="primary"):
    with st.spinner(f"Analyzing {ticker}..."):
        passed, msg = execute_fundamental_screen(ticker)
        
        if not passed:
            st.error(f"**Fundamental Screen Failed for {ticker}**: {msg}")
            st.warning("The strategy halts here to protect capital based on fundamental rules.")
        else:
            if "Bypassing" in msg:
                st.warning(f"⚠️ {msg}")
            else:
                st.success(f"**{ticker}** passed fundamental screening.")
                
            data, trades, bnh_final = run_backtests(ticker, start_date, end_date, initial_cap)
            
            if data is None or data.empty:
                st.error("No price data found. Please check the ticker symbol and dates.")
            else:
                final_strat_cap = data['Strategy_Equity'].iloc[-1]
                strat_return = ((final_strat_cap / initial_cap) - 1) * 100
                bnh_return = ((bnh_final / initial_cap) - 1) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Strategy Final Capital", f"₹{final_strat_cap:,.0f}", f"{strat_return:.1f}%")
                col2.metric("Buy & Hold Final Capital", f"₹{bnh_final:,.0f}", f"{bnh_return:.1f}%", 
                            delta_color="off" if bnh_return > strat_return else "normal")
                col3.metric("Strategy Total Trades", len(trades) if not trades.empty else 0)
                col4.metric("Strategy Win vs B&H", f"₹{(final_strat_cap - bnh_final):,.0f}")
                
                st.markdown("### 📊 Equity Curve Comparison")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Strategy_Equity'], mode='lines', name='Swing Strategy', line=dict(color='#00ff00', width=2)))
                fig.add_trace(go.Scatter(x=data.index, y=data['Buy_Hold_Equity'], mode='lines', name='Buy & Hold', line=dict(color='#888888', width=2, dash='dot')))
                fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Portfolio Value (₹)", template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                if not trades.empty:
                    st.markdown("### 💸 Frictional Cost Drag (Strategy)")
                    total_costs = trades['Total_Costs'].sum()
                    st.info(f"Total Taxes & Brokerage Paid: **₹{total_costs:,.0f}**")
