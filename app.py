import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from prophet.plot import plot_plotly

# [PASTE YOUR FINAL APP CODE HERE - The one with Strategy Simulator]
# (I am skipping the full paste here to save space, but you must paste your full code)

#=======================Part 01 ================================================================

# 1. Load Data
df = pd.read_csv('sales_data.csv') # Make sure this matches your uploaded filename

# Calculate 'Sales' column as it's missing
df['Sales'] = df['Price'] * df['Units_Sold']

# 2. Prepare for Prophet (Rename columns to 'ds' and 'y')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
prophet_df = df[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})

# 3. Train
m = Prophet()
m.fit(prophet_df)

# 4. Save the model to a file
joblib.dump(m, 'forecast_model.pkl')
print("✅ Model trained and saved!")



#====================================Part 02 ======================================================

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="GrowthIQ | AI Sales Forecaster",
    page_icon="G",
    layout="wide"
)

# --- 2. LOAD BRAIN ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('forecast_model.pkl')
    except:
        return None

model = load_model()

# --- 3. SIDEBAR (CONTROLS) ---
st.sidebar.title("🎛️ Control Panel")

# A. Currency Converter
st.sidebar.subheader("💱 Currency Settings")
currency_mode = st.sidebar.radio("Display Revenue In:", ["USD ($)", "INR (₹)"], index=0)
exchange_rate = 84.0 if currency_mode == "INR (₹)" else 1.0
currency_symbol = "₹" if currency_mode == "INR (₹)" else "$"

# B. Forecast Settings
st.sidebar.subheader("📅 Time Horizon")
days = st.sidebar.slider("Days to Predict", 15, 120, 30)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Simulation Logic:** The simulator assumes a 10% discount leads to a 25% increase in sales volume (Price Elasticity).")

# --- 4. MAIN INTERFACE ---
st.title(" GrowthIQ: Business Intelligence Engine")
st.markdown("### AI-Driven Revenue Predictions & Strategic Insights")

if model is None:
    st.error("⚠️ **System Error:** Model file missing. Please run the training script.")
else:
    if st.button("Generate Strategic Forecast", type="primary"):
        with st.spinner("🤖 Analyzing market trends and running simulations..."):

            # --- BASE CALCULATIONS ---
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            future_forecast = forecast.tail(days).copy()

            # Apply Currency
            future_forecast['yhat'] = future_forecast['yhat'] * exchange_rate
            future_forecast['yhat_lower'] = future_forecast['yhat_lower'] * exchange_rate
            future_forecast['yhat_upper'] = future_forecast['yhat_upper'] * exchange_rate

            # Cumulative
            future_forecast['cumulative_revenue'] = future_forecast['yhat'].cumsum()

            # Metrics
            total_revenue = future_forecast['yhat'].sum()
            avg_daily = future_forecast['yhat'].mean()

            # Growth Analysis
            start_val = future_forecast['yhat'].iloc[0]
            end_val = future_forecast['yhat'].iloc[-1]
            growth_pct = ((end_val - start_val) / start_val) * 100

            # --- UI: THE "BIG NUMBERS" ---
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("💰 Projected Revenue", f"{currency_symbol}{total_revenue:,.0f}")
            with col2: st.metric("📊 Avg. Daily Sales", f"{currency_symbol}{avg_daily:,.0f}")
            with col3: st.metric("📈 Growth Trend", f"{growth_pct:.1f}%", delta_color="normal")
            with col4:
                best_day = future_forecast.loc[future_forecast['yhat'].idxmax()]['ds'].strftime('%b %d')
                st.metric("🏆 Best Sales Day", best_day)

            # --- STRATEGY & SIMULATION ENGINE ---
            st.markdown("---")
            st.subheader("🧠 AI Strategic Advisor & Simulator")

            # DEFINE VARIABLES FOR SIMULATION
            sim_revenue = 0
            sim_profit = 0
            base_profit = total_revenue * 0.30 # Assuming 30% base margin

            # LOGIC FOR DIFFERENT TRENDS
            if growth_pct < -5:
                # SCENARIO: DIPPING TREND (Defensive Retention)
                st.error(f"🛡️ **Strategy: DEFENSIVE RETENTION** (Trend: {growth_pct:.1f}%)")
                st.write("Recommendation: Launch a **10% Discount Campaign** to recover lost traffic.")

                # SIMULATION MATH
                discount_rate = 0.10      # We give 10% off
                volume_uplift = 0.25      # We expect 25% more people to buy
                new_margin = 0.30 - 0.10  # Margin drops from 30% to 20%

                # New Revenue = (Base Revenue * (1 - Discount)) * (1 + Uplift)
                sim_revenue = (total_revenue * (1 - discount_rate)) * (1 + volume_uplift)
                sim_profit = sim_revenue * new_margin

                msg = "✅ **Simulation Result:** By offering a 10% coupon, you sacrifice margin but gain volume. This recovers your revenue trajectory."

            elif growth_pct > 5:
                # SCENARIO: RISING TREND (Aggressive Expansion)
                st.success(f"🚀 **Strategy: AGGRESSIVE EXPANSION** (Trend: +{growth_pct:.1f}%)")
                st.write("Recommendation: Increase **Ad Spend by 15%** to fuel the fire.")

                # SIMULATION MATH
                ad_spend_increase = total_revenue * 0.05 # Spend 5% of rev on ads
                volume_uplift = 0.20 # Ads bring 20% more sales

                sim_revenue = total_revenue * (1 + volume_uplift)
                sim_profit = (sim_revenue * 0.30) - ad_spend_increase

                msg = "✅ **Simulation Result:** Increasing ad spend costs money upfront but maximizes total profit due to high volume."

            else:
                # SCENARIO: STABLE
                st.warning(f"⚖️ **Strategy: OPTIMIZATION** (Trend: Flat)")
                st.write("Recommendation: Create **Product Bundles** to increase Order Value.")

                sim_revenue = total_revenue * 1.10 # 10% increase from bundles
                sim_profit = sim_revenue * 0.30
                msg = "✅ **Simulation Result:** Bundling increases Average Order Value without lowering prices."

            # --- DISPLAY SIMULATION RESULTS ---
            with st.expander("🧪 Open Strategy Simulator (Before vs. After)", expanded=True):
                st.write(msg)

                # Data for Comparison Chart
                sim_data = pd.DataFrame({
                    'Metric': ['Total Revenue', 'Total Profit', 'Total Revenue', 'Total Profit'],
                    'Scenario': ['Current Path', 'Current Path', 'After Strategy', 'After Strategy'],
                    'Value': [total_revenue, base_profit, sim_revenue, sim_profit]
                })

                # Columns for Metrics
                sc1, sc2, sc3 = st.columns(3)

                # Calculate Differences
                rev_diff = sim_revenue - total_revenue
                prof_diff = sim_profit - base_profit

                with sc1:
                    st.metric("Expected Revenue Lift", f"{currency_symbol}{sim_revenue:,.0f}", delta=f"{currency_symbol}{rev_diff:,.0f}")
                with sc2:
                    st.metric("Expected Profit Impact", f"{currency_symbol}{sim_profit:,.0f}", delta=f"{currency_symbol}{prof_diff:,.0f}")
                with sc3:
                    st.metric("Profit Margin Impact", f"{(sim_profit/sim_revenue)*100:.1f}%", delta="-10%" if growth_pct < -5 else "Stable")

                # Comparison Chart
                fig_sim = px.bar(
                    sim_data,
                    x='Metric',
                    y='Value',
                    color='Scenario',
                    barmode='group',
                    text_auto='.2s',
                    color_discrete_map={'Current Path': '#EF553B', 'After Strategy': '#00CC96'}
                )
                fig_sim.update_layout(yaxis_title=f"Amount ({currency_symbol})")
                st.plotly_chart(fig_sim, use_container_width=True)


            # --- VISUALS (TABS) ---
            st.markdown("---")
            tab1, tab2, tab3, tab4 = st.tabs(["📉 Revenue Trend", "💰 Cash Flow", "📅 Best Days", "💾 Data"])

            with tab1:
                st.subheader(f"Daily Revenue Forecast ({currency_symbol})")
                fig = plot_plotly(model, forecast)
                fig.update_layout(yaxis_title=f"Sales ({currency_symbol})", height=500)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Cumulative Cash Accumulation")
                fig_area = px.area(future_forecast, x='ds', y='cumulative_revenue', color_discrete_sequence=['#00CC96'])
                st.plotly_chart(fig_area, use_container_width=True)

            with tab3:
                st.subheader("Day of Week Analysis")
                future_forecast['Day'] = future_forecast['ds'].dt.day_name()
                day_stats = future_forecast.groupby('Day')['yhat'].mean().reindex(
                    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                ).reset_index()
                fig_bar = px.bar(day_stats, x='Day', y='yhat', color='yhat', color_continuous_scale='Blues')
                st.plotly_chart(fig_bar, use_container_width=True)

            with tab4:
                st.subheader("Export Forecast Data")
                display_df = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'cumulative_revenue']].copy()
                display_df.columns = ['Date', f'Predicted ({currency_symbol})', 'Min Estimate', 'Max Estimate', 'Cumulative Total']
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Report (CSV)", csv, "forecast_report.csv", "text/csv")
                st.dataframe(display_df.style.format({f'Predicted ({currency_symbol})': '{:,.2f}', 'Cumulative Total': '{:,.2f}'}))


#=======================================Part 03 =================================================================
import os
import time
import subprocess
from pyngrok import ngrok

# --- 1. SET YOUR TOKEN (CRITICAL STEP) ---
# Replace the text inside the quotes with your actual token from the website
ngrok.set_auth_token("367GNNSjCOPWLZEXNhqWO447hxS_7CmC3cNbQgS5CkhmVcB9d") 
# -----------------------------------------

# 2. Cleanup old processes
os.system("fuser -k 8501/tcp")
ngrok.kill()

# 3. Start Streamlit
process = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "127.0.0.1", "--server.headless", "true"],
    stdout=open('/content/logs.txt', 'w'),
    stderr=subprocess.STDOUT
)

print("⏳ Starting Streamlit... (Waiting 5 seconds)")
time.sleep(5)

# 4. Check & Launch
if process.poll() is not None:
    print("❌ Streamlit crashed! Checking logs...")
    with open('/content/logs.txt', 'r') as f:
        print(f.read())
else:
    print("✅ Streamlit is running successfully!")
    try:
        # Create the tunnel
        public_url = ngrok.connect(8501).public_url
        print(f"🚀 OPEN DASHBOARD: {public_url}")
    except Exception as e:
        print(f"Ngrok Error: {e}")
        print("Double check that you pasted your token correctly inside the quotes!")
