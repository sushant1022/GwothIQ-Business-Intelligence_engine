
import pandas as pd
from prophet import Prophet
import joblib

try:
    df = pd.read_csv('sales_data.csv')
    if 'Sales' not in df.columns:
        df['Sales'] = df['Price'] * df['Units_Sold']
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    prophet_df = df[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
    
    m = Prophet()
    m.fit(prophet_df)
    joblib.dump(m, 'forecast_model.pkl')
    print("Model Trained!")
except Exception as e:
    print(f"Error: {e}")
