from datetime import datetime, timedelta

import streamlit as st
import altair as alt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
import mlflow

import models
import preprocess

st.set_page_config(page_title="Time Series Prediction Dashboard", layout="wide")
st.title('Stock Market Predictions Dashboard')

date_from_selector, date_to_selector = st.sidebar.columns(2)
date_from = date_from_selector.date_input("Date from", datetime.today().date() - timedelta(days = 30))
date_to = date_to_selector.date_input("Date to", datetime.today().date())
symbol = st.sidebar.text_input("Stock tiker", 'AAPL', placeholder='^GSPC')
model_selector, horizon_selector = st.sidebar.columns(2)
model_name = model_selector.selectbox("Model", ["LSTM", "RNN", "ResNLS", "ARIMA"])
horizon = horizon_selector.selectbox("Horizon", [1,5,7,14,30,60,90], index=1)

# Load original quotes
ticker = yf.Ticker(symbol)
original_quotes = ticker.history(start=date_from, end=date_to)

# Get model predictions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
quote_type = 'Close'

client = mlflow.MlflowClient()
model_versions = client.search_model_versions(f"name='{model_name}' and tag.status='production' tag.horizon='{horizon}'")
latest_version = max(model_versions, key=lambda x: x.version)
model_uri = f"models:/{model_name}/{latest_version.version}"
model = mlflow.pytorch.load_model(model_uri)

scaler = MinMaxScaler(feature_range=(0, 1))
data = torch.FloatTensor(scaler.fit_transform(original_quotes[[quote_type]])).to(device)
X, y = preprocess.create_sequences(data, lookback=5)

#TODO: finetune model for one epoch on the original_quotes 
model.eval()
model_predictions = model(X[-1].unsqueeze(0))
model_predictions = scaler.inverse_transform(model_predictions.detach().cpu())
model_predictions = pd.DataFrame(model_predictions.T, columns=[quote_type],
                                 index=pd.date_range(start=date_to,
                                                     periods=horizon,
                                                     freq='B', name='Date'),
)

model_predictions = pd.concat([model_predictions,original_quotes[[quote_type]].iloc[[-1]]])
model_predictions = model_predictions.assign(source='prediction')

# Merge data on the date column
original_quotes = original_quotes[[quote_type]].assign(source='original')
data = pd.concat([model_predictions,original_quotes])

# Display original quotes and model predictions
st.write(f'{ticker.info['longName']} predictions for {horizon} day(s)')
data = data.reset_index()
# data = data.melt('Date',var_name='source',value_name=quote_type)

interval = alt.selection_interval(encodings=['x'])

base = alt.Chart(data).mark_line().encode(
    x=alt.X('Date:T', axis=alt.Axis(format = '%d %b', title='Time')),
    y=alt.Y(f'{quote_type}:Q', title='Price'),
)

chart = base.mark_line(point=True).encode(
    x=alt.X('Date:T', scale=alt.Scale(domain=interval)),
    y=f'{quote_type}:Q',
    tooltip=[alt.Tooltip('Date:T',format='%Y-%m-%d'), alt.Tooltip(f'{quote_type}:Q')],
    color=alt.Color('source:N',legend=alt.Legend(title=None)),
    # order=alt.Order('source:N', sort='descending')
).properties(
    width=800,
    height=300
)

view = base.add_params(
    interval
).properties(
    width=800,
    height=50
)

st.altair_chart(chart & view, use_container_width=True)
