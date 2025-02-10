import yfinance as yf
import pandas as pd

# Define the tickers for the indices
tickers = {
    "SP500": "^GSPC",
    "EWZ": "EWZ"
}

# Function to get historical data for a given ticker
def get_index_data(ticker, period="max"):
    index = yf.Ticker(ticker)
    hist = index.history(period=period)
    return hist

# Fetch and save data for each index to a CSV file
for index_name, ticker in tickers.items():
    print(f"Fetching data for {index_name} ({ticker})...")
    data = get_index_data(ticker)
    
    filename = f"data/{index_name}.csv"
    data.to_csv(filename)
    
    print(f"Data for {index_name} saved to {filename}\n")
    print("=" * 50 + "\n")