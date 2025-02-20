import baostock as bs
import pandas as pd

tickers = {
    "SSEC": "sh.000001",  # Shanghai Composite Index
    "CSI300": "sh.000300",  # Alternative for CSI 300
    "CSI500": "sh.000905"   # Alternative for CSI 500  
} 
    
# download CSI dataset
lg = bs.login()

fields= "Date,Open,High,Low,Close,Volume"
for index_name, ticker in tickers.items():
    print(f"Fetching data for {index_name} ({ticker})...")
    rs = bs.query_history_k_data_plus(ticker, fields, start_date="2009-01-01", end_date="2022-12-31", frequency="d", adjustflag="2")
    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())
    data = pd.DataFrame(data_list, columns=rs.fields)
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)
    
    filename = f"data/{index_name}.csv"
    data.to_csv(filename)
    
    print(f"Data for {index_name} saved to {filename}\n")
    print("=" * 50 + "\n")

bs.logout()