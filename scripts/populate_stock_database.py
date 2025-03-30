import os
import argparse
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

PROJECT_DIR = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Train a model with configurable parameters.")
parser.add_argument('--dataset', type=str, default=PROJECT_DIR/'data/sp500', metavar='DATASET_PATH',
                    help=("Path to dataset folder. "
                          "All CSV files inside the folder and subfolders will be loaded into the database, "
                          "using the folder name as the table name and file names as stocks ids/tickers. "
                          "Default: %(default)s"
                          )
                    )
args = parser.parse_args()

# Database connection
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgers')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'stock_market')
engine = create_engine(f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432/{POSTGRES_DB}')

# Load all CSV files
csv_dir = args.dataset
for file in csv_dir.rglob('*.csv'):
    print(f"Loading {file.name:9s} to the database...", end='\r', flush=True)
    
    df = pd.read_csv(file)
    df = df.assign(ticker=file.stem)
    df.columns = df.columns.str.lower()
    
    df.to_sql(csv_dir.stem, engine, if_exists='append', index=False)
