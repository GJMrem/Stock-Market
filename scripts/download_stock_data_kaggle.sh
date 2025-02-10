#!/bin/sh
set -e  # Exit immediately if a command exits with a non-zero status

echo "Enter the absolute dir path to download the dataset (leave empty to use the default: ./data)"
read -r DOWNLOAD_DIR

START_DIR=$PWD

if [ -z "$DOWNLOAD_DIR" ]; then 
  DOWNLOAD_DIR="$START_DIR/data"
fi

mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

if ! curl -L -C - -o ./stock-market-data.zip \
  https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/stock-market-data; then
  echo "Failed to download dataset"
  exit 1
fi

if ! unzip -n stock-market-data.zip; then
  echo "Failed to unzip dataset"
  rm -f stock-market-data.zip
  exit 1
fi

rm stock-market-data.zip

if [ "$DOWNLOAD_DIR" = "$START_DIR/data" ]; then
  mv stock_market_data/* .
  rmdir stock_market_data
else
  ln -sfn "$DOWNLOAD_DIR/stock_market_data" "$START_DIR/data"
fi

echo "Dataset downloaded and extracted successfully to $DOWNLOAD_DIR"