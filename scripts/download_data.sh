#!/bin/sh
echo "Enter the absolute dir path to download the dataset (leave empty to use the default: ./data)\n"
read DOWNLOAD_DIR

START_DIR=$PWD

if [ ! -z "$DOWNLOAD_DIR" ]; then 
  mkdir -p $DOWNLOAD_DIR
else 
  DOWNLOAD_DIR="."
fi

cd $DOWNLOAD_DIR
curl -L -C - -o ./stock-market-data.zip\
  https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/stock-market-data

unzip -n stock-market-data.zip
rm stock-market-data.zip
echo $PWD

if [ "$DOWNLOAD_DIR" = "." ]; then 
  mv ./stock_market_data ./data_test
else
  ln -s "$DOWNLOAD_DIR/stock_market_data" "$START_DIR/data_test"
fi

echo "Dataset downloaded and extracted successfully."