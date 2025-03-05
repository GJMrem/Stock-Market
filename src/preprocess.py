import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator

def prepare_time_series_data(
    data: np.ndarray, 
    train_size: int, 
    val_size: int, 
    test_size: int = 1, 
    lookback: int = 5, 
    horizon: int = 1, 
    num_features: int = 1, 
    scaler: BaseEstimator = MinMaxScaler((0, 1)),  
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
): 
    # Split data
    data_train, data_val, data_test = data[:train_size], data[train_size:train_size+val_size], data[train_size+val_size:train_size+val_size+test_size]

    # Normalize data  
    data_train = torch.FloatTensor(scaler.fit_transform(data_train.reshape(-1, num_features))).to(device)
    data_val = torch.FloatTensor(scaler.transform(data_val.reshape(-1, num_features))).to(device)
    data_test = torch.FloatTensor(scaler.transform(data_test.reshape(-1, num_features))).to(device)
    
    # Create sequences
    X_train, y_train = create_sequences(data_train, lookback, horizon)
    X_val, y_val = create_sequences(data_val, lookback, horizon)  
    X_test, y_test = create_sequences(data_test, lookback, horizon)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_sequences(data:torch.Tensor, lookback:int, horizon:int = 1):
    # Create the sequences, assuming first column is the target
    X = data.unfold(dimension=0, size=lookback, step=1)[:-horizon].transpose(-1,1)
    y = data[:,0].unfold(dimension=0, size=horizon, step=1)[lookback:].transpose(-1,1)
    return X, y