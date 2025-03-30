import torch
from numpy.typing import ArrayLike
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.base import TransformerMixin

def split_data(data: ArrayLike,
               train_size: int | float = 0.7,
               val_size: int | float = 0.15,
               test_size: int | float = 0.15
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    n_samples = len(data)
    
    # Convert ratios to absolute sizes
    if isinstance(train_size, float): train_size = int(train_size * n_samples)
    if isinstance(val_size, float): val_size = int(val_size * n_samples)
    if isinstance(test_size, float): test_size = int(test_size * n_samples)
    
    # Ensure the sizes don't exceed the dataset size
    if train_size + val_size + test_size > n_samples:
        raise ValueError("The sum of train_size, val_size, and test_size exceeds the dataset size.")
    
    data_train = data[:train_size]
    data_val = data[train_size:train_size+val_size]
    data_test = data[train_size+val_size:train_size+val_size+test_size]
     
    return data_train, data_val, data_test

def scale_data(
    *data: ArrayLike,
    scaler: TransformerMixin = FunctionTransformer(lambda x: x)
):
    # Scale data assuming first argument is the training data and the rest are validation/test data
    scaled_data = []
    scaled_data.append(torch.FloatTensor(scaler.fit_transform(data[0])))
    for i in range(1, len(data)):
        scaled_data.append(torch.FloatTensor(scaler.transform(data[i])))
    return scaled_data

def create_sequences(data:torch.Tensor, lookback:int, horizon:int = 1):
    # Create the sequences, assuming the first column is the target
    X = data.unfold(dimension=0, size=lookback, step=1)[:-horizon].transpose(-1,1) #(N,lookback, num_factors)
    y = data[:,0].unfold(dimension=0, size=horizon, step=1)[lookback:].transpose(-1,1) #(N, horizon)
    
    # Find sequences containing NaN values
    X_has_nan = torch.isnan(X).any(dim=(1,2))
    y_has_nan = torch.isnan(y).any(dim=1)
    valid_mask = ~(X_has_nan | y_has_nan)
    
    # Filter out sequences with NaN values
    X = X[valid_mask]
    y = y[valid_mask]
    
    return X, y

def prepare_time_series_data(
    data: ArrayLike,
    train_size: int | float = 0.7,
    val_size: int | float = 0.15,
    test_size: int | float = 0.15,
    lookback: int = 5,
    *,
    horizon: int = 1,
    scaler: TransformerMixin = FunctionTransformer(lambda x: x)
): 
    data_train, data_val, data_test = split_data(data,train_size, val_size, test_size)
    
    data_train, data_val, data_test = scale_data(data_train, data_val, data_test, scaler=scaler)
    
    X_train, y_train = create_sequences(data_train, lookback, horizon)
    X_val, y_val = create_sequences(data_val, lookback, horizon)  
    X_test, y_test = create_sequences(data_test, lookback, horizon)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def prepare_time_series_data_presplit(
    data_train: ArrayLike,
    data_val: ArrayLike,
    data_test: ArrayLike,
    lookback: int = 5,
    *,
    horizon: int = 1,
    scaler: TransformerMixin = FunctionTransformer(lambda x: x)
): 
    # Normalize data  
    data_train, data_val, data_test = scale_data(data_train, data_val, data_test, scaler=scaler)
    
    # Create sequences
    X_train, y_train = create_sequences(data_train, lookback, horizon)
    X_val, y_val = create_sequences(data_val, lookback, horizon)  
    X_test, y_test = create_sequences(data_test, lookback, horizon)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

