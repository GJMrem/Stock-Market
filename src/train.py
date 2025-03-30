from pathlib import Path
import torch

import mlflow
from tqdm.std import tqdm as tqdm_std
from tqdm.auto import tqdm, trange

PROJECT_DIR = Path(__file__).resolve().parents[1]
class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    epoch: int = 0,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    pbar: tqdm_std = None
)-> float:
    model.train()
    epoch_loss = 0.0  
    
    if pbar: pbar.reset() 
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        
        loss = criterion(output, y)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()               
        if pbar:
            pbar.update()
            pbar.set_description(f"Current training loss: {loss.item():.5f}", refresh=True)
 
    epoch_loss /= len(dataloader)
    if pbar: pbar.set_description(f"Total training loss: {epoch_loss:.5f}", refresh=True)
    mlflow.log_metric('train_loss', epoch_loss, step=epoch+1)

    return epoch_loss

def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    *,
    epoch: int = 0,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    metric: torch.nn.Module = None,  
    pbar: tqdm_std = None
)-> tuple[float, float]:

    model.eval()
    epoch_loss = 0.0
    epoch_metric = 0.0
    
    if pbar: pbar.reset()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        
        loss = criterion(output, y)
        epoch_loss += loss.item()
        
        if metric:
            val_metric = metric(output, y)
            epoch_metric += val_metric.item()
            
        if pbar:
            pbar.update()
            pbar.set_description(f"Current validation loss: {loss.item():.5f}", refresh=True)
        
    epoch_loss /= len(dataloader)
    if pbar: pbar.set_description(f"Total validation loss: {epoch_loss:.5f}", refresh=True)
    mlflow.log_metric('val_loss', epoch_loss, step=epoch+1)
    
    if metric:
        epoch_metric /= len(dataloader)
        mlflow.log_metric('val_metric', epoch_metric, step=epoch+1)
        
    return epoch_loss, epoch_metric

def run_training(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    *,
    num_epochs: int,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    metric: torch.nn.Module = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    early_stopper: EarlyStopper = None,
    val_interval: int = None
) -> tuple[list, list]:
    if val_interval is None: val_interval = max(num_epochs // 100, 1)
    
    train_losses = []
    val_losses = []
    
    out_pbar = trange(num_epochs, unit='epoch')
    train_pbar = tqdm(total=len(train_loader), unit='batch')
    val_pbar = tqdm(total=len(val_loader), unit='batch')
    
    for epoch in out_pbar:
        try:
            out_pbar.set_description(f"Epoch:{epoch+1}/{num_epochs}" + (f" Best Validation Metric: {early_stopper.min_validation_loss:.5f}\r" if early_stopper else ''), refresh=True)
            
            # Training
            epoch_train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch=epoch, device=device, pbar=train_pbar)
            train_losses.append(epoch_train_loss)
            
            if scheduler:
                scheduler.step(epoch_train_loss)
                
            if (epoch+1) % val_interval != 0:
                continue
            
            # Validation
            epoch_val_loss, epoch_val_metric = validate(model, val_loader, criterion, epoch=epoch, metric=metric, pbar=val_pbar)
            val_losses.append(epoch_val_loss)
            
            if not early_stopper:
                continue
            
            if epoch_val_loss < early_stopper.min_validation_loss:                             
                torch.save(model.state_dict(), PROJECT_DIR/'models'/f'{type(model).__name__}.pt')
                mlflow.log_metric('val_loss_best', early_stopper.min_validation_loss, step=epoch)
                
            if early_stopper(epoch_val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                mlflow.log_metric('early_stop_epoch', epoch+1)
                break
            
        except KeyboardInterrupt:
            print(f"Interrupt at epoch {epoch+1}")
            mlflow.log_metric('early_stop_epoch', epoch+1)
            break
        
    train_pbar.close()
    val_pbar.close()

    return train_losses, val_losses


if __name__ == '__main__':
    import os
    import argparse  
    import json
    
    from sqlalchemy import create_engine
    import pandas as pd
    
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn import metrics
    
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from mlflow.models import infer_signature
    
    import models
    from preprocess import split_data, prepare_time_series_data
    
    torch.manual_seed(42)  
    parser = argparse.ArgumentParser(description="Train a model with configurable parameters.")
    parser.add_argument('--dataset', type=str, default='sp500', help="")
    parser.add_argument('--config', type=str, default=PROJECT_DIR/'configs/config.json', help="Path to config file")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to train the model on")
    args = parser.parse_args()
    
    with open(Path(args.config)) as json_file:
        config = json.load(json_file)  
        model_name = config['model']
        training_params = config['training_params']
        model_params = config['model_params']
        data_params = config['data_params']
    
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'stock_market')
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
    
    mlflow.set_experiment("Stock Market Predictions")
    mlflow.start_run(run_name=model_name)
    mlflow.log_dict(config, "config.json")
    mlflow.log_params({**training_params, **data_params, **model_params})
    
    device = torch.device(args.device)
    engine = create_engine(f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}')
    quote_type = 'close'
    df = pd.read_sql(f'SELECT date, ticker, {quote_type} FROM {args.dataset} ORDER BY date', engine, index_col='date') 
    mlflow.log_input(mlflow.data.from_pandas(df, targets = quote_type, name=args.dataset))
        
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    X_trains, y_trains = [], []
    X_vals, y_vals = [], []
    X_tests, y_tests = [], []

    # Group and split data by ticker
    for ticker, group in df.groupby('ticker'):
        data = group[[quote_type]].values
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_time_series_data(data, scaler=scaler, **data_params)
    
        X_trains.append(X_train)
        y_trains.append(y_train)
    
        X_vals.append(X_val)
        y_vals.append(y_val) 
    
        X_tests.append(X_test)
        y_tests.append(y_test)

    # Concatenate all data
    X_train = torch.cat(X_trains, dim=0)
    y_train = torch.cat(y_trains, dim=0)
    
    X_val = torch.cat(X_vals, dim=0)
    y_val = torch.cat(y_vals, dim=0)
    
    X_test = torch.cat(X_tests, dim=0)
    y_test = torch.cat(y_tests, dim=0)  
    
    Model = getattr(models, model_name)
    model = Model(input_size=1, output_size=data_params['horizon'], **model_params).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(),lr=training_params['learning_rate'])
    metric = nn.L1Loss().to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=training_params['lr_decay'], patience=training_params['lr_patience'])
    early_stopper = EarlyStopper(patience=training_params['patience'], min_delta=training_params['min_delta'])

    
    mlflow.log_params({
        'criterion': type(criterion).__name__,
        'optimizer': type(optimizer).__name__,
        'metric': type(metric).__name__,
        'scheduler': type(scheduler).__name__,
    })

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=training_params['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=training_params['batch_size'], shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=training_params['batch_size'], shuffle=False)
    
    train_losses, val_losses = run_training(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=training_params['num_epochs'],
        device=device,
        metric=metric,
        scheduler=scheduler,
        early_stopper=early_stopper,
        val_interval=training_params['val_interval']        
    )
  
    try:   
        model.load_state_dict(torch.load(PROJECT_DIR/'models'/f'{type(model).__name__}.pt'))
    except (FileNotFoundError, RuntimeError) as e:
        print(e)
    
    mae = 0.0
    mse = 0.0
    rmse = 0.0
    
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(test_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            X = X.cpu().detach().numpy()           
            y = y.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
            
            mae += metrics.mean_absolute_error(y_pred, y)
            mse += metrics.mean_squared_error(y_pred, y)
            rmse += np.sqrt(metrics.mean_squared_error(y_pred, y))

    mae /= len(test_loader)
    mse /= len(test_loader)
    rmse /= len(test_loader)
    
    mlflow.pytorch.log_model(
        registered_model_name=model_name,
        artifact_path=model_name,
        pytorch_model=model,
        input_example=X,
        signature=infer_signature(X, y),
    )

    mlflow.log_metrics({
        'test_mae': mae,
        'test_mse': mse,
        'test_rmse': rmse
    })
    
    client = mlflow.tracking.MlflowClient()
    all_versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = max(int(v.version) for v in all_versions)
    
    client.set_model_version_tag(model_name, latest_version, 'status', 'production')
    client.set_model_version_tag(model_name, latest_version, 'horizon', data_params['horizon'])
    mlflow.end_run()
