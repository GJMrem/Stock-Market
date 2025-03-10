import torch
from tqdm.std import tqdm as tqdm_std
from tqdm.auto import tqdm, trange
import mlflow

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
    pbar: tqdm_std = None
):
    model.train()
    epoch_loss = 0.0  
    
    if pbar: pbar.reset() 
    for X, y in dataloader:
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
    metric: torch.nn.Module = None,  
    pbar: tqdm_std = None
):

    model.eval()
    epoch_loss = 0.0
    epoch_metric = 0.0
    
    if pbar: pbar.reset()
    for X, y in dataloader:       
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
    num_epochs: int,
    metric: torch.nn.Module = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    early_stopper: EarlyStopper = None,
    val_interval: int = None
) -> tuple[list, list]:
    if val_interval is None:
        val_interval = max(num_epochs // 100, 1)
    
    train_losses = []
    val_losses = []
    
    out_pbar = trange(num_epochs, unit='epoch')
    train_pbar = tqdm(total=len(train_loader), unit='batch')
    val_pbar = tqdm(total=len(val_loader), unit='batch')
    
    for epoch in out_pbar:
        try:
            out_pbar.set_description(f"Epoch:{epoch+1}/{num_epochs}" + (f" Best Validation Metric: {early_stopper.min_validation_loss:.5f}\r" if early_stopper else ''), refresh=True)
            
            # Training
            epoch_train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch=epoch, pbar=train_pbar)
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
                torch.save(model.state_dict(), f'../models/{type(model).__name__}.pt')
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

def train_epoch_with_val(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    metric: torch.nn.Module = None,
    epoch: int = 0,
    val_interval: int = 1,
    train_pbar: tqdm_std = None,
    val_pbar: tqdm_std = None,
):
    model.train()
    epoch_train_loss = 0.0
    
    if train_pbar: train_pbar.reset()    
    for batch, (X, y) in enumerate(train_loader):
        output = model(X)
        
        loss = criterion(output, y)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_train_loss += loss.item()
                  
        if train_pbar:
            train_pbar.update()
            train_pbar.set_description(f"Current training loss: {loss.item():.5f}", refresh=True)
        
        if batch % val_interval != 0:
            continue
        
        epoch_val_loss, epoch_val_metric = validate(model, val_loader, criterion, epoch=epoch, metric=metric, pbar=val_pbar)
    epoch_train_loss /= len(train_loader)
    if train_pbar: train_pbar.set_description(f"Total training loss: {epoch_train_loss:.5f}", refresh=True)
    mlflow.log_metric('train_loss', epoch_train_loss, step=epoch+1)

    return epoch_train_loss, epoch_val_loss