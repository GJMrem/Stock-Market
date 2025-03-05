import torch
from tqdm.auto import tqdm, trange
import mlflow

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
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
    val_frequency: int = None
):
    
    train_losses = []
    val_losses = []
    if val_frequency is None:
        val_frequency = num_epochs//100
        
    out_pbar = trange(num_epochs, unit='epoch')
    train_pbar = tqdm(total=len(train_loader), unit='batch')
    val_pbar = tqdm(total=len(val_loader), unit='batch')
    
    for epoch in out_pbar:
        try:
            out_pbar.set_description(f"Epoch:{epoch+1}/{num_epochs}" + (f" Best Validation Metric: {early_stopper.min_validation_loss:.5f}\r" if early_stopper else ''), refresh=True)
    
            # Training        
            model.train()
            epoch_train_loss = 0.0
            train_pbar.reset()
            for X, y in train_loader:
                train_pbar.update()
                    
                output = model(X)
                
                loss = criterion(output, y)
                loss.backward()
                            
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_train_loss += loss.item()               
                train_pbar.set_description(f"Current training loss: {loss.item():.5f}", refresh=True)
                
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
            train_pbar.set_description(f"Total training loss: {epoch_train_loss:.5f}", refresh=True)
            mlflow.log_metric('train_loss', epoch_train_loss, step=epoch+1)
            if scheduler:
                scheduler.step(loss)
                        
            if (epoch+1) % val_frequency != 0:
                continue
           
            # Validation        
            model.eval()
            val_pbar.reset()
            
            epoch_val_loss = 0.0
            epoch_val_metric = 0.0
            for X, y in val_loader:
                val_pbar.update()
                                        
                output = model(X)
                
                loss = criterion(output, y)
                epoch_val_loss += loss.item()
                
                if metric:
                    val_metric = metric(output, y)
                    epoch_val_metric += val_metric.item()
                
                val_pbar.set_description(f"Current validation loss: {loss.item():.5f}", refresh=True)
                
            epoch_val_loss /= len(val_loader)
            val_losses.append(epoch_val_loss)
            val_pbar.set_description(f"Total validation loss: {epoch_val_loss:.5f}", refresh=True)
            mlflow.log_metric('val_loss', epoch_val_loss, step=epoch+1)
            
            if metric:
                epoch_val_metric /= len(val_loader)
                mlflow.log_metric('val_metric', epoch_val_loss, step=epoch+1)
            
            if not early_stopper:
                continue
            
            if epoch_val_loss < early_stopper.min_validation_loss:                             
                torch.save(model.state_dict(), f'../models/{type(model).__name__}.pt')
                mlflow.log_metric('val_loss_best', early_stopper.min_validation_loss, step=epoch)
                            
            if early_stopper(epoch_val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                mlflow.log_metric('early_stop_epoch', epoch+1)
                break
        
        except(KeyboardInterrupt):
            print(f"Interrupt at epoch {epoch+1}")
            mlflow.log_metric('early_stop_epoch', epoch+1)
            break
                           
    train_pbar.close()
    val_pbar.close()

    return train_losses, val_losses