import torch
import torch.nn as nn

__all__ = ['Baseline','LSTMModel','RNNModel','ResNLS']

class Baseline(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(Baseline, self).__init__()
        self.num_factors = input_size
        self.horizon = output_size
        
    def forward(self, x):
        # (N,L,num_factors), assumimg target time series  
        return x[:,-1,0].tile(self.horizon)
    
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, output_size=1,*, hidden_size=1, num_layers=1, dropout=0):
        super(LSTMModel, self).__init__()
        self.num_factors = input_size
        self.horizon = output_size
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
                
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
class RNNModel(nn.Module):
    def __init__(self, input_size=1, output_size=1,*, hidden_size=1, num_layers=1, dropout=0):
        super(RNNModel, self).__init__()
        self.num_factors = input_size
        self.horizon = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out
    
class ResNLS(nn.Module):

    def __init__(self, input_size, output_size, hidden_size = 32, kernel_size=3, stride=1, padding=1, eps=1e-5, dropout = 0.2):
        super(ResNLS, self).__init__()
        self.num_factors = input_size
        self.horizon = output_size
        
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.eps = eps
        self.dropout = dropout
        
        # intialise weights of the attention mechanism
        self.weight = nn.Parameter(torch.zeros(1))

        # intialise cnn structure
        self.cnn = nn.Sequential(           
            nn.Conv1d(input_size, hidden_size, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size, eps),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_size, hidden_size, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size, eps),
        )
        self.linear = nn.Linear(hidden_size, input_size)
        
        # intialise lstm structure
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.final_linear = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # CNN expects input of shape (N,C_in​,L) = (batch_size, num_factors, lookback)
        cnn_output = self.cnn(x.transpose(-1, 1))
        cnn_output = self.linear(cnn_output.transpose(-1,1))

        x = x + self.weight * cnn_output

        # LSTM expects input of shape (N,L,H_in) = (batch_size, lookback, num_factors)
        out, _  = self.lstm(x)
        y_hat = self.final_linear(out[:,-1,:])

        return y_hat