import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

class S4DKernel(nn.Module):
    """Simplified S4D Kernel."""
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        log_dt = torch.rand(d_model) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        self.dt = nn.Parameter(torch.exp(log_dt))
        
        # Initialize A to have negative real part (stable)
        A_real = 0.5 * torch.ones(d_model, N)
        A_imag = math.pi * torch.arange(N).unsqueeze(0).repeat(d_model, 1)
        self.A_real = nn.Parameter(A_real)
        self.A_imag = nn.Parameter(A_imag)
        
        self.C = nn.Parameter(torch.randn(d_model, N, dtype=torch.cfloat))

    def forward(self, L):
        dt = self.dt.unsqueeze(-1)
        A = -torch.exp(self.A_real) + 1j * self.A_imag
        A_dt = A * dt
        
        roots = torch.exp(A_dt) # (d_model, N)
        powers = torch.arange(L, device=roots.device).unsqueeze(-1).unsqueeze(-1) # (L, 1, 1)
        vander = torch.pow(roots.unsqueeze(0), powers) # (L, d_model, N)
        
        K = torch.einsum('ldn,dn->ld', vander, self.C).real
        return K

class S4DLayer(nn.Module):
    def __init__(self, d_model, N=64):
        super().__init__()
        self.kernel = S4DKernel(d_model, N)
        self.D = nn.Parameter(torch.randn(d_model))
        # Gating and mixing
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, u):
        B, L, D = u.shape
        K = self.kernel(L)
        
        K_f = torch.fft.rfft(K, n=2*L, dim=0)
        u_f = torch.fft.rfft(u, n=2*L, dim=1)
        
        y_f = K_f.unsqueeze(0) * u_f
        y = torch.fft.irfft(y_f, n=2*L, dim=1)[:, :L, :]
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)
        
        y_proj = self.linear1(y)
        y1, y2 = y_proj.chunk(2, dim=-1)
        y = y1 * F.silu(y2)
        return self.linear2(y)

class ConvS4DModel(nn.Module):
    def __init__(self, in_channels, n_classes, d_model=32, N=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=15, padding=7, groups=d_model)
        self.pool = nn.AvgPool1d(4)
        
        self.s4d = S4DLayer(d_model, N)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.pool(x)
        
        # S4D expects (Batch, Time, Channels)
        x = rearrange(x, 'b c l -> b l c')
        
        x = self.s4d(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        return self.classifier(x)

class ConvS4DClassifier(ClassifierMixin, BaseEstimator):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.three_d_array = True
        return tags

    def __init__(self, in_channels=8, d_model=32, N=32, lr=0.005, epochs=30, batch_size=32, ewc_lambda=0.4):
        self.in_channels = in_channels
        self.d_model = d_model
        self.N = N
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.ewc_lambda = ewc_lambda
        
        self.model = None
        self.classes_ = None
        self.label_encoder = None
        self.fisher = {}
        self.optpar = {}

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        n_classes = len(self.classes_)
        
        if len(X.shape) == 2:
            raise ValueError("Input X must be 3-dimensional (N, Channels, Time)")
            
        self.in_channels = X.shape[1]
        self.model = ConvS4DModel(self.in_channels, n_classes, self.d_model, self.N)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y_encoded, dtype=torch.long))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        self._compute_fisher(loader)
        return self

    def partial_fit(self, X, y, classes=None):
        if self.model is None:
            if classes is not None:
                self.classes_ = classes
            return self.fit(X, y)
            
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        if self.label_encoder is not None:
            y_encoded = self.label_encoder.transform(y)
        else:
            y_encoded = y
            
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_encoded, dtype=torch.long)
        
        for _ in range(3): # A few steps for online adaptation
            optimizer.zero_grad()
            outputs = self.model(X_t)
            loss = criterion(outputs, y_t)
            
            ewc_loss = 0.0
            for name, param in self.model.named_parameters():
                if name in self.fisher:
                    fisher = self.fisher[name].to(param.device)
                    optpar = self.optpar[name].to(param.device)
                    ewc_loss += (fisher * (param - optpar).abs().pow(2)).sum()
            
            loss = loss + (self.ewc_lambda / 2) * ewc_loss
            loss.backward()
            optimizer.step()
            
        self._update_fisher(X_t, y_t)
        return self

    def _compute_fisher(self, loader):
        self.model.eval()
        fisher = {}
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param.data, dtype=torch.float32)
            
        criterion = nn.CrossEntropyLoss()
        for batch_X, batch_y in loader:
            self.model.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.abs().pow(2) / len(loader)
                    
        self.fisher = fisher
        self.optpar = {n: p.data.clone() for n, p in self.model.named_parameters()}
        
    def _update_fisher(self, X, y, alpha=0.9):
        self.model.eval()
        self.model.zero_grad()
        outputs = self.model(X)
        loss = nn.CrossEntropyLoss()(outputs, y)
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.fisher:
                self.fisher[name] = alpha * self.fisher[name] + (1 - alpha) * param.grad.data.abs().pow(2)
        self.optpar = {n: p.data.clone() for n, p in self.model.named_parameters()}

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.tensor(X, dtype=torch.float32))
            _, predicted = torch.max(outputs.data, 1)
            
        preds = predicted.numpy()
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(preds)
        return preds

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.tensor(X, dtype=torch.float32))
            probs = torch.softmax(outputs, dim=1)
        return probs.numpy()

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model is not None:
            state['model_state'] = self.model.state_dict()
            del state['model']
        return state

    def __setstate__(self, state):
        model_state = state.pop('model_state', None)
        self.__dict__.update(state)
        if model_state is not None:
            n_classes = len(self.classes_) if self.classes_ is not None else 2
            self.model = ConvS4DModel(self.in_channels, n_classes, self.d_model, self.N)
            self.model.load_state_dict(model_state)
