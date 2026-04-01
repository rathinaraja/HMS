import os
import time
import random
import copy
import argparse
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Yahoo Finance imports
import yfinance as yf

# ==========================================
# 0. REPRODUCIBILITY UTILS
# ==========================================
def set_seed(seed=42):
    """Sets seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. CENTRALIZED CONFIGURATION
# ==========================================
class Config:
    def __init__(self, optimizer_name='All', gpu_id=0, dataset='boston', model_type='mlp', 
                 ticker='AAPL', lookback=10):
        self.OPTIMIZER_NAME = optimizer_name
        self.DATASET = dataset
        self.MODEL_TYPE = model_type  # 'mlp', 'lstm', or 'gru'
        self.TICKER = ticker
        self.LOOKBACK = lookback  # For sequential models
        self.LEARNING_RATE = 0.0001
        self.NUM_EPOCHS = 1000
        self.BATCH_SIZE = 32
        self.NUM_EXECUTIONS = 5
        self.BASE_RESULT_DIR = f'Deep_learning_results/{dataset}_{model_type}/'
        self.HMS_R = 1.0
        self.HMS_T = 1000  # In iterations (not epochs)
        self.HMS_DECAY = 0.9
        self.SEED = 42
        
        # GPU Selection
        if torch.cuda.is_available():
            self.DEVICE = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU: {torch.cuda.get_device_name(self.DEVICE)} (ID: {gpu_id})")
        else:
            self.DEVICE = torch.device('cpu')
            print("Using CPU")

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
def load_and_process_data(dataset='boston', ticker='AAPL', lookback=10):
    """
    Load dataset
    
    For Yahoo Finance (stock data):
        - Input features: Open, High, Low, Volume, Previous Close (5 features)
        - Output: Close price (regression target)
        - Creates sequences for LSTM/GRU with lookback window
    
    For Boston Housing:
        - Standard tabular regression
    """
    if dataset.lower() == 'yahoo':
        print(f"Downloading {ticker} stock data from Yahoo Finance...")
        
        # Download data using yfinance directly
        df = yf.download(ticker, start='2012-01-01', end=datetime.now(), progress=False)
        
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Feature engineering
        df = df.copy()
        df['Prev_Close'] = df['Close'].shift(1)
        df = df.dropna()
        
        # Input features: Open, High, Low, Volume, Previous Close
        feature_cols = ['Open', 'High', 'Low', 'Volume', 'Prev_Close']
        target_col = 'Close'
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        print(f"  Total samples: {len(X)}")
        print(f"  Features: {feature_cols}")
        print(f"  Target: {target_col}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
        # For sequential models, create sequences
        if lookback > 1:
            X_seq, y_seq = create_sequences(X, y, lookback)
            print(f"  Created sequences with lookback={lookback}: {len(X_seq)} samples")
            return X_seq, y_seq, len(feature_cols)
        
        return X, y, len(feature_cols)
    
    elif dataset.lower() == 'boston':
        url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(url, sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        
        cols = [f'feat_{i}' for i in range(13)] + ['target']
        df = pd.DataFrame(np.column_stack([data, target]), columns=cols)
        df_clean = df.dropna()
        
        X = df_clean.iloc[:, :-1].values
        y = df_clean.iloc[:, -1].values
        
        return X, y, 13
    else:
        raise ValueError(f"Dataset {dataset} not supported yet")

def create_sequences(X, y, lookback):
    """
    Create sequences for LSTM/GRU
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        lookback: Number of time steps to look back
    
    Returns:
        X_seq: (n_samples - lookback, lookback, n_features)
        y_seq: (n_samples - lookback,)
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    
    return np.array(X_seq), np.array(y_seq)

def prepare_tensors(X, y, model_type='mlp'):
    """
    Prepare PyTorch tensors
    
    For MLP: X shape (n_samples, n_features)
    For LSTM/GRU: X shape (n_samples, lookback, n_features)
    """
    if model_type in ['lstm', 'gru']:
        # X is already (n_samples, lookback, n_features)
        X_t = torch.tensor(X, dtype=torch.float32)
    else:
        # MLP: flatten if needed
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        X_t = torch.tensor(X, dtype=torch.float32)
    
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return X_t, y_t

# ==========================================
# 3. MODEL ARCHITECTURES
# ==========================================
class MLPModel(nn.Module):
    """Multi-Layer Perceptron"""
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(), 
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class LSTMModel(nn.Module):
    """LSTM for time series"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

class GRUModel(nn.Module):
    """GRU for time series"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        gru_out, _ = self.gru(x)
        # Take last time step
        last_output = gru_out[:, -1, :]
        return self.fc(last_output)

# ==========================================
# 3. MODEL & OPTIMIZATION UTILS
# ==========================================
def get_model_and_optimizer(name, device, input_dim, model_type='mlp', lr=0.0001):
    """Create model and optimizer on specified device"""
    
    # Create model based on type
    if model_type == 'mlp':
        model = MLPModel(input_dim).to(device)
    elif model_type == 'lstm':
        model = LSTMModel(input_dim, hidden_dim=64, num_layers=2).to(device)
    elif model_type == 'gru':
        model = GRUModel(input_dim, hidden_dim=64, num_layers=2).to(device)
    else:
        raise ValueError(f"Model type {model_type} not supported")
    
    # Complete Optimizer List
    if name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif name == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.6)
    elif name == 'NAG':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.6, nesterov=True)
    elif name == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif name == 'AdaDelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif name == 'AdaMax':
        optimizer = optim.Adamax(model.parameters(), lr=lr)
    elif name == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=lr)
    elif name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif name == 'RAdam':
        optimizer = optim.RAdam(model.parameters(), lr=lr)
    elif name == 'ASGD':
        optimizer = optim.ASGD(model.parameters(), lr=lr)
    elif name == 'Rprop':
        optimizer = optim.Rprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Optimizer {name} not found.")
        
    return model, optimizer, nn.MSELoss()

class HMSHandler:
    """HMS Handler - Applied Per Iteration"""
    def __init__(self, model, r=1.0, t=1000, decay_rate=0.9):
        self.model = model
        self.r = r
        self.initial_r = r
        self.t = t
        self.decay_rate = decay_rate
        self.iteration = 0
        self.prev_w = [p.data.clone() for p in model.parameters()]

    def step(self):
        """Apply HMS after each iteration (batch)"""
        with torch.no_grad():
            for i, p in enumerate(self.model.parameters()):
                w_prev, w_curr = self.prev_w[i], p.data
                
                # Skip zero weights
                mask_nonzero = (w_prev != 0) & (w_curr != 0)
                if not mask_nonzero.any():
                    continue
                
                # Extract non-zero weights
                prev_nz = w_prev[mask_nonzero]
                curr_nz = w_curr[mask_nonzero]
                
                # Compute harmonic mean
                abs_prev = torch.abs(prev_nz)
                abs_curr = torch.abs(curr_nz)
                hm = 2 * abs_prev * abs_curr / (abs_prev + abs_curr)
                
                # Compute HMS scalar
                min_abs = torch.minimum(abs_prev, abs_curr)
                hms_scalar = torch.abs(hm - min_abs) * self.r
                
                # Apply HMS based on direction
                result_nz = curr_nz.clone()
                
                # Decreasing: slow down (subtract HMS)
                mask_dec = (prev_nz > curr_nz)
                result_nz[mask_dec] = curr_nz[mask_dec] - hms_scalar[mask_dec]
                
                # Increasing: accelerate (add HMS)
                mask_inc = (prev_nz < curr_nz)
                result_nz[mask_inc] = curr_nz[mask_inc] + hms_scalar[mask_inc]
                
                # Update weights
                p.data[mask_nonzero] = result_nz
                self.prev_w[i] = p.data.clone()
        
        # Decay r every t iterations
        self.iteration += 1
        if self.iteration % self.t == 0:
            self.r = round(self.r * self.decay_rate, 4)

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_engine(model, optimizer, criterion, loader, epochs, device, use_hms=False, r=1.0, t=1000, decay=0.9, verbose=False):
    """Training loop with optional HMS per iteration"""
    hms = HMSHandler(model, r, t, decay) if use_hms else None
    
    epoch_loss_history = []
    iteration_loss_history = []
    
    model.train()
    global_iter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            # Apply HMS after optimizer step (per iteration)
            if use_hms:
                hms.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss * xb.size(0)
            batch_count += 1
            
            # Log iteration loss
            iteration_loss_history.append({
                'epoch': epoch,
                'iteration': batch_idx,
                'global_iteration': global_iter,
                'loss': batch_loss
            })
            global_iter += 1
        
        # Log epoch loss
        epoch_loss_history.append(epoch_loss)
        
        # Print progress every 100 epochs
        if verbose and (epoch + 1) % 100 == 0:
            print(f"      Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}")
    
    return epoch_loss_history, iteration_loss_history

def evaluate(model, X, y, device):
    """Evaluate model on test set"""
    model.eval()
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        mean_loss = nn.MSELoss()(model(X), y)
        total_loss = mean_loss.item() * X.size(0)
        return total_loss

# ==========================================
# 5. EXPERIMENT RUNNER
# ==========================================
def run_experiment(optimizer_name, cfg, raw_X, raw_y, input_dim):
    """Run experiments for a single optimizer"""
    out_dir = os.path.join(cfg.BASE_RESULT_DIR, optimizer_name)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Running: {optimizer_name} on {cfg.DEVICE}")
    print(f"Model: {cfg.MODEL_TYPE.upper()}")
    print(f"Input dim: {input_dim}")
    print(f"{'='*70}")
    
    final_results = {'Generic': [], 'HMS': []}

    for run_id in tqdm(range(cfg.NUM_EXECUTIONS), desc=f"{optimizer_name}"):
        try:
            # Set seed for this run
            set_seed(cfg.SEED + run_id)
            
            # 1. Split & Preprocess
            print(f"\n  Run {run_id+1}: Splitting data...")
            print(f"    Raw X shape: {raw_X.shape}, Raw y shape: {raw_y.shape}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                raw_X, raw_y, test_size=0.2, random_state=cfg.SEED + run_id, shuffle=False
            )
            
            print(f"    Train X: {X_train.shape}, Test X: {X_test.shape}")
            
            # Normalize
            if cfg.MODEL_TYPE in ['lstm', 'gru']:
                # For sequential models: normalize across samples, preserve temporal structure
                n_samples, lookback, n_features = X_train.shape
                print(f"    Reshaping for normalization: {n_samples} x {lookback} x {n_features}")
                
                X_train_2d = X_train.reshape(-1, n_features)
                X_test_2d = X_test.reshape(-1, n_features)
                
                mm_x = MinMaxScaler()
                X_train_n = mm_x.fit_transform(X_train_2d).reshape(n_samples, lookback, n_features)
                X_test_n = mm_x.transform(X_test_2d).reshape(X_test.shape[0], lookback, n_features)
            else:
                print(f"    Normalizing MLP data...")
                mm_x = MinMaxScaler()
                X_train_n = mm_x.fit_transform(X_train)
                X_test_n = mm_x.transform(X_test)
            
            mm_y = MinMaxScaler()
            y_train_n = mm_y.fit_transform(y_train.reshape(-1, 1))
            y_test_n = mm_y.transform(y_test.reshape(-1, 1))
            
            print(f"    Preparing tensors...")
            Xt_train, yt_train = prepare_tensors(X_train_n, y_train_n, cfg.MODEL_TYPE)
            Xt_test, yt_test = prepare_tensors(X_test_n, y_test_n, cfg.MODEL_TYPE)
            
            print(f"    Tensor shapes - Train: {Xt_train.shape}, Test: {Xt_test.shape}")
            
            # For time series, don't shuffle the dataloader
            shuffle_data = False if cfg.MODEL_TYPE in ['lstm', 'gru'] else True
            
            loader = DataLoader(
                TensorDataset(Xt_train, yt_train), 
                batch_size=cfg.BATCH_SIZE, 
                shuffle=shuffle_data
            )
            
            print(f"    DataLoader created with {len(loader)} batches (shuffle={shuffle_data})")
            
            # 2. Initialize Master Weights
            print(f"    Creating models...")
            master, _, _ = get_model_and_optimizer(
                optimizer_name, cfg.DEVICE, input_dim, cfg.MODEL_TYPE, cfg.LEARNING_RATE
            )
            init_state = copy.deepcopy(master.state_dict())
            
            # 3. Train Generic (No HMS)
            print(f"    Training base model...")
            model_gen, opt_gen, crit = get_model_and_optimizer(
                optimizer_name, cfg.DEVICE, input_dim, cfg.MODEL_TYPE, cfg.LEARNING_RATE
            )
            model_gen.load_state_dict(copy.deepcopy(init_state))
            
            hist_gen_epoch, hist_gen_iter = train_engine(
                model_gen, opt_gen, crit, loader, cfg.NUM_EPOCHS, cfg.DEVICE, 
                use_hms=False
            )
            
            print(f"    Evaluating base model...")
            loss_gen = evaluate(model_gen, Xt_test, yt_test, cfg.DEVICE)
            print(f"    Base model test loss: {loss_gen:.4f}")
            
            # 4. Train HMS
            print(f"    Training HMS model...")
            model_hms, opt_hms, crit = get_model_and_optimizer(
                optimizer_name, cfg.DEVICE, input_dim, cfg.MODEL_TYPE, cfg.LEARNING_RATE
            )
            model_hms.load_state_dict(copy.deepcopy(init_state))
            
            hist_hms_epoch, hist_hms_iter = train_engine(
                model_hms, opt_hms, crit, loader, cfg.NUM_EPOCHS, cfg.DEVICE, 
                use_hms=True, r=cfg.HMS_R, t=cfg.HMS_T, decay=cfg.HMS_DECAY
            )
            
            print(f"    Evaluating HMS model...")
            loss_hms = evaluate(model_hms, Xt_test, yt_test, cfg.DEVICE)
            print(f"    HMS model test loss: {loss_hms:.4f}")
            
            # 5. Store results
            final_results['Generic'].append(loss_gen)
            final_results['HMS'].append(loss_hms)
            
            # 6. Save convergence logs
            print(f"    Saving convergence logs...")
            pd.DataFrame({
                'epoch': range(len(hist_gen_epoch)),
                'generic_loss': hist_gen_epoch,
                'hms_loss': hist_hms_epoch
            }).to_csv(f"{out_dir}/run_{run_id+1}_epoch_losses.csv", index=False)
            
            if run_id == 0:
                pd.DataFrame(hist_gen_iter).to_csv(
                    f"{out_dir}/run_{run_id+1}_generic_iteration_losses.csv", index=False
                )
                pd.DataFrame(hist_hms_iter).to_csv(
                    f"{out_dir}/run_{run_id+1}_hms_iteration_losses.csv", index=False
                )
            
            print(f"    ✓ Run {run_id+1} completed successfully")
            
        except Exception as e:
            print(f"\n    ✗ Error in run {run_id+1}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    # 7. Statistical Analysis
    print(f"\n  Performing statistical analysis...")
    summary_df = pd.DataFrame(final_results)
    summary_df.to_csv(f"{out_dir}/{optimizer_name}_summary.csv", index=False)

    mean_gen = np.mean(final_results['Generic'])
    std_gen = np.std(final_results['Generic'])
    mean_hms = np.mean(final_results['HMS'])
    std_hms = np.std(final_results['HMS'])
    
    t_stat, p_val = stats.ttest_rel(final_results['Generic'], final_results['HMS'])
    improvement_pct = ((mean_gen - mean_hms) / mean_gen * 100)

    # 8. Generate Report
    report = f"""
{'='*70}
EXPERIMENT SUMMARY: {optimizer_name}
{'='*70}
Configuration:
  Dataset: {cfg.DATASET.upper()} ({cfg.TICKER if cfg.DATASET == 'yahoo' else 'N/A'})
  Model: {cfg.MODEL_TYPE.upper()}
  Input dim: {input_dim}
  Epochs: {cfg.NUM_EPOCHS}
  Batch Size: {cfg.BATCH_SIZE}
  Learning Rate: {cfg.LEARNING_RATE}
  Runs: {cfg.NUM_EXECUTIONS}
  Device: {cfg.DEVICE}

HMS Parameters:
  r (initial): {cfg.HMS_R}
  t (decay interval): {cfg.HMS_T} iterations
  decay rate: {cfg.HMS_DECAY}

Results (Test MSE over {cfg.NUM_EXECUTIONS} runs):
  {optimizer_name} (Base):     {mean_gen:.5f} ± {std_gen:.5f}
  {optimizer_name} + HMS:       {mean_hms:.5f} ± {std_hms:.5f}

Statistical Analysis:
  Improvement: {(mean_gen - mean_hms):.5f} MSE ({improvement_pct:+.2f}%)
  t-statistic: {t_stat:.4f}
  p-value: {p_val:.6f}
  Significant: {'YES (p < 0.05)' if p_val < 0.05 else 'NO (p >= 0.05)'}

Individual Results:
  Generic: {final_results['Generic']}
  HMS:     {final_results['HMS']}
{'='*70}
"""
    
    with open(f"{out_dir}/{optimizer_name}_report.txt", "w") as f:
        f.write(report)
    print(report)
    
    return {
        'optimizer': optimizer_name,
        'mean_generic': mean_gen,
        'std_generic': std_gen,
        'mean_hms': mean_hms,
        'std_hms': std_hms,
        'improvement_pct': improvement_pct,
        'p_value': p_val,
        'significant': p_val < 0.05
    }

# ==========================================
# 6. MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HMS Optimizer Evaluation')
    parser.add_argument('--optimizer', type=str, default='All', 
                       help='Optimizer name or "All"')
    parser.add_argument('--gpu', type=int, default=0, 
                       help='GPU ID')
    parser.add_argument('--dataset', type=str, default='boston',
                       choices=['boston', 'yahoo'],
                       help='Dataset name')
    parser.add_argument('--model', type=str, default='mlp',
                       choices=['mlp', 'lstm', 'gru'],
                       help='Model architecture')
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Stock ticker for Yahoo Finance (if dataset=yahoo)')
    parser.add_argument('--lookback', type=int, default=10,
                       help='Lookback window for LSTM/GRU')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of epochs')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of runs')
    
    args = parser.parse_args()
    
    # Initialize Config
    cfg = Config(
        optimizer_name=args.optimizer, 
        gpu_id=args.gpu, 
        dataset=args.dataset,
        model_type=args.model,
        ticker=args.ticker,
        lookback=args.lookback
    )
    cfg.NUM_EPOCHS = args.epochs
    cfg.NUM_EXECUTIONS = args.runs
    
    set_seed(cfg.SEED)
    
    # Load Data
    print(f"Loading {cfg.DATASET} dataset...")
    if cfg.DATASET == 'yahoo':
        raw_X, raw_y, input_dim = load_and_process_data(
            cfg.DATASET, 
            ticker=cfg.TICKER,
            lookback=cfg.LOOKBACK if cfg.MODEL_TYPE in ['lstm', 'gru'] else 1
        )
    else:
        raw_X, raw_y, input_dim = load_and_process_data(cfg.DATASET)
    
    print(f"  Samples: {len(raw_X)}, Input dim: {input_dim}")
    
    # Define Optimizers
    ALL_OPTIMIZERS = [
        'SGD', 'Momentum', 'NAG', 'AdaGrad', 'RMSprop', 'AdaDelta', 
        'Adam', 'AdaMax', 'NAdam', 'AdamW', 'RAdam', 'ASGD', 'Rprop'
    ]
    
    if cfg.OPTIMIZER_NAME == 'All':
        run_list = ALL_OPTIMIZERS
    else:
        if cfg.OPTIMIZER_NAME not in ALL_OPTIMIZERS:
            print(f"Error: {cfg.OPTIMIZER_NAME} not in supported list: {ALL_OPTIMIZERS}")
            exit(1)
        run_list = [cfg.OPTIMIZER_NAME]
    
    # Execute Experiments
    all_results = []
    for opt_name in run_list:
        result = run_experiment(opt_name, cfg, raw_X, raw_y, input_dim)
        all_results.append(result)
    
    # Save Global Summary
    if len(all_results) > 1:
        summary_df = pd.DataFrame(all_results)
        summary_file = os.path.join(cfg.BASE_RESULT_DIR, 'global_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ Global summary saved: {summary_file}")
        print(f"\n{summary_df.to_string(index=False)}")