import os
import sys
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

# Import modular data loaders and configs
from data_loaders import DatasetLoader
from dataset_configs import DatasetConfig 

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
    def __init__(self, output_dir, optimizer_name='All', gpu_id=0, dataset='boston', model_type='mlp'):
        self.OPTIMIZER_NAME = optimizer_name
        self.DATASET = dataset
        self.MODEL_TYPE = model_type  # 'linear', 'mlp', 'lstm', or 'gru'
        self.LEARNING_RATE = 0.0001
        self.NUM_EPOCHS = 1000
        self.BATCH_SIZE = 32
        self.NUM_EXECUTIONS = 5
        self.BASE_RESULT_DIR = f'{output_dir}/{dataset}/{model_type}/'
        
        # Lookahead Parameters
        self.LOOKAHEAD_K = 5             # Synchronization period (3-10 iterations)
        self.LOOKAHEAD_ALPHA = 0.5       # Slow weights step size (0.3-0.8)
        
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
def load_and_process_data(dataset='boston'):
    """
    Load dataset using DatasetLoader
    
    Returns:
        X: Features (numpy array)
        y: Target (numpy array)
        input_dim: Number of input features (ACTUAL dimension from data)
    """
    # Validate dataset
    available_datasets = DatasetConfig.get_available_datasets()
    if dataset not in available_datasets:
        raise ValueError(f"Dataset '{dataset}' not supported. Available: {available_datasets}")
    
    # Get dataset configuration
    config = DatasetConfig.get_config(dataset)
    
    print(f"Loading dataset: {config['name']}")
    
    # Load data using the centralized loader
    X, y = DatasetLoader.LOADERS[dataset]()
    
    # Get ACTUAL input dimension from the loaded data
    actual_input_dim = X.shape[1]
    
    print(f"  Samples: {len(X)}")
    print(f"  Features (actual): {actual_input_dim}")
    print(f"  Features (config): {config['input_dim']}")
    if actual_input_dim != config['input_dim']:
        print(f"  ⚠️  WARNING: Config mismatch! Using actual dimension: {actual_input_dim}")
    print(f"  Output dim: {config['output_dim']}")
    print(f"  Task: {config['task']}")
    
    # Handle multi-output datasets (e.g., energy efficiency)
    if len(y.shape) > 1 and y.shape[1] > 1:
        print(f"  Note: Multi-output dataset, using first output column")
        y = y[:, 0]
    
    # Return ACTUAL input dimension from the data, not from config
    return X, y, actual_input_dim

def prepare_tensors(X, y):
    """Prepare PyTorch tensors"""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return X_t, y_t

# ==========================================
# 3. MODEL ARCHITECTURES
# ==========================================
class LinearModel(nn.Module):
    """Simple Linear Regression"""
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

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
    """LSTM for sequential data (not typically used for tabular regression)"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        # For tabular data, we treat each sample as a sequence of length 1
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # Reshape for LSTM: (batch, seq_len=1, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

class GRUModel(nn.Module):
    """GRU for sequential data (not typically used for tabular regression)"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # Reshape for GRU: (batch, seq_len=1, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        return self.fc(last_output)

# ==========================================
# 4. MODEL & OPTIMIZATION UTILS
# ==========================================
def get_model_and_optimizer(name, device, input_dim, model_type='mlp', lr=0.0001):
    """Create model and optimizer on specified device"""
    
    # Create model based on type
    if model_type == 'linear':
        model = LinearModel(input_dim).to(device)
    elif model_type == 'mlp':
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

# ==========================================
# 5. LOOKAHEAD HANDLER
# ==========================================
class LookaheadHandler:
    """
    Lookahead Optimizer Handler - Applied Periodically
    
    Based on: "Lookahead Optimizer: k steps forward, 1 step back" (2019)
    https://arxiv.org/pdf/1907.08610
    
    Core Principle: Dual-loop system with "fast" and "slow" weights
    
    Algorithm:
        1. Fast weights θ_fast updated by inner optimizer (SGD, Adam, etc.) for k steps
        2. After k steps, slow weights are updated:
           θ_slow = θ_slow + α * (θ_fast - θ_slow)
        3. Fast weights reset to slow weights:
           θ_fast = θ_slow
        4. Repeat
    
    Where:
        - k: synchronization period (how many fast steps before slow update)
        - α: slow weights interpolation coefficient (step size for slow weights)
        - θ_fast: parameters being optimized by inner optimizer
        - θ_slow: slowly moving parameters that pull fast weights back
    
    Key Innovation:
        - Fast weights explore the loss landscape aggressively
        - Slow weights provide a stable, less noisy trajectory
        - Periodic synchronization prevents oscillation and improves convergence
        
    Benefits:
        - Reduced variance in training
        - Better generalization (slow weights act as implicit regularization)
        - Works with ANY base optimizer (SGD, Adam, RMSprop, etc.)
        - Minimal computational overhead
    """
    
    def __init__(self, model, k=5, alpha=0.5):
        """
        Args:
            model: PyTorch model
            k: Synchronization period (3-10, default 5)
               - Number of fast optimizer steps before slow weight update
               - Larger k: more exploration, less frequent synchronization
               - Smaller k: more conservative, frequent synchronization
            alpha: Slow weights step size (0.3-0.8, default 0.5)
                  - Interpolation coefficient for slow weight updates
                  - Larger alpha: slow weights move faster toward fast weights
                  - Smaller alpha: slow weights move more conservatively
        """
        self.model = model
        self.k = k
        self.alpha = alpha
        
        # Initialize slow weights (cached copy of model parameters)
        self.slow_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.slow_weights[name] = param.data.clone().detach()
        
        # Track number of steps since last synchronization
        self.step_count = 0
    
    def step(self):
        """
        Perform Lookahead step after each optimizer.step()
        
        Every k steps:
        1. Update slow weights: θ_slow += α * (θ_fast - θ_slow)
        2. Reset fast weights to slow weights: θ_fast = θ_slow
        """
        self.step_count += 1
        
        # Only synchronize every k steps
        if self.step_count % self.k == 0:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue
                    
                    # Get slow and fast weights
                    slow_weight = self.slow_weights[name]
                    fast_weight = param.data
                    
                    # Update slow weights: θ_slow = θ_slow + α * (θ_fast - θ_slow)
                    # Equivalent to: θ_slow = (1 - α) * θ_slow + α * θ_fast
                    slow_weight.add_(fast_weight - slow_weight, alpha=self.alpha)
                    
                    # Reset fast weights to slow weights: θ_fast = θ_slow
                    param.data.copy_(slow_weight)
    
    def reset(self):
        """Reset slow weights and step counter"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.slow_weights[name] = param.data.clone().detach()
        self.step_count = 0

# ==========================================
# 6. TRAINING ENGINE
# ==========================================
def train_engine(model, optimizer, criterion, loader, epochs, device, 
                 use_lookahead=False, k=5, alpha=0.5, verbose=False):
    """Training loop with optional Lookahead per iteration"""
    lookahead = LookaheadHandler(model, k, alpha) if use_lookahead else None
    
    epoch_loss_history = []
    iteration_loss_history = []
    
    model.train()
    global_iter = 0
    
    # Add progress bar for epochs
    epoch_iterator = tqdm(range(epochs), desc=f"  {'Lookahead' if use_lookahead else 'Base'} Training", leave=False)
    
    for epoch in epoch_iterator:
        epoch_loss = 0.0
        
        for batch_idx, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            # Apply Lookahead AFTER optimizer step (every k steps)
            if use_lookahead:
                lookahead.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss * xb.size(0)
            
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
        
        # Update progress bar
        epoch_iterator.set_postfix({'loss': f'{epoch_loss:.2f}'})
    
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
# 7. EXPERIMENT RUNNER
# ==========================================
def run_experiment(optimizer_name, cfg, raw_X, raw_y, input_dim):
    """Run experiments for a single optimizer"""
    out_dir = os.path.join(cfg.BASE_RESULT_DIR, optimizer_name)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Running: {optimizer_name} on {cfg.DEVICE}")
    print(f"Dataset: {cfg.DATASET}")
    print(f"Model: {cfg.MODEL_TYPE.upper()}")
    print(f"Input dim: {input_dim}")
    print(f"{'='*70}")
    
    final_results = {'Generic': [], 'Lookahead': []}

    for run_id in tqdm(range(cfg.NUM_EXECUTIONS), desc=f"{optimizer_name}"):
        # Set seed for this run
        set_seed(cfg.SEED + run_id)
        
        # 1. Split & Preprocess
        X_train, X_test, y_train, y_test = train_test_split(
            raw_X, raw_y, test_size=0.2, random_state=cfg.SEED + run_id, shuffle=True
        )
        
        # Normalize
        mm_x = MinMaxScaler()
        X_train_n = mm_x.fit_transform(X_train)
        X_test_n = mm_x.transform(X_test)
        
        mm_y = MinMaxScaler()
        y_train_n = mm_y.fit_transform(y_train.reshape(-1, 1))
        y_test_n = mm_y.transform(y_test.reshape(-1, 1))
        
        Xt_train, yt_train = prepare_tensors(X_train_n, y_train_n)
        Xt_test, yt_test = prepare_tensors(X_test_n, y_test_n)
        
        loader = DataLoader(
            TensorDataset(Xt_train, yt_train), 
            batch_size=cfg.BATCH_SIZE, 
            shuffle=True
        )
        
        # 2. Initialize Master Weights
        master, _, _ = get_model_and_optimizer(
            optimizer_name, cfg.DEVICE, input_dim, cfg.MODEL_TYPE, cfg.LEARNING_RATE
        )
        init_state = copy.deepcopy(master.state_dict())
        
        # 3. Train Generic (No Lookahead)
        model_gen, opt_gen, crit = get_model_and_optimizer(
            optimizer_name, cfg.DEVICE, input_dim, cfg.MODEL_TYPE, cfg.LEARNING_RATE
        )
        model_gen.load_state_dict(copy.deepcopy(init_state))
        hist_gen_epoch, hist_gen_iter = train_engine(
            model_gen, opt_gen, crit, loader, cfg.NUM_EPOCHS, cfg.DEVICE, 
            use_lookahead=False
        )
        loss_gen = evaluate(model_gen, Xt_test, yt_test, cfg.DEVICE)
        
        # 4. Train Lookahead
        model_lookahead, opt_lookahead, crit = get_model_and_optimizer(
            optimizer_name, cfg.DEVICE, input_dim, cfg.MODEL_TYPE, cfg.LEARNING_RATE
        )
        model_lookahead.load_state_dict(copy.deepcopy(init_state))
        hist_lookahead_epoch, hist_lookahead_iter = train_engine(
            model_lookahead, opt_lookahead, crit, loader, cfg.NUM_EPOCHS, cfg.DEVICE, 
            use_lookahead=True, 
            k=cfg.LOOKAHEAD_K, 
            alpha=cfg.LOOKAHEAD_ALPHA
        )
        loss_lookahead = evaluate(model_lookahead, Xt_test, yt_test, cfg.DEVICE)
        
        # 5. Store results
        final_results['Generic'].append(loss_gen)
        final_results['Lookahead'].append(loss_lookahead)
        
        # 6. Save convergence logs
        pd.DataFrame({
            'epoch': range(len(hist_gen_epoch)),
            'generic_loss': hist_gen_epoch,
            'lookahead_loss': hist_lookahead_epoch
        }).to_csv(f"{out_dir}/run_{run_id+1}_epoch_losses.csv", index=False)

    # 7. Statistical Analysis
    summary_df = pd.DataFrame(final_results)
    summary_df.to_csv(f"{out_dir}/{optimizer_name}_summary.csv", index=False)

    mean_gen = np.mean(final_results['Generic'])
    std_gen = np.std(final_results['Generic'])
    mean_lookahead = np.mean(final_results['Lookahead'])
    std_lookahead = np.std(final_results['Lookahead'])
    
    t_stat, p_val = stats.ttest_rel(final_results['Generic'], final_results['Lookahead'])
    improvement_pct = ((mean_gen - mean_lookahead) / mean_gen * 100)

    # 8. Generate Report
    report = f"""
            {'='*70}
            EXPERIMENT SUMMARY: {optimizer_name}
            {'='*70}
            Configuration:
              Dataset: {cfg.DATASET.upper()}
              Model: {cfg.MODEL_TYPE.upper()}
              Input dim: {input_dim}
              Epochs: {cfg.NUM_EPOCHS}
              Batch Size: {cfg.BATCH_SIZE}
              Learning Rate: {cfg.LEARNING_RATE}
              Runs: {cfg.NUM_EXECUTIONS}
              Device: {cfg.DEVICE}
            
            Lookahead Parameters:
              k (sync period): {cfg.LOOKAHEAD_K} steps
              α (slow step size): {cfg.LOOKAHEAD_ALPHA}
            
            Results (Test MSE over {cfg.NUM_EXECUTIONS} runs):
              {optimizer_name} (Base):          {mean_gen:.5f} ± {std_gen:.5f}
              {optimizer_name} + Lookahead:     {mean_lookahead:.5f} ± {std_lookahead:.5f}
            
            Statistical Analysis:
              Improvement: {(mean_gen - mean_lookahead):.5f} MSE ({improvement_pct:+.2f}%)
              t-statistic: {t_stat:.4f}
              p-value: {p_val:.6f}
              Significant: {'YES (p < 0.05)' if p_val < 0.05 else 'NO (p >= 0.05)'}
            
            Individual Results:
              Generic:    {final_results['Generic']}
              Lookahead:  {final_results['Lookahead']}
            {'='*70}
            """
    
    with open(f"{out_dir}/{optimizer_name}_report.txt", "w") as f:
        f.write(report)
    print(report)
    
    return {
        'optimizer': optimizer_name,
        'mean_generic': mean_gen,
        'std_generic': std_gen,
        'mean_lookahead': mean_lookahead,
        'std_lookahead': std_lookahead,
        'improvement_pct': improvement_pct,
        'p_value': p_val,
        'significant': p_val < 0.05
    }

# ==========================================
# 8. MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # Get available datasets
    available_datasets = DatasetConfig.get_available_datasets()
    
    parser = argparse.ArgumentParser(description='Lookahead Optimizer Evaluation')
    parser.add_argument('--optimizer', type=str, default='All', help='Optimizer name or "All"')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--dataset', type=str, default='boston', choices=available_datasets, help=f'Dataset name. Available: {", ".join(available_datasets)}')
    parser.add_argument('--model', type=str, default='mlp', choices=['linear', 'mlp', 'lstm', 'gru'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--output', type=str, default='./results', help='Path to the folder where results will be saved')
    parser.add_argument('--k', type=int, default=5, help='Lookahead synchronization period (3-10)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Lookahead slow weights step size (0.3-0.8)')
    
    args = parser.parse_args()
    
    # Initialize Config
    cfg = Config(output_dir=args.output, optimizer_name=args.optimizer, gpu_id=args.gpu, dataset=args.dataset, model_type=args.model)
    cfg.NUM_EPOCHS = args.epochs
    cfg.NUM_EXECUTIONS = args.runs
    cfg.LOOKAHEAD_K = args.k
    cfg.LOOKAHEAD_ALPHA = args.alpha
    
    set_seed(cfg.SEED)
    
    # Load Data
    print(f"\n{'='*70}")
    print(f"LOADING DATASET: {cfg.DATASET}")
    print(f"{'='*70}")
    
    # Get ACTUAL input dimension from the loaded data (not from config)
    raw_X, raw_y, input_dim = load_and_process_data(cfg.DATASET)
    
    print(f"\n{'='*70}")
    print(f"DATA LOADED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"  Total samples: {len(raw_X)}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Model: {cfg.MODEL_TYPE.upper()}")
    print(f"{'='*70}\n")
    
    # Define Optimizers
    ALL_OPTIMIZERS = ['SGD', 'Momentum', 'NAG', 'AdaGrad', 'RMSprop', 'AdaDelta','Adam', 'AdaMax', 'NAdam', 'AdamW', 'RAdam', 'ASGD', 'Rprop']
    
    if cfg.OPTIMIZER_NAME == 'All':
        run_list = ALL_OPTIMIZERS
    else:
        if cfg.OPTIMIZER_NAME not in ALL_OPTIMIZERS:
            print(f"Error: {cfg.OPTIMIZER_NAME} not in supported list: {ALL_OPTIMIZERS}")
            exit(1)
        run_list = [cfg.OPTIMIZER_NAME]
    
    print(f"Running optimizers: {', '.join(run_list)}\n")
    
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
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {cfg.BASE_RESULT_DIR}")
    print(f"{'='*70}")
