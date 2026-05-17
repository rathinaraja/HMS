# HMS to HBF Migration Guide

## Complete Replacement Summary

This document shows all changes made to replace HMS (Harmonic Mean-based Scalar) with HBF (Heavy-Ball with Friction).

---

## 1. Configuration Changes

### OLD (HMS):
```python
class Config:
    def __init__(self, ...):
        self.HMS_R = 1.0              # Harmonic mean scaling
        self.HMS_T = 1000             # Decay interval (iterations)
        self.HMS_DECAY = 0.9          # Decay rate
```

### NEW (HBF):
```python
class Config:
    def __init__(self, ...):
        self.HBF_BETA = 0.9           # Momentum coefficient (0.8-0.95)
        self.HBF_GAMMA = 0.1          # Friction coefficient (0.05-0.2)
        self.HBF_DECAY_INTERVAL = 1000  # Gamma decay interval (iterations)
        self.HBF_GAMMA_DECAY = 0.95   # Gamma decay rate (0.9-0.99)
```

---

## 2. Handler Class Replacement

### OLD (HMSHandler):
```python
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
                
                # Harmonic mean calculation
                mask_nonzero = (w_prev != 0) & (w_curr != 0)
                if not mask_nonzero.any():
                    continue
                
                prev_nz = w_prev[mask_nonzero]
                curr_nz = w_curr[mask_nonzero]
                
                abs_prev = torch.abs(prev_nz)
                abs_curr = torch.abs(curr_nz)
                hm = 2 * abs_prev * abs_curr / (abs_prev + abs_curr)
                
                min_abs = torch.minimum(abs_prev, abs_curr)
                hms_scalar = torch.abs(hm - min_abs) * self.r
                
                result_nz = curr_nz.clone()
                mask_dec = (prev_nz > curr_nz)
                result_nz[mask_dec] = curr_nz[mask_dec] - hms_scalar[mask_dec]
                
                mask_inc = (prev_nz < curr_nz)
                result_nz[mask_inc] = curr_nz[mask_inc] + hms_scalar[mask_inc]
                
                p.data[mask_nonzero] = result_nz
                self.prev_w[i] = p.data.clone()
        
        self.iteration += 1
        if self.iteration % self.t == 0:
            self.r = round(self.r * self.decay_rate, 4)
```

### NEW (HBFHandler):
```python
class HBFHandler:
    """
    Heavy-Ball with Friction (HBF) Handler - Applied Per Iteration
    
    Based on: https://arxiv.org/pdf/2202.05928
    
    Algorithm:
        v_t = β * v_{t-1} + (1 - β) * g_t - γ * sign(v_{t-1}) * |g_t|
        w_{t+1} = w_t - α * v_t
    """
    
    def __init__(self, model, beta=0.9, gamma=0.1, decay_gamma=True, 
                 gamma_decay_rate=0.95, decay_interval=1000):
        self.model = model
        self.beta = beta
        self.gamma = gamma
        self.initial_gamma = gamma
        self.decay_gamma = decay_gamma
        self.gamma_decay_rate = gamma_decay_rate
        self.decay_interval = decay_interval
        self.iteration = 0
        
        # Initialize velocity for each parameter
        self.velocity = [torch.zeros_like(p.data) for p in model.parameters()]

    def step(self):
        """Apply HBF after each iteration (batch)"""
        with torch.no_grad():
            for i, param in enumerate(self.model.parameters()):
                if param.grad is None:
                    continue
                
                g_t = param.grad.data
                v_prev = self.velocity[i]
                
                # Heavy-Ball with Friction update
                momentum_term = self.beta * v_prev
                gradient_term = (1 - self.beta) * g_t
                friction_term = self.gamma * torch.sign(v_prev) * torch.abs(g_t)
                
                v_t = momentum_term + gradient_term - friction_term
                
                self.velocity[i] = v_t.clone()
                param.data.sub_(v_t)
        
        self.iteration += 1
        
        if self.decay_gamma and self.iteration % self.decay_interval == 0:
            self.gamma = max(0.01, self.gamma * self.gamma_decay_rate)
```

---

## 3. Training Engine Changes

### OLD (HMS):
```python
def train_engine(model, optimizer, criterion, loader, epochs, device, 
                 use_hms=False, r=1.0, t=1000, decay=0.9):
    hms = HMSHandler(model, r, t, decay) if use_hms else None
    
    # ... training loop
    
    if use_hms:
        hms.step()
```

### NEW (HBF):
```python
def train_engine(model, optimizer, criterion, loader, epochs, device, 
                 use_hbf=False, beta=0.9, gamma=0.1, decay_interval=1000, 
                 gamma_decay=0.95, verbose=False):
    hbf = HBFHandler(
        model, 
        beta=beta, 
        gamma=gamma, 
        decay_gamma=True,
        gamma_decay_rate=gamma_decay,
        decay_interval=decay_interval
    ) if use_hbf else None
    
    # ... training loop
    
    if use_hbf:
        hbf.step()
```

---

## 4. Experiment Runner Changes

### OLD (HMS):
```python
hist_hms_epoch, hist_hms_iter = train_engine(
    model_hms, opt_hms, crit, loader, cfg.NUM_EPOCHS, cfg.DEVICE, 
    use_hms=True, r=cfg.HMS_R, t=cfg.HMS_T, decay=cfg.HMS_DECAY
)

final_results = {'Generic': [], 'HMS': []}
final_results['HMS'].append(loss_hms)
```

### NEW (HBF):
```python
hist_hbf_epoch, hist_hbf_iter = train_engine(
    model_hbf, opt_hbf, crit, loader, cfg.NUM_EPOCHS, cfg.DEVICE, 
    use_hbf=True, 
    beta=cfg.HBF_BETA, 
    gamma=cfg.HBF_GAMMA, 
    decay_interval=cfg.HBF_DECAY_INTERVAL,
    gamma_decay=cfg.HBF_GAMMA_DECAY
)

final_results = {'Generic': [], 'HBF': []}
final_results['HBF'].append(loss_hbf)
```

---

## 5. Command-Line Arguments

### NEW Arguments Added:
```python
parser.add_argument('--beta', type=float, default=0.9,
                   help='HBF momentum coefficient (0.8-0.95)')
parser.add_argument('--gamma', type=float, default=0.1,
                   help='HBF friction coefficient (0.05-0.2)')
```

---

## 6. Default Values Summary

| Parameter | Default Value | Range | Description |
|-----------|---------------|-------|-------------|
| `HBF_BETA` | 0.9 | 0.8 - 0.95 | Momentum coefficient (heavy-ball) |
| `HBF_GAMMA` | 0.1 | 0.05 - 0.2 | Friction coefficient (stabilization) |
| `HBF_DECAY_INTERVAL` | 1000 | 500 - 2000 | Decay gamma every N iterations |
| `HBF_GAMMA_DECAY` | 0.95 | 0.9 - 0.99 | Gamma decay rate |

---

## 7. Usage Examples

### Quick Test (10 epochs, 1 run):
```bash
python Deep_learning_HBF.py --optimizer Adam --epochs 10 --runs 1
```

### Custom HBF Parameters:
```bash
python Deep_learning_HBF.py --optimizer Adam --epochs 1000 --runs 5 \
    --beta 0.85 --gamma 0.15
```

### Yahoo Finance Time Series with LSTM:
```bash
python Deep_learning_HBF.py --dataset yahoo --model lstm --ticker AAPL \
    --lookback 10 --epochs 1000 --runs 5 --optimizer Adam
```

### All Optimizers (Full Experiment):
```bash
python Deep_learning_HBF.py --optimizer All --epochs 1000 --runs 5
```

---

## 8. Key Algorithm Differences

### HMS (Harmonic Mean-based Scalar):
- **Operates in**: Weight space
- **Core mechanism**: Harmonic mean smoothing of weight changes
- **Key idea**: Outlier-resistant averaging
- **Equation**: `hms = |HM(|w_prev|, |w_curr|) - min(|w_prev|, |w_curr|)| × r`

### HBF (Heavy-Ball with Friction):
- **Operates in**: Velocity/momentum space
- **Core mechanism**: Friction opposes velocity direction
- **Key idea**: Physics-inspired damping
- **Equation**: `v_t = β·v_{t-1} + (1-β)·g_t - γ·sign(v_{t-1})·|g_t|`

---

## 9. Expected Performance

Both methods aim to stabilize training and prevent overshooting. HBF may:
- Converge faster in early epochs (momentum acceleration)
- Be more stable in later epochs (friction damping)
- Work better with adaptive optimizers (Adam, RMSprop)
- Require less hyperparameter tuning (only 2 main params: β, γ)

---

## 10. Reproducibility Notes

✅ **Preserved**:
- Random seed initialization
- Data splitting strategy
- Model architectures
- Optimizer configurations
- Training loop structure
- Evaluation metrics
- Result saving format

❌ **Changed**:
- Regularization mechanism (HMS → HBF)
- Parameter names (HMS_R → HBF_BETA, HBF_GAMMA)
- Handler implementation

**Note**: Results will differ between HMS and HBF due to different regularization mechanisms, but the experimental setup remains identical for fair comparison.

---

## 11. File Structure

```
Deep_learning_results/
└── {dataset}_{model}/
    └── {optimizer}/
        ├── run_1_epoch_losses.csv          # Base vs HBF per epoch
        ├── run_1_generic_iteration_losses.csv
        ├── run_1_hbf_iteration_losses.csv
        ├── run_2_epoch_losses.csv
        ├── ...
        ├── {optimizer}_summary.csv         # All runs summary
        └── {optimizer}_report.txt          # Statistical analysis
```

---

## 12. Quick Reference: Parameter Selection

### Conservative (Stable):
```python
beta = 0.9      # Standard momentum
gamma = 0.05    # Low friction
```

### Balanced (Default):
```python
beta = 0.9      # Standard momentum
gamma = 0.1     # Medium friction
```

### Aggressive (Fast Convergence):
```python
beta = 0.95     # High momentum
gamma = 0.15    # High friction
```

---

## Conclusion

The HBF implementation is a **drop-in replacement** for HMS with:
- ✅ Cleaner algorithm (physics-inspired)
- ✅ Fewer edge cases (no zero-weight handling)
- ✅ Better theoretical foundation
- ✅ Simpler hyperparameter space
- ✅ Maintained reproducibility setup

All experimental infrastructure remains unchanged for valid comparison!
