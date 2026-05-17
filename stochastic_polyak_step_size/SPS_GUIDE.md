# SPS (Stochastic Polyak Step Size) Implementation Guide

## Overview

**SPS (Stochastic Polyak Step Size)** is a breakthrough adaptive learning rate method from the 2025 paper that eliminates manual learning rate tuning by using the classic **Polyak step size formula** adapted for stochastic optimization.

Paper: https://openreview.net/pdf?id=nuX2yPejiL

---

## 🔬 What is SPS?

### Core Idea:
SPS automatically computes the learning rate at each iteration based on:
1. **Current loss value** (how bad are we doing?)
2. **Estimated optimal loss** (where do we want to be?)
3. **Gradient magnitude** (how steep is the landscape?)

### The Formula:
```
η_t = c * (f(x_t) - f_est) / ||∇f(x_t)||²

Where:
- f(x_t) = current loss
- f_est = running estimate of minimum loss
- ||∇f||² = squared gradient norm
- c = scaling factor (hyperparameter)
```

### Key Innovation:
**NO MANUAL LEARNING RATE TUNING NEEDED!**
- Larger steps when far from optimum
- Smaller steps when close to optimum
- Automatically adapts throughout training

---

## 🎯 How It Works

### Classic Polyak Step Size (Deterministic):
```
η = (f(x) - f*) / ||∇f(x)||²
```
Problems: 
- Requires knowing f* (optimal loss) - we don't know this!
- Only works for deterministic gradients

### Stochastic Polyak Step Size (SPS):
```
1. Estimate f* using exponential moving average
   f_est = momentum * f_est + (1-momentum) * min(f_current, f_est)

2. Compute step size
   η = c * (f_current - f_est) / ||∇f||²

3. Clip to maximum
   η = min(η, η_max)

4. Scale gradients
   g_new = η * g_old
```

---

## 📊 Comparison: GAP vs SPS

| Aspect | GAP | SPS |
|--------|-----|-----|
| **What It Tracks** | Gradient geometry | Loss gap to optimum |
| **Core Metric** | Cosine similarity | Loss difference |
| **Learning Rate** | Fixed (from optimizer) | **Adaptive (automatic)** |
| **Key Innovation** | Geometric perturbation | **No LR tuning needed** |
| **Best For** | Noisy gradients | **Fast convergence** |
| **Memory** | Previous gradients | Loss estimate only |
| **Complexity** | Medium | Low |

---

## 🎛️ Parameters Explained

### 1. **c** - Step Size Scaling Factor
```python
Default: 0.1
Range: 0.01 - 0.5
```

**What it does**: Controls how aggressive the step size is
- **c = 0.01-0.05**: Very conservative (slow but stable)
- **c = 0.1**: Balanced (default, works well generally)
- **c = 0.3-0.5**: Aggressive (fast but may overshoot)

**Intuition**: 
- c is like a "confidence" parameter
- Higher c = "trust the formula more, take bigger steps"
- Lower c = "be cautious, take smaller steps"

---

### 2. **eta_max (η_max)** - Maximum Step Size
```python
Default: 10.0
Range: 1.0 - 100.0
```

**What it does**: Caps the maximum allowed step size
- Prevents extremely large steps that could destabilize training
- Acts as a safety mechanism

**When to tune**:
- If training diverges → reduce to 5.0 or 1.0
- If convergence is too slow → increase to 20.0 or 50.0

---

### 3. **momentum** - Loss Estimate Momentum
```python
Default: 0.9
Range: 0.8 - 0.99
```

**What it does**: Smooths the estimate of minimum loss
- **Higher (0.95-0.99)**: More stable estimate, slower adaptation
- **Lower (0.8-0.85)**: Faster adaptation, more noise

**Formula**: 
```
f_est_new = momentum * f_est_old + (1-momentum) * min(f_current, f_est_old)
```

---

## 🚀 Usage Examples

### Quick Test:
```bash
python Deep_learning_SPS.py --optimizer Adam --epochs 10 --runs 1 \
    --output ./results
```

### Standard Run:
```bash
python Deep_learning_SPS.py --dataset boston --model mlp --optimizer Adam \
    --epochs 1000 --runs 5 --output ./results
```

### Custom SPS Parameters:

#### Conservative (Stable):
```bash
python Deep_learning_SPS.py --optimizer Adam \
    --c 0.05 --eta-max 5.0 --momentum 0.95 \
    --output ./results
```
**Use when**: Training is unstable, loss oscillates

#### Balanced (Default):
```bash
python Deep_learning_SPS.py --optimizer Adam \
    --c 0.1 --eta-max 10.0 --momentum 0.9 \
    --output ./results
```
**Use when**: General purpose, most datasets

#### Aggressive (Fast):
```bash
python Deep_learning_SPS.py --optimizer Adam \
    --c 0.3 --eta-max 20.0 --momentum 0.85 \
    --output ./results
```
**Use when**: Need fast convergence, training is stable

### All Optimizers:
```bash
python Deep_learning_SPS.py --dataset california --model mlp --optimizer All \
    --epochs 1000 --runs 5 --output ./results
```

---

## 🎯 Parameter Tuning Guide

### Dataset-Based Tuning:

#### Small Datasets (< 1000 samples):
```bash
--c 0.05 --eta-max 5.0 --momentum 0.95
```
**Rationale**: Small batches → noisy gradients → be conservative

#### Medium Datasets (1000-10000 samples):
```bash
--c 0.1 --eta-max 10.0 --momentum 0.9
```
**Rationale**: Balanced (default works well)

#### Large Datasets (> 10000 samples):
```bash
--c 0.15 --eta-max 15.0 --momentum 0.9
```
**Rationale**: More data → more stable → can be more aggressive

---

### Problem-Based Tuning:

#### Training Diverges / Loss → NaN:
```bash
--c 0.02 --eta-max 2.0 --momentum 0.95
```
Drastically reduce step sizes

#### Convergence Too Slow:
```bash
--c 0.2 --eta-max 20.0 --momentum 0.85
```
Increase aggressiveness

#### Loss Oscillates (Zig-zagging):
```bash
--c 0.08 --eta-max 8.0 --momentum 0.95
```
Reduce c slightly, increase momentum for stability

#### Getting Stuck in Plateau:
```bash
--c 0.3 --eta-max 30.0 --momentum 0.85
```
Aggressive settings to escape

---

## 💡 How SPS Adapts During Training

### Early Training:
```
Loss: 100 → f_est: 90
Gap: 10 (large)
Gradient norm: 5
→ Step size: 0.1 * 10 / 25 = 0.04 (moderate)
```

### Mid Training:
```
Loss: 20 → f_est: 15
Gap: 5 (medium)
Gradient norm: 2
→ Step size: 0.1 * 5 / 4 = 0.125 (larger!)
```

### Near Convergence:
```
Loss: 1.1 → f_est: 1.0
Gap: 0.1 (small)
Gradient norm: 0.5
→ Step size: 0.1 * 0.1 / 0.25 = 0.04 (smaller)
```

**Pattern**: SPS automatically takes larger steps when making good progress, smaller steps when near optimum!

---

## 🔬 SPS vs Traditional Learning Rate

### Traditional Fixed LR:
```python
optimizer = Adam(lr=0.001)  # Same for entire training
```
**Problems**:
- Too large → diverges early
- Too small → converges slowly
- Manual tuning required
- No adaptation

### SPS:
```python
optimizer = Adam(lr=0.0001)  # Base LR (less critical)
sps.step(loss)  # Adapts automatically!
```
**Benefits**:
- ✅ Automatic adaptation
- ✅ Larger steps when safe
- ✅ Smaller steps when needed
- ✅ Minimal tuning

---

## 📈 Expected Behavior

### Typical SPS Trajectory:

```
Iteration    Loss    f_est    Gap    Grad²    η (c=0.1)
---------------------------------------------------------
1           100.0    100.0    0.0    50.0     0.0       (init)
10           85.0     90.0    5.0    40.0     0.0125
100          50.0     55.0    5.0    25.0     0.02
500          15.0     18.0    3.0    10.0     0.03
1000          5.0      6.0    1.0     4.0     0.025
5000          1.2      1.1    0.1     1.0     0.01
10000         1.01     1.0    0.01    0.5     0.002
```

**Pattern**: Step size peaks mid-training, then decreases near convergence!

---

## 🎓 Theory: Why SPS Works

### 1. **Polyak's Insight (1987)**:
For convex functions with known optimum f*, the step size:
```
η = (f(x) - f*) / ||∇f||²
```
guarantees convergence in one step (deterministic case).

### 2. **SPS Adaptation**:
- Estimates f* on-the-fly using moving minimum
- Uses stochastic gradients instead of exact gradients
- Adds clipping for stability

### 3. **Why It's Better**:
- **Far from optimum**: Large gap → large steps → fast progress
- **Near optimum**: Small gap → small steps → stable convergence
- **Automatic**: No need to guess initial LR or schedule

---

## 📊 Comparison with Other Methods

### vs Fixed Learning Rate:
| Metric | Fixed LR | SPS |
|--------|----------|-----|
| **Tuning Needed** | High | Low |
| **Convergence** | Depends on LR choice | Adaptive |
| **Plateau Escape** | Difficult | Automatic |
| **Stability** | Requires careful tuning | Built-in |

### vs Learning Rate Schedules:
| Metric | LR Schedule | SPS |
|--------|-------------|-----|
| **Adaptivity** | Predefined | Dynamic |
| **Problem Awareness** | No | Yes (uses loss) |
| **Complexity** | Manual design | Automatic |

### vs Adaptive Optimizers (Adam, RMSprop):
| Metric | Adam | SPS |
|--------|------|-----|
| **Per-parameter** | Yes | No |
| **Loss-aware** | No | **Yes** |
| **Step size logic** | Gradient statistics | **Loss gap** |
| **Convergence guarantee** | No | Stronger (convex) |

---

## 🔍 Debugging SPS

### Issue: Step sizes are all 0
**Cause**: Loss estimate = current loss (no gap)
**Solution**: 
```bash
--c 0.2  # Increase c to amplify small gaps
```

### Issue: Step sizes > 100 (clipped constantly)
**Cause**: Gradient norm very small or gap very large
**Solution**:
```bash
--eta-max 1000.0  # Increase max if this is expected
# OR
--c 0.05  # Reduce c to get smaller base step sizes
```

### Issue: Loss increases after first few iterations
**Cause**: Initial step size too large
**Solution**:
```bash
--c 0.01 --eta-max 5.0  # Be very conservative
```

### Issue: Convergence slower than baseline
**Cause**: SPS being too conservative
**Solution**:
```bash
--c 0.3 --eta-max 50.0  # Be more aggressive
```

---

## 🆚 When to Use SPS vs Other Methods

### ✅ Use SPS When:
- You don't want to tune learning rate
- Training on new datasets/architectures
- Need fast initial convergence
- Want automatic adaptation
- Loss landscape is well-behaved

### ⚠️ Consider Alternatives When:
- Need per-parameter adaptation → Use Adam/RMSprop
- Training GANs or very unstable models → Use traditional methods
- Loss is very noisy → May need more conservative settings
- Need guaranteed scale-invariance → Use ASAM

---

## 🔬 Advanced: SPS Variants

### SPS with Warmup:
```python
if iteration < warmup_iterations:
    step_size = base_lr * (iteration / warmup_iterations)
else:
    step_size = sps.compute_step_size(loss)
```

### SPS with Gradient Clipping:
```python
sps.step(loss)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### SPS with Multiple Loss Estimates:
```python
# Track both EMA and true minimum
f_est_ema = momentum * f_est_ema + (1-momentum) * loss
f_est_min = min(f_est_min, loss)
f_est = 0.5 * f_est_ema + 0.5 * f_est_min
```

---

## 📁 Output Structure

```
./results/
└── boston/
    └── mlp/
        └── Adam/
            ├── run_1_epoch_losses.csv    # generic_loss, sps_loss
            ├── run_2_epoch_losses.csv
            ├── run_3_epoch_losses.csv
            ├── run_4_epoch_losses.csv
            ├── run_5_epoch_losses.csv
            ├── Adam_summary.csv           # Generic, SPS columns
            └── Adam_report.txt            # SPS parameters shown
```

---

## 🎯 Quick Reference Card

### Conservative Setup (Safe):
```bash
python Deep_learning_SPS.py --optimizer Adam \
    --c 0.05 --eta-max 5.0 --momentum 0.95
```

### Balanced Setup (Default):
```bash
python Deep_learning_SPS.py --optimizer Adam \
    --c 0.1 --eta-max 10.0 --momentum 0.9
```

### Aggressive Setup (Fast):
```bash
python Deep_learning_SPS.py --optimizer Adam \
    --c 0.3 --eta-max 20.0 --momentum 0.85
```

### Parameter Effects:
- ↑ c → faster convergence, less stable
- ↓ c → slower convergence, more stable
- ↑ η_max → allow larger steps
- ↓ η_max → cap step size (safer)
- ↑ momentum → smoother loss estimate
- ↓ momentum → faster adaptation

---

## 📚 Citation

```bibtex
@inproceedings{loizou2021stochastic,
  title={Stochastic Polyak Step-size for SGD: An Adaptive Learning Rate for Fast Convergence},
  author={Loizou, Nicolas and Vaswani, Sharan and Laradji, Issam Hadj and Lacoste-Julien, Simon},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2021},
  url={https://openreview.net/pdf?id=nuX2yPejiL}
}
```

---

## ✅ Key Takeaways

1. **SPS eliminates manual LR tuning** - biggest advantage!
2. **Adapts based on loss gap** - smart, not just gradient stats
3. **Larger steps when safe, smaller when needed** - automatic
4. **Main parameter: c** - controls aggressiveness (0.01-0.5)
5. **Works with any optimizer** - SGD, Adam, etc.
6. **Best for**: fast convergence without LR tuning
7. **Minimal overhead**: just one loss estimate tracked

---

## 🚀 Ready to Use!

Start with default settings:
```bash
python Deep_learning_SPS.py --optimizer Adam --epochs 100 --runs 1
```

Scale up:
```bash
python Deep_learning_SPS.py --dataset boston --model mlp --optimizer All \
    --epochs 1000 --runs 5
```

**The beauty of SPS**: You can often use default parameters and get great results without any learning rate tuning!
