# SlowMo (Slow Momentum) Implementation Guide

## Overview

**SlowMo (Slow Momentum)** is an optimization technique from the 2020 paper that applies momentum **directly to parameters** (not gradients) to stabilize training. Originally designed for distributed training across multiple workers, it also provides significant benefits in single-worker settings.

Paper: https://arxiv.org/pdf/1910.00643  
Authors: Wang et al., 2020

---

## 🔬 What is SlowMo?

### Core Idea:
SlowMo applies **momentum to the parameters themselves**, creating a "slow momentum" that accumulates parameter changes over time and stabilizes the optimization trajectory.

### Key Innovation:
```
Traditional Momentum: Applied to GRADIENTS
  v_t = β*v_{t-1} + g_t
  θ_t = θ_{t-1} - η*v_t

SlowMo: Applied to PARAMETERS
  Δθ = θ_current - θ_cached
  v_t = β*v_{t-1} + Δθ
  θ_t = θ_t + η_slow*v_t
```

**The difference**: SlowMo tracks how parameters are moving (not just gradients), then applies momentum to smooth that trajectory.

---

## 🎯 The Algorithm

### Every k iterations:

```
1. Compute parameter change:
   Δθ = θ_current - θ_{k steps ago}
   
2. Update slow momentum buffer:
   v = β*v_old + Δθ
   
3. Apply slow momentum to parameters:
   θ = θ + η_slow*v
   
4. Cache current parameters for next period
```

### Visual Timeline:
```
Step 1-10:  Local optimizer updates θ (Adam, SGD, etc.)
            Cache θ_0 at start

Step 10:    Δθ = θ_10 - θ_0
            v_10 = β*v_0 + Δθ
            θ_10 = θ_10 + η_slow*v_10
            Cache θ_10

Step 11-20: Local optimizer continues...
            
Step 20:    Repeat slow momentum update...
```

---

## 📊 Comparison: Lookahead vs SlowMo

| Aspect | Lookahead | SlowMo |
|--------|-----------|--------|
| **Core Mechanism** | Fast-slow weight interpolation | **Momentum on parameters** |
| **Update Rule** | θ_slow += α(θ_fast - θ_slow) | **v = β*v + Δθ; θ += η*v** |
| **Memory** | Slow weights copy | **Momentum buffer** |
| **Key Difference** | Interpolation | **Momentum accumulation** |
| **Designed For** | General optimization | **Distributed training** |
| **Main Benefit** | Variance reduction | **Trajectory stabilization** |
| **Parameters** | k, α | **β, k, η_slow** |

---

## 🎛️ Parameters Explained

### 1. **β (beta)** - Slow Momentum Coefficient
```python
Default: 0.6
Range: 0.4 - 0.8
```

**What it does**: Controls how much past momentum is retained

**Formula**: `v_t = β*v_{t-1} + Δθ`

**Intuition**:
- **β = 0.4**: Low memory, responds quickly to changes
- **β = 0.6**: Balanced (default, works well)
- **β = 0.8**: High memory, very smooth but slow to adapt

**Examples**:
```
If v_old = 10, Δθ = 5:
  β = 0.4: v_new = 0.4*10 + 5 = 9   (40% old, 100% new)
  β = 0.6: v_new = 0.6*10 + 5 = 11  (60% old, 100% new)
  β = 0.8: v_new = 0.8*10 + 5 = 13  (80% old, 100% new)
```

**When to tune**:
- Training unstable → **increase to 0.7-0.8** (more smoothing)
- Training too slow → **decrease to 0.4-0.5** (less smoothing)

---

### 2. **k** - Synchronization Period
```python
Default: 10
Range: 5 - 20
```

**What it does**: How many local optimizer steps between slow momentum updates

**Intuition**:
- **k = 5**: Frequent momentum updates (more overhead, more stable)
- **k = 10**: Balanced (default)
- **k = 20**: Rare updates (less overhead, more exploration)

**Trade-off**:
```
Small k (5):   [Local×5→SlowMo] [Local×5→SlowMo] ...
               ↑ More frequent stabilization
               ↑ More overhead
               
Large k (20):  [Local×20→SlowMo] [Local×20→SlowMo] ...
               ↑ Less frequent stabilization
               ↑ Less overhead
               ↑ More exploration
```

**When to tune**:
- Small datasets → **k = 5-7** (need frequent stabilization)
- Large datasets → **k = 15-20** (can afford exploration)

---

### 3. **η_slow (lr_slow)** - Slow Momentum Learning Rate
```python
Default: 1.0
Range: 0.5 - 2.0
```

**What it does**: Step size for applying slow momentum

**Formula**: `θ = θ + η_slow*v`

**Intuition**:
- **η_slow = 0.5**: Conservative, gentle momentum effect
- **η_slow = 1.0**: Balanced (default)
- **η_slow = 2.0**: Aggressive, strong momentum effect

**When to tune**:
- Momentum too weak → **increase to 1.5-2.0**
- Training unstable → **decrease to 0.5-0.7**

---

## 🚀 Usage Examples

### Quick Test:
```bash
python Deep_learning_SlowMo.py --optimizer Adam --epochs 10 --runs 1 \
    --output ./results
```

### Standard Run:
```bash
python Deep_learning_SlowMo.py --dataset boston --model mlp --optimizer Adam \
    --epochs 1000 --runs 5 --output ./results
```

### Custom SlowMo Parameters:

#### Conservative (Maximum Stability):
```bash
python Deep_learning_SlowMo.py --optimizer Adam \
    --beta 0.8 --k 5 --lr-slow 0.5 \
    --output ./results
```
**Use when**: Training is very unstable

#### Balanced (Default):
```bash
python Deep_learning_SlowMo.py --optimizer Adam \
    --beta 0.6 --k 10 --lr-slow 1.0 \
    --output ./results
```
**Use when**: General purpose

#### Aggressive (Maximum Speed):
```bash
python Deep_learning_SlowMo.py --optimizer Adam \
    --beta 0.4 --k 20 --lr-slow 1.5 \
    --output ./results
```
**Use when**: Training is stable, want fast convergence

---

## 🎯 Parameter Tuning Guide

### Dataset-Based Tuning:

#### Small Datasets (< 1000 samples):
```bash
--beta 0.7 --k 5 --lr-slow 0.8
```
**Rationale**: Noisy gradients → need strong smoothing

#### Medium Datasets (1000-10000 samples):
```bash
--beta 0.6 --k 10 --lr-slow 1.0
```
**Rationale**: Balanced (default works well)

#### Large Datasets (> 10000 samples):
```bash
--beta 0.5 --k 15 --lr-slow 1.2
```
**Rationale**: More data → less need for smoothing

---

### Problem-Based Tuning:

#### Training is Unstable / Loss Oscillates:
```bash
--beta 0.8 --k 5 --lr-slow 0.5
```
Strong smoothing + frequent updates + gentle application

#### Convergence Too Slow:
```bash
--beta 0.4 --k 15 --lr-slow 1.5
```
Weak smoothing + rare updates + aggressive application

#### Stuck in Local Minimum:
```bash
--beta 0.3 --k 20 --lr-slow 2.0
```
Minimal smoothing to allow exploration

#### Overfitting:
```bash
--beta 0.7 --k 8 --lr-slow 0.8
```
Slow momentum acts as implicit regularization

---

### Base Optimizer Recommendations:

#### With SGD:
```bash
--beta 0.6 --k 10 --lr-slow 1.0
```
Standard settings work well

#### With Adam:
```bash
--beta 0.5 --k 12 --lr-slow 1.0
```
Adam already smooth, less momentum needed

#### With RMSprop:
```bash
--beta 0.6 --k 10 --lr-slow 0.8
```
Similar to Adam

#### With Momentum SGD:
```bash
--beta 0.4 --k 15 --lr-slow 1.0
```
Already has gradient momentum, less parameter momentum needed

---

## 💡 How SlowMo Works (Detailed Example)

### Setup: β=0.6, k=10, η_slow=1.0

```
Initialization:
  θ_0 = initial weights
  v_0 = zeros (momentum buffer)
  cached_θ = θ_0

Iterations 1-10 (Local optimizer runs):
  θ_0 --Adam--> θ_1 --Adam--> ... --Adam--> θ_10
  v stays at v_0 (no update yet)

Iteration 10 (SlowMo update):
  1. Δθ = θ_10 - cached_θ = θ_10 - θ_0
  2. v_10 = 0.6*v_0 + Δθ = Δθ (since v_0=0)
  3. θ_10 = θ_10 + 1.0*v_10 = θ_10 + Δθ
  4. cached_θ = θ_10

Iterations 11-20 (Local continues):
  θ_10 --Adam--> θ_11 --Adam--> ... --Adam--> θ_20

Iteration 20 (SlowMo update):
  1. Δθ = θ_20 - θ_10
  2. v_20 = 0.6*v_10 + Δθ
  3. θ_20 = θ_20 + 1.0*v_20
  4. cached_θ = θ_20
  ...
```

**Key**: Momentum accumulates **parameter changes**, not gradient information!

---

## 🎓 Theory: Why SlowMo Works

### 1. **Variance Reduction**
Local optimizer sees noisy gradients → parameters jump around
SlowMo accumulates these jumps with momentum → smooth trajectory

### 2. **Trajectory Stabilization**
Parameter momentum creates a "preferred direction" based on past movement
Prevents wild oscillations in parameter space

### 3. **Implicit Regularization**
Smooth parameter updates prevent overfitting to individual mini-batches
Similar effect to weight averaging

### 4. **Different from Gradient Momentum**
Traditional momentum smooths **gradients** → helps with gradient variance
SlowMo smooths **parameters** → helps with optimization trajectory

---

## 📈 Expected Behavior

### Training Curve Comparison:

```
Loss
│
│  ╱╲╱╲╱╲         ← Local optimizer (noisy)
│ ╱      ╲╱╲
│          ╲╱
│
│   ___           ← SlowMo (smooth!)
│  ╱   ╲___
│ ╱        ╲___
│
└─────────────── Iterations
```

**SlowMo produces much smoother parameter trajectories!**

---

## 🆚 Comparison with Other Methods

### vs Standard Momentum:
| Aspect | Standard Momentum | SlowMo |
|--------|-------------------|--------|
| **Applied To** | Gradients | **Parameters** |
| **Updates** | Every step | **Every k steps** |
| **Purpose** | Accelerate convergence | **Stabilize trajectory** |
| **Memory** | Gradient buffer | **Parameter buffer** |

### vs Lookahead:
| Aspect | Lookahead | SlowMo |
|--------|-----------|--------|
| **Mechanism** | Interpolation | **Momentum** |
| **Formula** | θ_slow += α(θ_fast-θ_slow) | **v = β*v + Δθ; θ += η*v** |
| **Memory** | Slow weights | **Momentum buffer** |
| **Effect** | Pull-back | **Smooth acceleration** |

### Original Use Case:
```
Distributed Training (Original Paper):
  - Multiple workers train locally for k steps
  - Workers communicate and apply slow momentum
  - Reduces communication frequency
  - Stabilizes across heterogeneous workers

Single Worker (Our Adaptation):
  - Single optimizer trains for k steps
  - Apply slow momentum to own parameters
  - Still benefits from trajectory stabilization
  - Reduces variance in parameter updates
```

---

## 🔬 Advanced: SlowMo Variants

### 1. Adaptive k:
```python
# Start with small k, increase over time
k = min(5 + epoch // 10, 20)
slowmo = SlowMoHandler(model, beta=0.6, k=k, lr_slow=1.0)
```

### 2. Decaying β:
```python
# Start with high momentum, decay over time
beta = max(0.4, 0.8 - epoch * 0.01)
slowmo = SlowMoHandler(model, beta=beta, k=10, lr_slow=1.0)
```

### 3. SlowMo + Gradient Clipping:
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
slowmo.step()
```

---

## 📊 Paper Results (Distributed Training)

From the original paper on ImageNet:

| Method | Top-1 Accuracy | Communication Rounds |
|--------|----------------|---------------------|
| Local SGD | 75.8% | 100 |
| **SlowMo** | **76.2%** | **100** |
| Baseline (no distribution) | 76.3% | N/A |

**Key**: SlowMo nearly matches baseline while dramatically reducing communication!

---

## 🔍 Debugging SlowMo

### Issue: No improvement over baseline
**Cause**: Parameters not well-tuned
**Solution**:
```bash
# Try stronger momentum
--beta 0.7 --lr-slow 1.2

# Try more frequent updates
--k 5
```

### Issue: Training becomes unstable
**Cause**: lr_slow too large or β too small
**Solution**:
```bash
--beta 0.8 --lr-slow 0.5
```

### Issue: Convergence too slow
**Cause**: Too much smoothing
**Solution**:
```bash
--beta 0.4 --k 15 --lr-slow 1.5
```

---

## 🎯 When to Use SlowMo

### ✅ Use SlowMo When:
- Training is **noisy** or **unstable**
- Want **smoother convergence curves**
- Working with **small batch sizes** (high variance)
- Need **variance reduction**
- Training on **non-IID data** (like federated learning)

### ⚠️ Consider Alternatives When:
- Training is already very smooth → minimal benefit
- Need per-parameter adaptation → use Adam/RMSprop
- Want automatic LR tuning → use SPS
- Need sharpness-aware training → use ASAM

---

## 💎 SlowMo's Unique Feature

**Momentum on Parameters, Not Gradients**

This makes SlowMo complementary to gradient-based momentum:

```python
# Can combine both!
optimizer = SGD(params, lr=0.01, momentum=0.9)  # Gradient momentum
slowmo = SlowMoHandler(model, beta=0.6, k=10)   # Parameter momentum

# Both work together:
loss.backward()
optimizer.step()  # Applies gradient momentum
slowmo.step()     # Applies parameter momentum
```

**Double momentum for extra stability!**

---

## 📁 Output Structure

```
./results/
└── boston/
    └── mlp/
        └── Adam/
            ├── run_1_epoch_losses.csv    # generic_loss, slowmo_loss
            ├── run_2_epoch_losses.csv
            ├── run_3_epoch_losses.csv
            ├── run_4_epoch_losses.csv
            ├── run_5_epoch_losses.csv
            ├── Adam_summary.csv           # Generic, SlowMo columns
            └── Adam_report.txt            # SlowMo parameters shown
```

---

## 🎯 Quick Reference Card

### Conservative Setup:
```bash
python Deep_learning_SlowMo.py --optimizer Adam \
    --beta 0.8 --k 5 --lr-slow 0.5
```

### Balanced Setup (Default):
```bash
python Deep_learning_SlowMo.py --optimizer Adam \
    --beta 0.6 --k 10 --lr-slow 1.0
```

### Aggressive Setup:
```bash
python Deep_learning_SlowMo.py --optimizer Adam \
    --beta 0.4 --k 20 --lr-slow 1.5
```

### Parameter Effects:
- ↑ β → more smoothing, slower adaptation
- ↓ β → less smoothing, faster adaptation
- ↑ k → less frequent updates, more exploration
- ↓ k → more frequent updates, more stability
- ↑ η_slow → stronger momentum effect
- ↓ η_slow → gentler momentum effect

---

## 📚 Citation

```bibtex
@inproceedings{wang2020slowmo,
  title={SlowMo: Improving Communication-Efficient Distributed SGD with Slow Momentum},
  author={Wang, Jianyu and Liu, Qinghua and Liang, Hao and Joshi, Gauri and Poor, H. Vincent},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```

---

## ✅ Key Takeaways

1. **Momentum on parameters** (not gradients) - unique approach!
2. **Three parameters** (β, k, η_slow) - more tuning than Lookahead
3. **Originally for distributed** - but works great single-worker too!
4. **Trajectory stabilization** - creates smoother optimization path
5. **Complementary to gradient momentum** - can use both!
6. **Variance reduction** - especially good for noisy/small batch training
7. **Implicit regularization** - smooth updates prevent overfitting
8. **Works with any optimizer** - universal wrapper!

---

## 🌟 Why SlowMo is Special

**The Only Method That Applies Momentum to Parameters Themselves!**

```
Everyone else:           SlowMo:
Modify gradients    →    Modify parameter trajectory
Modify LR           →    Add parameter momentum
Interpolate weights →    Accumulate parameter changes
```

**Fundamentally different approach to stabilization!**

---

## 🚀 Ready to Use!

Start with defaults:
```bash
python Deep_learning_SlowMo.py --optimizer Adam --epochs 100 --runs 1
```

Scale up:
```bash
python Deep_learning_SlowMo.py --dataset boston --model mlp --optimizer All \
    --epochs 1000 --runs 5
```

**The beauty of SlowMo**: Momentum where you never thought to apply it - on the parameters! 🎯
