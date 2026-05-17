# HMS (Harmonic Mean Smoothing) Implementation Guide

## Overview

**HMS (Harmonic Mean Smoothing)** is a weight smoothing technique that uses the harmonic mean to create robust, outlier-resistant weight updates with asymmetric behavior that resists weight decay while encouraging weight growth.

---

## 🔬 What is HMS?

### Core Idea:
HMS applies the **harmonic mean** to smooth weight transitions, providing:
1. **Outlier resistance** - harmonic mean less affected by extreme values
2. **Asymmetric updates** - different behavior for increasing vs decreasing weights
3. **Adaptive smoothing** - strength decays over time

### Mathematical Foundation:

**Harmonic Mean**:
```
HM(a, b) = 2ab / (a + b)
```

Properties:
- Always ≤ min(a, b)
- Heavily influenced by smaller value
- Robust to outliers (unlike arithmetic mean)

**HMS Formula**:
```
For each weight w:
1. HM = 2*|w_prev|*|w_curr| / (|w_prev| + |w_curr|)
2. HMS_scalar = |HM - min(|w_prev|, |w_curr|)| * r
3. If w_prev > w_curr:  w_new = w_curr - HMS_scalar  (slow down decrease)
   If w_prev < w_curr:  w_new = w_curr + HMS_scalar  (accelerate increase)
```

---

## 🎯 The Algorithm (Detailed)

### Step-by-Step Process:

```python
For each parameter tensor in the model:
    
    Step 1: Get current and previous weights
        w_prev = weights from last iteration
        w_curr = weights after optimizer.step()
    
    Step 2: Filter out zeros (harmonic mean undefined at 0)
        mask = (w_prev != 0) & (w_curr != 0)
        
    Step 3: Compute harmonic mean
        HM = 2 * |w_prev| * |w_curr| / (|w_prev| + |w_curr|)
        
    Step 4: Compute HMS correction scalar
        min_abs = min(|w_prev|, |w_curr|)
        HMS = |HM - min_abs| * r
        
    Step 5: Apply asymmetric correction
        If w_prev > w_curr (decreasing):
            w_new = w_curr - HMS  # Resist decrease
        If w_prev < w_curr (increasing):
            w_new = w_new + HMS   # Accelerate increase
            
    Step 6: Update weights
        Replace w_curr with w_new
        
    Step 7: Cache for next iteration
        w_prev = w_new

Every t iterations:
    Step 8: Decay smoothing strength
        r = r * decay_rate
```

---

## 📊 Comparison: SlowMo vs HMS

| Aspect | SlowMo | HMS |
|--------|--------|-----|
| **Core Mechanism** | Momentum on parameters | **Harmonic mean smoothing** |
| **What It Tracks** | Momentum buffer | **Previous weights** |
| **Update Type** | v = β*v + Δθ; θ += η*v | **Asymmetric HM correction** |
| **Robustness** | Variance reduction | **Outlier resistance** |
| **Symmetry** | Symmetric | **Asymmetric (slow decay, fast growth)** |
| **Memory** | Momentum buffer | **Previous weight copy** |
| **Decay** | No decay | **Adaptive r decay** |
| **Parameters** | β, k, η_slow | **r, t, decay_rate** |
| **Best For** | Distributed training | **Outlier resistance** |

---

## 🎛️ Parameters Explained

### 1. **r** - Smoothing Strength
```python
Default: 1.0
Range: 0.5 - 2.0
```

**What it does**: Controls magnitude of HMS correction

**Intuition**:
- **r = 0.5**: Weak smoothing (50% of calculated HMS applied)
- **r = 1.0**: Full smoothing (100% of calculated HMS applied)
- **r = 2.0**: Strong smoothing (200% of calculated HMS applied)

**Examples**:
```
If HMS_scalar = 0.01:
  r = 0.5 → correction = 0.005
  r = 1.0 → correction = 0.01
  r = 2.0 → correction = 0.02
```

**When to tune**:
- Noisy training → **increase to 1.5-2.0** (more smoothing)
- Smooth training → **decrease to 0.5-0.7** (less smoothing)
- Default 1.0 works well for most cases

---

### 2. **t** - Decay Interval
```python
Default: 1000
Range: 500 - 2000
```

**What it does**: Iterations between r decay steps

**Intuition**:
- **t = 500**: Frequent decay (r weakens quickly)
- **t = 1000**: Moderate decay (balanced)
- **t = 2000**: Rare decay (r stays strong longer)

**Timeline Example (decay_rate=0.9)**:
```
t = 500:
  Iteration 0:    r = 1.0
  Iteration 500:  r = 0.9
  Iteration 1000: r = 0.81
  Iteration 1500: r = 0.729

t = 1000:
  Iteration 0:    r = 1.0
  Iteration 1000: r = 0.9
  Iteration 2000: r = 0.81
  Iteration 3000: r = 0.729
```

**When to tune**:
- Short training (few epochs) → **t = 500-700**
- Long training (many epochs) → **t = 1500-2000**
- Want strong early smoothing → **t = 1000-1500**

---

### 3. **decay_rate** - R Decay Rate
```python
Default: 0.9
Range: 0.85 - 0.95
```

**What it does**: How much to multiply r by at each decay

**Formula**: `r_new = r_old * decay_rate`

**Intuition**:
- **decay_rate = 0.85**: Fast decay (15% reduction per step)
- **decay_rate = 0.9**: Moderate decay (10% reduction per step)
- **decay_rate = 0.95**: Slow decay (5% reduction per step)

**r Evolution (starting r=1.0, t=1000)**:
```
decay_rate = 0.85:
  0 iters:    r = 1.000
  1000 iters: r = 0.850
  2000 iters: r = 0.723
  3000 iters: r = 0.614

decay_rate = 0.9:
  0 iters:    r = 1.000
  1000 iters: r = 0.900
  2000 iters: r = 0.810
  3000 iters: r = 0.729

decay_rate = 0.95:
  0 iters:    r = 1.000
  1000 iters: r = 0.950
  2000 iters: r = 0.903
  3000 iters: r = 0.857
```

**When to tune**:
- Want smoothing to fade quickly → **0.85-0.87**
- Want smoothing to persist → **0.93-0.95**
- Balanced → **0.9** (default)

---

## 🚀 Usage Examples

### Quick Test:
```bash
python Deep_learning_HMS.py --optimizer Adam --epochs 10 --runs 1 \
    --output ./results
```

### Standard Run:
```bash
python Deep_learning_HMS.py --dataset boston --model mlp --optimizer Adam \
    --epochs 1000 --runs 5 --output ./results
```

### Custom HMS Parameters:

#### Strong Smoothing (Maximum Outlier Resistance):
```bash
python Deep_learning_HMS.py --optimizer Adam \
    --r 2.0 --t 500 --decay-rate 0.95 \
    --output ./results
```
**Use when**: Very noisy gradients, outliers common

#### Balanced (Default):
```bash
python Deep_learning_HMS.py --optimizer Adam \
    --r 1.0 --t 1000 --decay-rate 0.9 \
    --output ./results
```
**Use when**: General purpose

#### Weak Smoothing (Minimal Intervention):
```bash
python Deep_learning_HMS.py --optimizer Adam \
    --r 0.5 --t 2000 --decay-rate 0.85 \
    --output ./results
```
**Use when**: Clean data, want minimal modification

---

## 🎯 Parameter Tuning Guide

### Dataset-Based Tuning:

#### Small Noisy Datasets (< 1000 samples):
```bash
--r 1.5 --t 700 --decay-rate 0.92
```
**Rationale**: High noise → need strong persistent smoothing

#### Medium Datasets (1000-10000 samples):
```bash
--r 1.0 --t 1000 --decay-rate 0.9
```
**Rationale**: Balanced (default works well)

#### Large Clean Datasets (> 10000 samples):
```bash
--r 0.7 --t 1500 --decay-rate 0.88
```
**Rationale**: Low noise → less smoothing needed

---

### Problem-Based Tuning:

#### Training with Outliers / Spikes:
```bash
--r 1.8 --t 800 --decay-rate 0.93
```
Strong smoothing to resist outliers

#### Training Too Slow:
```bash
--r 0.6 --t 1500 --decay-rate 0.85
```
Reduce smoothing to allow faster progress

#### Weights Vanishing (Going to Zero):
```bash
--r 1.2 --t 1000 --decay-rate 0.9
```
HMS resists decrease, helps maintain weights

#### Weights Exploding:
```bash
--r 0.5 --t 2000 --decay-rate 0.85
```
Reduce HMS effect to allow normal regularization

---

### Base Optimizer Recommendations:

#### With SGD:
```bash
--r 1.2 --t 1000 --decay-rate 0.9
```
SGD has high variance → benefit from smoothing

#### With Adam:
```bash
--r 0.8 --t 1200 --decay-rate 0.9
```
Adam already smooth → less HMS needed

#### With RMSprop:
```bash
--r 0.9 --t 1100 --decay-rate 0.9
```
Similar to Adam

#### With Momentum SGD:
```bash
--r 1.0 --t 1000 --decay-rate 0.9
```
Standard settings

---

## 💡 How HMS Works (Detailed Example)

### Example: r=1.0, t=1000, decay_rate=0.9

```
Initialization (iteration 0):
  w_0 = 0.5
  r = 1.0

Iteration 1:
  Optimizer updates: w_0 → w_1 = 0.52 (increased)
  
  HM = 2*0.5*0.52 / (0.5+0.52) = 0.5098
  min_abs = min(0.5, 0.52) = 0.5
  HMS = |0.5098 - 0.5| * 1.0 = 0.0098
  
  Since w increased: w_1 = 0.52 + 0.0098 = 0.5298 ← Accelerated!
  
  Cache: w_prev = 0.5298

Iteration 2:
  Optimizer updates: w_1 → w_2 = 0.51 (decreased)
  
  HM = 2*0.5298*0.51 / (0.5298+0.51) = 0.5198
  min_abs = min(0.5298, 0.51) = 0.51
  HMS = |0.5198 - 0.51| * 1.0 = 0.0098
  
  Since w decreased: w_2 = 0.51 - 0.0098 = 0.5002 ← Resisted!
  
  Cache: w_prev = 0.5002

...

Iteration 1000:
  r decays: r = 1.0 * 0.9 = 0.9
  
  Now HMS corrections are 10% weaker

Iteration 2000:
  r decays: r = 0.9 * 0.9 = 0.81
  
  HMS corrections are 19% weaker than initial
```

**Pattern**: 
- Increases are amplified (encourages weight growth)
- Decreases are damped (resists weight decay)
- Effect gradually weakens over time

---

## 🎓 Theory: Why HMS Works

### 1. **Harmonic Mean Properties**
```
For positive a, b:
  HM(a,b) ≤ GM(a,b) ≤ AM(a,b)
  
Where:
  HM = 2ab/(a+b)      (Harmonic Mean)
  GM = √(ab)          (Geometric Mean)
  AM = (a+b)/2        (Arithmetic Mean)
```

**Key**: HM heavily influenced by smaller value → robust to outliers!

**Example**:
```
a = 1, b = 100:
  HM = 2*1*100/(1+100) = 1.98  ← Close to smaller value
  AM = (1+100)/2 = 50.5         ← Affected by outlier

a = 10, b = 12:
  HM = 2*10*12/(10+12) = 10.91 ← Balanced
  AM = (10+12)/2 = 11           ← Similar
```

---

### 2. **Asymmetric Updates**
Traditional smoothing treats increases and decreases the same.

HMS is asymmetric:
```
Decreasing weights: w_new = w - HMS  (resist)
Increasing weights: w_new = w + HMS  (accelerate)
```

**Benefit**: 
- Fights weight decay (common problem in deep learning)
- Encourages feature learning (weight growth)
- Still allows convergence (r decays)

---

### 3. **Adaptive Decay**
Early training: Strong smoothing (high r) → resist noisy updates
Late training: Weak smoothing (low r) → allow fine-tuning

This matches training dynamics:
- Early: Exploration, high variance → need smoothing
- Late: Convergence, low variance → less smoothing needed

---

## 📈 Expected Behavior

### Training Curve:

```
Loss
│
│  ╱╲╱╲  ╱╲         ← Baseline (noisy, with spikes)
│ ╱    ╲╱  ╲╱╲
│            ╲
│
│   ___             ← HMS (smooth, outlier-resistant)
│  ╱   ╲___
│ ╱        ╲___
│
└──────────────── Iterations
```

**HMS characteristics**:
- Smoother curves (outlier-resistant)
- Fewer spikes (harmonic mean effect)
- Slightly slower initial descent (resists rapid change)
- Better final convergence (adaptive decay)

---

## 🆚 Comparison with Other Methods

### vs Gradient Clipping:
| Aspect | Gradient Clipping | HMS |
|--------|------------------|-----|
| **Applied To** | Gradients | **Weights** |
| **Mechanism** | Hard threshold | **Soft smoothing** |
| **Asymmetry** | No | **Yes (slow decay, fast growth)** |
| **Adaptation** | Fixed threshold | **Decaying strength** |

### vs Weight Decay:
| Aspect | Weight Decay | HMS |
|--------|--------------|-----|
| **Purpose** | Regularization | **Smoothing** |
| **Effect** | Shrink weights | **Smooth transitions** |
| **Direction** | Always toward zero | **Directional (asymmetric)** |

### vs Lookahead:
| Aspect | Lookahead | HMS |
|--------|-----------|-----|
| **Mechanism** | Weight interpolation | **Harmonic smoothing** |
| **Frequency** | Every k steps | **Every iteration** |
| **Computation** | Store slow weights | **Store prev weights** |
| **Effect** | Pull-back | **Outlier resistance** |

---

## 🔬 Advanced: HMS Variants

### 1. HMS with Warmup:
```python
# Don't apply HMS in early iterations
if iteration > warmup_iterations:
    hms.step()
```

### 2. Adaptive r Based on Loss:
```python
# Increase r when loss spikes
if loss > prev_loss * 1.5:
    hms.r = min(hms.r * 1.2, 2.0)
```

### 3. Layer-Specific r:
```python
# Different smoothing for different layers
r_values = {
    'layer1': 1.5,  # Strong smoothing
    'layer2': 1.0,  # Medium
    'layer3': 0.5   # Weak
}
```

### 4. HMS + Gradient Clipping:
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
hms.step()  # Double protection!
```

---

## 📊 Mathematical Properties

### Harmonic Mean Facts:

1. **Undefined at zero**: 2*0*b/(0+b) = 0 (HMS handles this)

2. **Always ≤ minimum**:
   ```
   HM(a,b) ≤ min(a,b)
   ```

3. **Closer to smaller value**:
   ```
   If a << b: HM ≈ 2a
   If a = b:  HM = a = b
   ```

4. **Outlier resistance**:
   ```
   HM(1, 1000) = 1.998  ← Barely affected
   AM(1, 1000) = 500.5  ← Heavily affected
   ```

---

## 🔍 Debugging HMS

### Issue: Weights shrinking to zero
**Cause**: HMS resists decrease, but if optimizer consistently pushes down, it helps convergence
**Solution**: 
```bash
--r 1.5 --t 800  # Stronger resistance
```

### Issue: Weights exploding
**Cause**: HMS accelerates increases, might amplify growth
**Solution**:
```bash
--r 0.5 --decay-rate 0.85  # Weaker effect, faster decay
```

### Issue: No improvement over baseline
**Cause**: Clean data with no outliers → HMS not needed
**Solution**: Expected! HMS excels with noisy/outlier data

### Issue: Training slower than baseline
**Cause**: HMS adds computation per iteration
**Solution**: Normal! Overhead is ~5-10% but benefits can outweigh

---

## 🎯 When to Use HMS

### ✅ Use HMS When:
- Training has **outlier gradients** or weight updates
- Dealing with **noisy data**
- Weights are **decaying too fast** (vanishing)
- Want **robust, stable training**
- Using **small batch sizes** (high variance)

### ⚠️ Consider Alternatives When:
- Data is clean with low noise → minimal benefit
- Need maximum speed → HMS adds overhead
- Want momentum-based methods → use SlowMo
- Need sharpness-aware training → use ASAM

---

## 💎 HMS's Unique Features

### 1. **Asymmetric Updates**
Only method that explicitly:
- Resists weight decreases
- Accelerates weight increases

### 2. **Outlier Resistance**
Harmonic mean naturally robust to outliers

### 3. **Self-Adapting**
Automatic decay → strong early, weak late

### 4. **Per-Weight Smoothing**
Applied to each weight individually, not global

---

## 📁 Output Structure

```
./results/
└── boston/
    └── mlp/
        └── Adam/
            ├── run_1_epoch_losses.csv    # generic_loss, hms_loss
            ├── run_2_epoch_losses.csv
            ├── run_3_epoch_losses.csv
            ├── run_4_epoch_losses.csv
            ├── run_5_epoch_losses.csv
            ├── Adam_summary.csv           # Generic, HMS columns
            └── Adam_report.txt            # HMS parameters shown
```

---

## 🎯 Quick Reference Card

### Conservative (Maximum Smoothing):
```bash
python Deep_learning_HMS.py --optimizer Adam \
    --r 2.0 --t 500 --decay-rate 0.95
```

### Balanced (Default):
```bash
python Deep_learning_HMS.py --optimizer Adam \
    --r 1.0 --t 1000 --decay-rate 0.9
```

### Aggressive (Minimal Smoothing):
```bash
python Deep_learning_HMS.py --optimizer Adam \
    --r 0.5 --t 2000 --decay-rate 0.85
```

### Parameter Effects:
- ↑ r → stronger smoothing
- ↓ r → weaker smoothing
- ↑ t → slower decay (smoothing lasts longer)
- ↓ t → faster decay (smoothing fades sooner)
- ↑ decay_rate → slower r reduction
- ↓ decay_rate → faster r reduction

---

## 📚 Mathematical Reference

### Harmonic Mean:
```
HM(a, b) = 2ab / (a + b)
```

### HMS Scalar:
```
HMS = |HM - min(|w_prev|, |w_curr|)| * r
```

### Asymmetric Update:
```
w_new = {
    w_curr - HMS,  if w_prev > w_curr  (decreasing)
    w_curr + HMS,  if w_prev < w_curr  (increasing)
    w_curr,        if w_prev = w_curr  (unchanged)
}
```

### R Decay:
```
r_t = r_0 * (decay_rate)^(t/interval)

Example: r_0=1.0, decay_rate=0.9, interval=1000
  r_1000 = 1.0 * 0.9^1 = 0.9
  r_2000 = 1.0 * 0.9^2 = 0.81
  r_3000 = 1.0 * 0.9^3 = 0.729
```

---

## ✅ Key Takeaways

1. **Harmonic mean** provides outlier resistance
2. **Asymmetric updates** - slows decreases, accelerates increases
3. **Adaptive decay** - strong early, weak late
4. **Per-iteration** - applied after every optimizer.step()
5. **Three parameters** - r (strength), t (interval), decay_rate
6. **Best for noisy data** - excels with outliers and variance
7. **Fights weight decay** - helps maintain feature weights
8. **Works with any optimizer** - universal wrapper

---

## 🌟 Why HMS is Unique

**Only method using harmonic mean for weight smoothing!**

Everyone else:
- Uses arithmetic operations
- Symmetric updates
- No explicit outlier resistance

**HMS**:
- Uses harmonic mean (mathematically robust)
- Asymmetric (directional preference)
- Explicitly designed for outlier resistance

---

## 🚀 Ready to Use!

Start with defaults:
```bash
python Deep_learning_HMS.py --optimizer Adam --epochs 100 --runs 1
```

Scale up:
```bash
python Deep_learning_HMS.py --dataset boston --model mlp --optimizer All \
    --epochs 1000 --runs 5
```

**The beauty of HMS**: Mathematical robustness meets practical asymmetry! 🎯
