# Multi-Output SGD Stabilization Guide (CUDA Online MLP)

## Overview

This document summarizes the final stabilization strategy for a
multi-output MLP trained with online SGD inside a CUDA kernel.

------------------------------------------------------------------------

## Architecture

    Input (15) → W1 (128) → W2 (128) → W3[o] (outputs)

Outputs: - comp_time - ratio - psnr - decomp_time (inactive)

------------------------------------------------------------------------

## Current State

### Achievements

-   Sequential inference (no duplicate predictions)
-   PCGrad-lite implemented
-   Uncertainty weighting working
-   Fast convergence achieved
-   Overshoot recovery improved dramatically

### Remaining Issues

-   Oscillation between outputs (even/odd timestep flip)
-   Occasional catastrophic spikes (e.g., ratio explosion)
-   Non-deterministic SGD ordering
-   Noise in comp_time destabilizing updates

------------------------------------------------------------------------

## Root Causes

### 1. Asynchronous SGD

Out-of-order updates from parallel workers create instability.

### 2. Underdamped Updates

Large step sizes cause overshooting and oscillations.

### 3. Noisy Targets

comp_time jitter introduces false gradients.

------------------------------------------------------------------------

## Final Solution Pipeline

### Step 1: Per-output gradients

-   Compute per-output gradients via backward pass

------------------------------------------------------------------------

### Step 2: Normalize gradients

    g[o] = g[o] / (||g[o]|| + 1e-6)

------------------------------------------------------------------------

### Step 3: Apply PCGrad (shared layers only)

    if dot(g_i, g_j) < -ε:
        g_i = g_i - proj_{g_j}(g_i)

------------------------------------------------------------------------

### Step 4: Combine gradients

    g_total = Σ g[o]

------------------------------------------------------------------------

### Step 5: Trust-region scaling

    g_total = g_total / ||g_total||
    step = min(max_step, k * avg_error)

------------------------------------------------------------------------

### Step 6: EMA smoothing

    g_ema = 0.85 * g_ema + 0.15 * g_total
    W -= step * g_ema

------------------------------------------------------------------------

### Step 7: Minimum update (avoid stall)

    step = max(step, min_step)

------------------------------------------------------------------------

### Step 8: Noise gating (comp_time)

    if |error_comp| < threshold:
        ignore comp_time gradient

------------------------------------------------------------------------

### Step 9: Anti-flip damping

    if dot(g_total, g_prev) < 0:
        g_total *= 0.5

------------------------------------------------------------------------

## Critical Fix: Ordered SGD

Ensure updates are applied in-order:

    for chunk in sequence:
        forward
        backward
        update

Alternative: - gradient buffer + single update thread

------------------------------------------------------------------------

## Recommended Hyperparameters

  Parameter         Value
  ----------------- --------------
  ema_decay         0.85--0.9
  k (step scale)    0.05--0.1
  max_step          0.01--0.03
  min_step          1e-4
  PCGrad cos        -0.2 to -0.3
  noise threshold   10--15%

------------------------------------------------------------------------

## Expected Outcome

### Before

-   Oscillation
-   Overshoot spikes
-   Non-reproducible runs

### After

-   Stable convergence
-   No catastrophic spikes
-   Smooth multi-output learning
-   Consistent performance

------------------------------------------------------------------------

## Key Insight

This is no longer a machine learning problem --- it is a control systems
stability problem.

Stability requires: - Ordered updates - Damping - Bounded steps - Noise
filtering