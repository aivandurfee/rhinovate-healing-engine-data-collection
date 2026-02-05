# The Missing Middle: Training Without Intermediate Healing Data

## Problem Statement

Rhinovate has strong **Day 0 (Before)** and **Day 180+ (After)** pairs from ASPS and sorted Reddit data, but very few real **intermediate** photos (Day 7, Day 30, Day 90). A generative model (Deformation U-Net, GAN, or diffusion) that predicts swelling at arbitrary timepoints needs supervision for those in-between states.

---

## Your Proposal: Linear Interpolation of Flow Field

**Idea:** Compute an optical flow (or deformation field) from Day 0 → Day 180, then linearly interpolate the flow to generate synthetic targets:
- Day 7 ≈ 90% swelling → flow weight ~0.04 (≈ 7/180)
- Day 90 ≈ 50% swelling → flow weight ~0.5

### Critique

| Aspect | Assessment |
|--------|------------|
| **Geometric plausibility** | ✅ Interpolating a *smooth* flow is reasonable: swelling reduction is largely monotonic and spatially continuous. |
| **Biological accuracy** | ⚠️ Swelling is **non-linear** in time. Most reduction happens in Weeks 1–4; Months 4–6 are subtle refinement. Linear interpolation assumes constant rate. |
| **Training signal** | ✅ Provides supervision where none exists. Better than no intermediate data. |
| **Risk of mode collapse** | ⚠️ If the model overfits to interpolated intermediates, it may learn "average" trajectories and miss real variability (e.g., asymmetric resolution, individual differences). |

### Verdict

**Use it as a *weak* supervisory signal, not ground truth.** Treat interpolated intermediates as:
- **Data augmentation** for the deformation model
- **Soft targets** (e.g., lower loss weight than real Before/After pairs)
- **Pre-training** only; fine-tune on any real Healing data you collect later

---

## Alternative Strategies (Ranked by Feasibility)

### 1. **Flow Interpolation + Temporal Prior (Recommended)**

- Compute flow \( F_{0 \to 180} \) from Before → After (e.g., with RAFT, or learned in-network).
- Generate synthetic targets: \( I_t = warp(I_0, \alpha_t \cdot F) \) where \( \alpha_t = f(t) \) is a **non-linear** swelling curve.
- Use a clinically informed curve, e.g.:
  - \( f(t) = 1 - (1 - e^{-t/\tau}) \) or
  - Piecewise: steep drop 0–30 days, gradual 30–180 days.

**Pros:** More realistic temporal dynamics. **Cons:** Requires choosing/learning \( f(t) \).

---

### 2. **Cycle Consistency + Few Real Midpoints**

- Train with Before/After pairs and a handful of real Day 7 / Day 30 photos.
- Add cycle loss: \( I_{90} = G(I_0, 90) \) should deform back toward \( I_0 \) when "reversed" (conceptually).
- Use the few real intermediates as hard anchors; let the model interpolate elsewhere.

**Pros:** Real data anchors the trajectory. **Cons:** Needs at least 50–100 real intermediates for stability.

---

### 3. **Diffusion / Latent Interpolation**

- Encode Before and After in a latent space (e.g., VAE, Stable Diffusion).
- Train a model to predict latent at time \( t \) via linear interpolation in latent space: \( z_t = (1-\alpha) z_0 + \alpha z_{180} \).
- Decode \( z_t \) to get synthetic \( I_t \).

**Pros:** No explicit flow; leverages learned face/nose manifold. **Cons:** May introduce artifacts; less control over anatomy.

---

### 4. **Curriculum Learning**

- **Stage 1:** Train only on Before/After pairs (strong supervision).
- **Stage 2:** Add interpolated intermediates with reduced loss weight.
- **Stage 3:** If available, fine-tune on real Healing data.

**Pros:** Simple, robust. Model learns geometry first, then temporal detail. **Cons:** Two-phase training.

---

### 5. **Semi-Supervised / Pseudo-Labels**

- Train an initial model on Before/After only.
- Run inference on unlabeled Healing posts (e.g., "3 weeks post-op") to generate pseudo-labels.
- Retrain with pseudo-labels as soft targets, with confidence weighting.

**Pros:** Uses Reddit Healing data you might have. **Cons:** Noisy; needs careful filtering.

---

## Practical Recommendation for V1

1. **Implement flow interpolation** with a **non-linear** \( \alpha(t) \) curve (Strategy 1).
2. **Weight losses:** Real Before/After pairs = 1.0; interpolated intermediates = 0.2–0.3.
3. **Use curriculum:** First train on pairs only; then add interpolated data.
4. **Log all Reddit Healing images** with timeframe metadata—even if sparse, they become precious fine-tuning data for V2.
5. **Validate** on any held-out real intermediates; report metrics separately for "real" vs "synthetic" test sets.

---

## Flow Implementation Sketch

```python
# Pseudocode for flow interpolation
def get_alpha(day: int) -> float:
    """Swelling retention: 1.0 = full swelling, 0.0 = fully healed."""
    # Non-linear: most drop in first 30 days
    tau = 30
    return np.exp(-day / tau) * (1 + 0.2 * np.exp(-day / 90))

# Training sample
flow_0_180 = compute_flow(before_img, after_img)  # e.g., RAFT
alpha = get_alpha(target_day)
synthetic_flow = alpha * flow_0_180
target_img = warp(before_img, synthetic_flow)
```

This gives you synthetic Day 7 (high α) and Day 90 (low α) with more realistic progression than linear \( \alpha = day/180 \).
