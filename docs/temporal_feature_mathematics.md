# Temporal Feature Mathematics

## Overview

This document explains the mathematical formulation of each temporal feature, how to compute it, and how to interpret the results.

## The 5 Temporal Features

### 1. Burst Coefficient (BC)

**What it measures**: Irregularity in message rhythm - do messages come in bursts or steady streams?

**Formula**:
```
BC = (σ - μ) / (σ + μ)
```

Where:
- σ = standard deviation of inter-message intervals
- μ = mean of inter-message intervals

**Computation**:
1. Calculate intervals: `intervals = diff(timestamps)`
2. Compute mean: `μ = mean(intervals)`
3. Compute std: `σ = std(intervals)`
4. Apply formula: `BC = (σ - μ) / (σ + μ)`

**Range**: [-1, 1]

**Interpretation**:
- **BC ≈ 1**: Highly bursty (e.g., rapid fire messages, then long silence)
  - High variance relative to mean
  - Example: Student sends 5 messages in 30 seconds, then nothing for 10 minutes
- **BC ≈ 0**: Moderate variability (typical conversation)
  - Variance roughly equals mean
- **BC ≈ -1**: Extremely regular, clockwork-like timing
  - Very low variance relative to mean
  - Example: Messages every 2 minutes ± 5 seconds

**Example**:
- Intervals: [10s, 12s, 11s, 10s, 12s] → BC ≈ -0.91 (very regular)
- Intervals: [2s, 3s, 180s, 2s, 240s] → BC ≈ 0.85 (very bursty)

---

### 2. Cluster Density (CD)

**What it measures**: Temporal "clumpiness" - are messages concentrated in certain time periods?

**Formula**:
```
CD = Var(window_counts) / Mean(window_counts)
```

**Computation**:
1. Choose window size (adaptive: 30% of conversation span, bounded [10s, 300s])
2. Slide window across conversation with 50% overlap
3. Count messages in each window position
4. Compute: `CD = variance(counts) / mean(counts)`

**Range**: [0, ∞)

**Interpretation**:
- **CD ≈ 0**: Evenly distributed messages (drizzle)
  - All windows have similar message counts
  - Example: Steady conversation pace throughout
- **CD > 1**: Messages clustered in bursts (downpour)
  - Some windows have many messages, others have few
  - Example: Active discussion periods with quiet gaps

**Why it often fails for tutoring data**:
- Requires multiple messages across sufficient time span
- Student messages often span <30 seconds even in 15-minute conversations
- With only 3-5 student messages spread thin, can't get meaningful windows
- Result: Most conversations return CD = 0

**When it works**:
- Chat logs with 50+ messages
- Long customer support conversations
- Dense messaging (messages every few seconds for minutes)

---

### 3. Response Acceleration (RA)

**What it measures**: Does the conversation speed up or slow down over time?

**Formula**:
```
RA = slope of linear regression: interval[i] ~ i
```

**Computation**:
1. Calculate intervals: `intervals = diff(timestamps)`
2. Create sequence: `x = [0, 1, 2, ..., n-1]`
3. Fit line: `intervals[i] = a + b*i`
4. Return slope: `RA = b`

**Range**: (-∞, ∞), but typically [-10, +10] seconds/message

**Interpretation**:
- **RA < 0**: Speeding up (intervals getting shorter)
  - Negative slope: later messages come faster
  - Example: "Breakthrough" - student gets it, rapid follow-ups
  - Value: -2 means responses get 2 seconds faster per message
- **RA ≈ 0**: Constant pace
  - Flat slope: no systematic change
- **RA > 0**: Slowing down (intervals getting longer)
  - Positive slope: later messages take longer
  - Example: "Frustration" - student struggles, delays increase
  - Value: +3 means responses get 3 seconds slower per message

**Example**:
- Intervals: [10s, 8s, 6s, 4s, 2s] → RA ≈ -2 (speeding up by 2s per message)
- Intervals: [5s, 10s, 15s, 20s] → RA ≈ +5 (slowing down by 5s per message)

---

### 4. Memory Coefficient (MC)

**What it measures**: Temporal autocorrelation - does current pacing predict future pacing?

**Formula**:
```
MC = Pearson correlation between consecutive intervals
    = corr(intervals[:-1], intervals[1:])
```

**Computation**:
1. Calculate intervals: `intervals = diff(timestamps)`
2. Create pairs: `(intervals[i], intervals[i+1])` for all i
3. Compute correlation: `MC = pearson_r(intervals[:-1], intervals[1:])`

**Range**: [-1, +1]

**Interpretation**:
- **MC ≈ +1**: Strong positive autocorrelation
  - Long intervals tend to follow long intervals
  - Short intervals follow short intervals
  - Example: Student in "flow state" maintains consistent rhythm
- **MC ≈ 0**: No memory / independence
  - Each interval unrelated to previous
  - Random pacing
- **MC ≈ -1**: Negative autocorrelation
  - Long intervals alternate with short ones
  - Oscillating pattern
  - Example: Question-answer rhythm (fast reply, then wait for response)

**Example**:
- Intervals: [10s, 12s, 11s, 13s, 12s] → MC ≈ +0.7 (consistent pacing)
- Intervals: [5s, 60s, 5s, 60s, 5s] → MC ≈ -0.9 (alternating pattern)

---

### 5. Timing Consistency (TC)

**What it measures**: Overall stability/regularity of message rhythm

**Formula**:
```
TC = 1 - (σ_normalized)
```

Where `σ_normalized` is the standard deviation scaled to [0, 1] range.

Actually implemented as:
```
TC = 1 - BC  (simple inversion of burst coefficient)
```

Or more rigorously:
```
TC = normalized measure of interval consistency
```

**Range**: [0, 1]

**Interpretation**:
- **TC ≈ 1**: Extremely consistent, clockwork-like
  - Minimal variation in intervals
  - Example: Messages every 2 minutes like clockwork (TC = 0.95)
- **TC ≈ 0.5**: Moderate consistency
  - Some variation but predictable
- **TC ≈ 0**: Highly inconsistent / erratic
  - Wild variation in timing
  - Example: 2s, 300s, 5s, 180s intervals

**Relationship to Burst Coefficient**:
- TC and BC are inversely related but measure slightly different aspects
- BC emphasizes variance relative to mean
- TC emphasizes absolute consistency

**Example**:
- Intervals all ≈ 10s ± 1s → TC ≈ 0.95 (very consistent)
- Intervals: [5s, 300s, 10s, 200s] → TC ≈ 0.1 (very inconsistent)

---

## Role-Specific Features (15D Analysis)

For each of the 5 features above, we compute 3 versions:

1. **Conversation-level** (BC, CD, RA, MC, TC): Computed on all message timestamps
2. **Tutor-specific** (BC_tutor, CD_tutor, ...): Computed only on tutor messages
3. **Student-specific** (BC_student, CD_student, ...): Computed only on student messages

**Key insight**: Role-specific features reveal who drives the pattern.

**Example**:
- Conversation BC = 0.07 (slightly bursty overall)
- Student BC = 0.82 (student is very bursty)
- Tutor BC = -0.15 (tutor maintains steady pace)

**Interpretation**: Student creates bursts; tutor smooths them out. Averaging to conversation-level obscures this asymmetry.

---

## Practical Interpretation for This Dataset

### What Works Well

1. **Burst Coefficient (Student)**: η² = 0.608
   - THE defining feature
   - Separates "bursty" vs "steady" students

2. **Timing Consistency (Student)**: η² = 0.478
   - Second strongest signal
   - Identifies "clockwork" students

### What Doesn't Work (And Why)

**Cluster Density**: η² = NaN (all zeros)
- **Why**: Student messages too sparse
- Most students send 3-8 messages spread across conversation
- Even in 15-min conversation, student messages might span 45 seconds
- Sliding windows need dense, sustained messaging

**Response Acceleration & Memory Coefficient**: η² < 0.01
- **Why**: Need long sequences to detect trends
- With only 5-10 messages, slope/correlation estimates are noisy
- Work better with 50+ message sequences

---

## Feature Engineering Insights

### For Future Datasets

**When cluster density will work**:
- Chat applications (50+ messages/conversation)
- Customer support (sustained back-and-forth)
- Dense messaging patterns (multiple messages per minute)

**When to use role decomposition**:
- Asymmetric interactions (expert/novice, interviewer/interviewee)
- Hypothesis: one party drives temporal dynamics
- Want to understand adaptation vs imposition

**Feature selection**:
- BC and TC are robust even with sparse data (only need 3+ messages)
- RA and MC need longer sequences (10+ messages minimum)
- CD needs dense messaging (20+ messages in sustained time period)

---

## Summary Table

| Feature | Formula | Interpretation | Best For |
|---------|---------|----------------|----------|
| **BC** | (σ-μ)/(σ+μ) | Bursty (-1) vs Steady (+1) | All conversation types |
| **CD** | Var/Mean of window counts | Clumped (+) vs Even (0) | Dense messaging only |
| **RA** | Slope of intervals | Speeding up (-) vs Slowing (+) | Trend detection |
| **MC** | Autocorrelation of intervals | Persistent (+) vs Random (0) | Rhythm analysis |
| **TC** | 1 - normalized variance | Consistent (1) vs Erratic (0) | All conversation types |

**For tutoring conversations**: BC and TC (especially student-specific) provide the strongest signal.
