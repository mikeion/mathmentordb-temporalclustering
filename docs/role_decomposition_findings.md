# Role Decomposition Analysis: Students Drive Temporal Patterns in Tutoring

## Key Finding

**Students drive temporal patterns in tutoring conversations, not tutors.**

When we separate temporal features by role (student vs tutor), we discover:
- **Student message timing** explains **60.8%** of variance between conversation types (η² = 0.608)
- **Tutor message timing** explains **<1%** of variance (η² < 0.01)
- **Traditional conversation-level analysis** obscures this by averaging the two roles together

**What this means**: The rhythm of tutoring conversations reflects student engagement strategies, not tutor behavior. Effective tutors adapt to student pacing rather than imposing their own rhythm.

## Understanding Temporal Features

Before diving into results, here's what we measured. Each feature captures a different aspect of message timing patterns.

### The 5 Temporal Features

**1. Burst Coefficient (BC)** - *Message rhythm regularity*
- **What it measures**: Do messages come in rapid bursts or steady streams?
- **Range**: -1 (clockwork regularity) to +1 (very bursty)
- **Interpretation**:
  - BC = 0.8: Rapid bursts followed by long silences (e.g., 5 messages in 30 seconds, then 10-minute gap)
  - BC = 0: Moderate variability (typical conversation)
  - BC = -0.9: Extremely regular timing (e.g., one message every 2 minutes like clockwork)
- **This is our strongest signal** (η² = 0.608 for students)

**2. Timing Consistency (TC)** - *Overall rhythm stability*
- **What it measures**: How stable is the message rhythm overall?
- **Range**: 0 (erratic) to 1 (perfectly consistent)
- **Interpretation**:
  - TC = 0.95: Nearly perfect regularity (e.g., messages at 10s ± 1s intervals)
  - TC = 0.5: Moderate consistency with some variation
  - TC = 0.1: Highly erratic timing (e.g., intervals of 2s, 300s, 5s, 180s)
- **This is our second strongest signal** (η² = 0.478 for students)

**3. Response Acceleration (RA)** - *Pacing changes over time*
- **What it measures**: Does the conversation speed up or slow down?
- **Range**: Negative (speeding up) to Positive (slowing down), measured in seconds/message
- **Interpretation**:
  - RA = -2: Responses get 2 seconds faster with each message (e.g., breakthrough moment)
  - RA = 0: Constant pace throughout
  - RA = +3: Responses get 3 seconds slower with each message (e.g., increasing difficulty)
- **Weak signal in this dataset** (η² = 0.003 for students)

**4. Memory Coefficient (MC)** - *Temporal autocorrelation*
- **What it measures**: Does current pacing predict future pacing?
- **Range**: -1 to +1 (correlation coefficient)
- **Interpretation**:
  - MC = +0.8: Long gaps followed by long gaps; short gaps followed by short gaps (sustained rhythm)
  - MC = 0: Each interval independent of the previous (random pacing)
  - MC = -0.8: Alternating pattern - long gaps alternate with short ones
- **Weak signal in this dataset** (η² = 0.002 for students)

**5. Cluster Density (CD)** - *Temporal clumpiness*
- **What it measures**: Are messages concentrated in certain time periods?
- **Range**: 0 (evenly distributed) to ∞ (highly clumped)
- **Interpretation**:
  - CD = 0: Messages evenly spread across conversation (steady drizzle)
  - CD > 1: Messages clustered in bursts with quiet periods (scattered downpours)
- **Does not work for this dataset** - student messages too sparse for sliding window analysis

**For full mathematical formulas**: See [Temporal Feature Mathematics](temporal_feature_mathematics.md)

---

## The Analysis

### What We Did

We analyzed 6,964 two-participant tutoring conversations from 2023 using two different approaches:

**Approach 1: Traditional Conversation-Level Analysis (5D)**
- Compute each temporal feature across the entire conversation (all messages from both student and tutor)
- Results in 5 features per conversation: BC, TC, RA, MC, CD
- This is the standard approach in conversation analysis

**Approach 2: Role-Specific Decomposition (15D)**
- Compute each temporal feature separately for:
  1. **Student messages only** → 5 features (BC_student, TC_student, RA_student, MC_student, CD_student)
  2. **Tutor messages only** → 5 features (BC_tutor, TC_tutor, RA_tutor, MC_tutor, CD_tutor)
  3. **All messages together** → 5 features (BC, TC, RA, MC, CD)
- Results in 15 features per conversation
- Allows us to see who drives the temporal patterns

**The question**: Does separating by role reveal something the conversation-level analysis misses?

### What We Found

**The answer is yes** - and the result is striking.

#### Effect Sizes: Who Drives Temporal Patterns?

We measured how well each feature distinguishes between conversation clusters using effect size (η²), which represents the proportion of variance explained. Higher values mean the feature is more important for distinguishing conversation types.

| Feature Category | Effect Size (η²) | Interpretation |
|-----------------|------------------|----------------|
| **Student features** | **0.478 - 0.608** | **Strong** - students drive patterns |
| Conversation features | 0.016 - 0.070 | **Weak** - signal washed out by averaging |
| Tutor features | 0.001 - 0.009 | **Negligible** - tutors don't drive patterns |

**Translation**: Student message timing patterns explain 60.8% of the differences between conversation types. Tutor patterns explain less than 1%. When you average them together (conversation-level), you lose 87% of the signal.

#### Which Features Matter Most?

**Student Features** - The strong discriminators:
- `burst_coefficient_student`: **η² = 0.608** ← THE defining feature
- `timing_consistency_student`: **η² = 0.478** ← Second strongest
- `response_acceleration_student`: η² = 0.003 (weak)
- `memory_coefficient_student`: η² = 0.002 (weak)
- `cluster_density_student`: Not computable (student messages too sparse)*

**Conversation Features** - Weak after averaging:
- `burst_coefficient`: η² = 0.070 (only 11% of student signal strength)
- `timing_consistency`: η² = 0.070
- `response_acceleration`: η² = 0.022
- `memory_coefficient`: η² = 0.016

**Tutor Features** - Essentially noise:
- `burst_coefficient_tutor`: η² = 0.009
- `timing_consistency_tutor`: η² = 0.002
- `memory_coefficient_tutor`: η² = 0.001
- `response_acceleration_tutor`: η² = 0.001

**Bottom line**: If you want to classify conversation types, measure the student's burst coefficient and timing consistency. Tutor behavior won't help.

*Note: Cluster density requires dense, sustained messaging for sliding window analysis. Students typically send only 3-8 messages spread across the conversation, making this metric non-computable for 99.6% of conversations.

## What This Means

### 1. We Found Two Student Engagement Archetypes (Not Conversation Types)

The clustering discovered two distinct ways students engage temporally. This is NOT about conversation types - it's about student behavior.

**Cluster 0 (93% of conversations): "Natural Conversational Flow"**
- Student burst coefficient: 0.077 (slightly bursty)
- Student timing consistency: 0.439 (moderate regularity)
- **What it looks like**: Student thinks, asks question, waits for response, thinks again. Natural ebb and flow with bursts of activity followed by processing time.

**Cluster 1 (7% of conversations): "Methodical Worker"**
- Student burst coefficient: -0.942 (extremely steady, anti-bursty)
- Student timing consistency: 0.968 (nearly perfect regularity)
- **What it looks like**: Student working through a problem set methodically, sending a message every 2-3 minutes like clockwork. Sustained, focused engagement.

**Key point**: These are student engagement strategies, not tutor teaching styles.

### 2. Tutors Adapt, They Don't Drive

The negligible effect sizes for tutor features (η² < 0.01) reveal that tutors don't impose a temporal structure. Instead:

- **Tutors match student pacing** - whether the student is bursty or steady, tutors adapt
- **No tutor "signature" exists** - you can't identify conversation types by tutor timing
- **Flexibility appears to be a feature of good tutoring** - responsive rather than rigid

This aligns with pedagogical best practices: effective tutors follow the student's lead rather than imposing their own rhythm.

### 3. Conversation-Level Analysis Destroys 87% of the Signal

When you average student and tutor timing together (traditional approach), the student signal gets washed out:

| Analysis Approach | Burst Coefficient η² | What You Learn |
|------------------|---------------------|----------------|
| **Student-only** | **0.608** | 60.8% of variance explained - strong signal |
| **Conversation-level** | 0.070 | 7% of variance explained - weak signal |
| **Signal loss** | **87%** | You lose almost all information |

**Why this happens**: Tutors maintain relatively constant pacing across all conversations (they adapt). Students vary dramatically. When you average constant + variable, you get moderate - which obscures the real driver.

**Analogy**: It's like averaging the temperature near a space heater with the temperature far from it. You get "moderately warm" (weak signal) instead of discovering where the heat source actually is (strong signal).

## Visualizations

All figures available in: [`figures/2023_conversations_two-participants/role_15d/`](../figures/2023_conversations_two-participants/role_15d/)

### Figure 1: Effect Size Comparison

![Effect Size Comparison](../figures/2023_conversations_two-participants/role_15d/figure1_effect_size_comparison.png)

**What this shows**: How well each feature distinguishes between the two conversation clusters. Higher bars = more important features.

**What to look for**:
- **Green bars (Student features)**: Two tower above everything
  - Burst Coefficient Student: η² = 0.608
  - Timing Consistency Student: η² = 0.478
- **Red bars (Tutor features)**: Barely visible, all < 0.01
- **Blue bars (Conversation features)**: Moderate at 0.070

**The takeaway**: This is visual proof that students drive temporal patterns. The two tallest bars are both student features. Tutor bars are essentially flat (noise). Conversation-level bars are moderate but miss 87% of the signal that student-only features capture.

---

### Figure 2: Feature Profiles Heatmap

![Feature Profiles](../figures/2023_conversations_two-participants/role_15d/figure2_feature_profiles.png)

**How to read this**:
- Each row = one temporal feature
- Each column = one cluster
- Colors show relative values within that row (green = high for that feature, red = low)
- Numbers = actual average values

**The three sections**:
1. **Top 5 rows**: Conversation-level features (all messages)
2. **Middle 5 rows**: Tutor-only features
3. **Bottom 5 rows**: Student-only features

**What to look for - the color pattern**:
- **Top section (Conversation)**: Some color differences, but moderate
- **Middle section (Tutor)**: Almost identical colors in both columns → tutors are the same regardless of cluster
- **Bottom section (Student)**: Dramatic color flips → THIS is where the two clusters differ

**The key rows** (bottom two):
- Row "burst_coefficient_student": 0.077 (greenish) vs -0.942 (deep red) → complete opposites
- Row "timing_consistency_student": 0.439 (red) vs 0.968 (green) → nearly doubled

**What this means**:

The two clusters represent different student engagement strategies:

**Cluster 0 (93%): "Natural Flow"**
- Student burst coefficient: 0.077 (slightly bursty)
- Student timing consistency: 0.439 (moderate)
- **Looks like**: Student asks question, waits, thinks, asks follow-up. Natural conversation rhythm with bursts and pauses.

**Cluster 1 (7%): "Methodical Worker"**
- Student burst coefficient: -0.942 (extremely steady, anti-bursty)
- Student timing consistency: 0.968 (nearly perfect)
- **Looks like**: Student working through problem set, one message every 2-3 minutes like clockwork. Sustained, consistent engagement.

**Why tutors are identical across clusters**: They adapt to whatever the student does. No tutor "signature" exists - they follow the student's lead.

**Open questions**:
- Does the "Methodical Worker" pattern predict better learning outcomes?
- Or does it just reflect problem type (structured exercises vs exploration)?
- Can tutors encourage students toward more consistent engagement?

---

### Figure 4: Why Role Decomposition Matters

![Role Decomposition Diagram](../figures/2023_conversations_two-participants/role_15d/figure4_role_decomposition.png)

**What this shows**: How the traditional 5D approach compares to our 15D role-specific approach.

**The comparison**:

**Traditional 5D Approach** (top):
- Compute each feature across entire conversation (all messages mixed together)
- Result: Moderate effect sizes (η² ≈ 0.07) - weak signal

**Role-Specific 15D Approach** (bottom, three boxes):
- **Red box (Tutor)**: η² < 0.01 - negligible
- **Blue box (Conversation)**: η² ≈ 0.07 - weak
- **Green box (Student)**: η² > 0.4 - strong ← THE signal

**Why decomposition reveals so much more**:

When you average tutor and student timing:
- Tutors: relatively constant pacing (low variance)
- Students: highly variable pacing (high variance)
- Average: moderate variance ← loses 87% of the signal

**Analogy**: Measuring room temperature with a space heater
- 5D: Average temperature across entire room = "moderately warm" (can't find heat source)
- 15D: Measure near vs far from heater = reveals exact location of heat source

**Methodological lesson**: In asymmetric interactions (expert-novice, interviewer-interviewee, doctor-patient), always decompose by role first. Averaging destroys the signal.

---

### Summary: The Complete Story

**Figure 1**: Students drive patterns (η² = 0.608), tutors don't (η² < 0.01)

**Figure 2**: Two student archetypes exist - "Natural Flow" (93%) vs "Methodical Worker" (7%)

**Figure 4**: Traditional conversation-level analysis misses this by averaging roles together

**Bottom line**: Temporal patterns in tutoring reflect student engagement strategies. Tutors adapt to students rather than imposing structure. Role decomposition is essential for understanding asymmetric interactions.

## Technical Details

### Clustering Method
- Algorithm: Hierarchical clustering with Ward linkage
- Features: 15D role-specific (5 student + 5 tutor + 5 conversation)
- Optimal clusters: k = 2 (determined by silhouette analysis)
- Silhouette score: 0.205
- Sample: 6,964 two-participant conversations from 2023

### Cluster Sizes

| Cluster | Count | Percentage | Label |
|---------|-------|------------|-------|
| 0 | 6,474 | 93.0% | "Natural Flow" |
| 1 | 490 | 7.0% | "Methodical Worker" |

### What Separates the Clusters

The primary discriminator is **student burst coefficient** (η² = 0.608):

**Cluster 0 "Natural Flow"**:
- Student BC: 0.077 (slightly bursty)
- Student TC: 0.439 (moderate consistency)
- Interpretation: Messages come in small bursts with gaps - typical conversation rhythm

**Cluster 1 "Methodical Worker"**:
- Student BC: -0.942 (extremely steady, anti-bursty)
- Student TC: 0.968 (nearly perfect consistency)
- Interpretation: Sustained, regular engagement like working through structured problems

## Research Implications

### For Platform Design: Support Student Pacing Diversity

Rather than prescribing conversation rhythms, platforms should accommodate different student engagement strategies:

**Design questions**:
- Does "Methodical Worker" pacing predict better learning outcomes?
- Should platforms encourage consistent engagement patterns?
- Can we match students with tutors based on temporal compatibility?

### For Tutor Training: Adaptation is a Feature

The negligible tutor effect sizes (η² < 0.01) show successful adaptation is already happening. This is good pedagogy.

**Training implications**:
- Teach tutors to recognize student temporal patterns (bursty vs steady)
- Emphasize flexibility as a skill, not inconsistency as a weakness
- Avoid prescribing rigid pacing strategies

### For Future Analysis: Always Decompose by Role

In asymmetric dyadic interactions (expert-novice, interviewer-interviewee, doctor-patient), conversation-level features obscure who drives the dynamics.

**Methodological recommendation**: Compute role-specific features before analyzing. Aggregation destroys signal in asymmetric interactions.

### For Power User Analysis

Our [power user data](power_user_analysis.md) shows high-volume tutors (100+ conversations). Open questions:
- Do experienced tutors show more temporal flexibility?
- Does tutor experience predict better student engagement patterns?
- Do certain tutors encourage students toward "Methodical Worker" behavior?

## Methodological Contribution

### When to Use Role-Specific Decomposition

**Use conversation-level features (5D)** when:
- Roles are symmetric (peer collaboration, friend chat)
- You believe both participants contribute equally
- You want emergent interaction dynamics

**Use role-specific features (15D)** when:
- **Roles are asymmetric** (expert-novice, interviewer-interviewee, doctor-patient)
- You want to know **who drives** the pattern
- Power dynamics or expertise differences exist

**Key insight from this analysis**: In tutoring, role decomposition reveals 87% more signal than conversation-level analysis. The student drives temporal patterns; averaging with tutor behavior destroys most of the signal.

### Trade-offs

**15D costs**:
- 3x more features (higher dimensionality)
- Larger sample sizes needed
- More complex interpretation

**15D benefits**:
- Reveals hidden signals obscured by averaging
- Identifies which role drives the pattern
- Actionable insights (e.g., "focus on student behavior, not tutor")

## Next Steps

### 1. Characterize "Methodical Worker" Students (7%)
- Do they have better learning outcomes than "Natural Flow" students?
- Are they working on different subject areas or problem types?
- Can we identify demographic or behavioral predictors?

### 2. Quantify Tutor Adaptation
- Measure within-tutor variance across different student types
- Create "flexibility scores" for tutors
- Test: Do flexible tutors get better student outcomes?

### 3. Stratify by Experience Level
- Novice tutors (< 10 convos) vs experts (100+): Does effect size change?
- Regular vs power users: Do different patterns emerge?

### 4. Predict Outcomes
Does student cluster membership predict:
- Session completion rates?
- Student satisfaction scores?
- Learning gains (if available)?

### 5. Hierarchical Modeling
Model conversations as nested within student temporal clusters:
```
θ_conversation ~ Normal(μ_cluster[student_type], σ)
```
Allows for cluster-level and individual-level effects.

## Files and Code

**Analysis scripts**:
- [`scripts/01b_extract_role_features.py`](../scripts/01b_extract_role_features.py) - Extract 15D role-specific features
- [`scripts/02b_cluster_analysis_role.py`](../scripts/02b_cluster_analysis_role.py) - Perform 15D clustering
- [`scripts/04_compare_5d_vs_15d.py`](../scripts/04_compare_5d_vs_15d.py) - Direct comparison analysis
- [`scripts/05b_create_figures_role.py`](../scripts/05b_create_figures_role.py) - Generate visualizations

**Results**:
- [`results/2023_conversations_two-participants_role_clustering_results.json`](../results/2023_conversations_two-participants_role_clustering_results.json) - Full clustering output with effect sizes

**Data**:
- [`data/processed/2023_conversations_two-participants_role_features.parquet`](../data/processed/2023_conversations_two-participants_role_features.parquet) - 15D feature matrix

---

**Last Updated**: November 2025
**Author**: Mike Ion
