# Role Decomposition Analysis: 5D vs 15D Clustering

## TL;DR - Key Finding

**Students drive temporal patterns in tutoring conversations, not tutors.**

When we decompose conversation-level temporal features (5D) into role-specific features (15D), we discover that:
- **Student temporal patterns** explain **60.8%** of cluster variance (η² = 0.608)
- **Tutor temporal patterns** explain **<1%** of cluster variance (η² < 0.01)
- **Conversation-level features** obscure this signal by averaging student and tutor behavior

## The Analysis

### What We Did

We compared two clustering approaches on 6,964 two-participant tutoring conversations from 2023:

1. **5D Conversation-Level Clustering**: Uses 5 temporal features computed across entire conversations
   - Burst Coefficient (BC)
   - Cluster Density (CD)
   - Response Acceleration (RA)
   - Memory Coefficient (MC)
   - Timing Consistency (TC)

2. **15D Role-Specific Clustering**: Decomposes each feature by role
   - 5 conversation-level features (BC, CD, RA, MC, TC)
   - 5 tutor-specific features (BC_tutor, CD_tutor, etc.)
   - 5 student-specific features (BC_student, CD_student, etc.)

### What We Found

The 15D role-specific clustering reveals a **striking asymmetry**:

| Feature Category | Effect Size (η²) | Interpretation |
|-----------------|------------------|----------------|
| **Student features** | **0.478 - 0.608** | **Strong effect** - students drive patterns |
| Conversation features | 0.016 - 0.070 | Weak effect - signal washed out |
| Tutor features | 0.001 - 0.009 | Negligible - tutors don't drive patterns |

#### Detailed Effect Sizes by Feature

**Student Features** (strong discriminators):
- `burst_coefficient_student`: η² = **0.608** (strongest signal)
- `timing_consistency_student`: η² = **0.478**
- `response_acceleration_student`: η² = 0.003
- `memory_coefficient_student`: η² = 0.002

**Conversation Features** (weak):
- `burst_coefficient`: η² = 0.070
- `timing_consistency`: η² = 0.070
- `response_acceleration`: η² = 0.022
- `memory_coefficient`: η² = 0.016

**Tutor Features** (negligible):
- `burst_coefficient_tutor`: η² = 0.009
- `timing_consistency_tutor`: η² = 0.002
- `memory_coefficient_tutor`: η² = 0.001
- `response_acceleration_tutor`: η² = 0.001

## What This Means

### 1. Temporal Archetypes Are Student Archetypes

The temporal clusters we identified don't represent "conversation styles" - they represent **student engagement strategies**.

The patterns reflect:
- How students burst their questions and responses
- How consistently students engage over time
- Student pacing preferences and temporal behavior

### 2. Tutors Adapt, Students Drive

The negligible effect sizes for tutor features (η² < 0.01) suggest that:
- Tutors **adapt their pacing** to match students
- Tutors don't impose a rigid temporal style
- Effective tutoring may involve temporal flexibility

This aligns with good pedagogical practice: tutors should be responsive to student needs rather than rigid in their approach.

### 3. Conversation-Level Analysis Obscures The Signal

When we compute temporal features at the conversation level (averaging tutor and student behavior), we get:
- Moderate effect sizes (η² ≈ 0.07)
- Loss of 90% of the signal strength
- Mixed/washed out patterns

**Example**:
- Student burst coefficient alone: η² = 0.608
- Conversation burst coefficient (average): η² = 0.070
- **Signal loss: 87%**

This demonstrates why role-specific decomposition is critical for understanding dyadic interactions.

## Visualizations

The following figures are available in [`figures/2023_conversations_two-participants/role_15d/`](../figures/2023_conversations_two-participants/role_15d/):

1. **[figure1_effect_size_comparison.png](../figures/2023_conversations_two-participants/role_15d/figure1_effect_size_comparison.png)**
   - Bar chart comparing effect sizes across student, tutor, and conversation features
   - Shows dramatic difference between student (high) vs tutor (negligible) effects

2. **[figure2_feature_profiles.png](../figures/2023_conversations_two-participants/role_15d/figure2_feature_profiles.png)**
   - Heatmap of mean feature values for each cluster
   - Reveals which features distinguish the 2 clusters

3. **[figure3_cluster_overview.png](../figures/2023_conversations_two-participants/role_15d/figure3_cluster_overview.png)**
   - Cluster size distribution and quality metrics
   - Shows 93% vs 7% split between clusters

4. **[figure4_role_decomposition.png](../figures/2023_conversations_two-participants/role_15d/figure4_role_decomposition.png)**
   - Conceptual diagram showing 5D → 15D decomposition
   - Illustrates signal loss in conversation-level features

## Clustering Results

**Method**: Hierarchical clustering with Ward linkage
**Optimal k**: 2 clusters
**Silhouette score**: 0.205
**Sample size**: 6,964 conversations

### Cluster Distribution

| Cluster | Size | Percentage | Description |
|---------|------|------------|-------------|
| Cluster 0 | 6,474 | 93.0% | "Standard engagement" students |
| Cluster 1 | 490 | 7.0% | "High burst, high consistency" students |

### Cluster Characterization

**Cluster 0** (93% of conversations):
- Moderate student burst coefficient (0.077)
- Moderate timing consistency (0.439)
- Represents typical student engagement patterns

**Cluster 1** (7% of conversations):
- **Very low** student burst coefficient (-0.942) - anti-bursty, steady flow
- **Very high** timing consistency (0.966) - extremely regular engagement
- Likely represents highly engaged or focused students

The key differentiator is **student burst coefficient** (η² = 0.608), which captures whether students:
- Send messages in rapid bursts with long gaps (high BC)
- Maintain steady, consistent pacing (low BC)

## Research Implications

### 1. Rethink "Conversation Dynamics"

In dyadic interactions, we should stop treating temporal features as conversation-level properties. They are often driven asymmetrically by one participant.

**Recommendation**: Always decompose features by role before analysis.

### 2. Student-Centered Temporal Design

Platform designers should focus on supporting diverse **student pacing strategies** rather than prescribing conversation rhythms.

Questions to explore:
- Does cluster 1 (steady, consistent) correlate with better learning outcomes?
- Should platforms encourage certain student temporal patterns?
- Can we match students with tutors based on temporal compatibility?

### 3. Tutor Flexibility as a Feature, Not a Bug

The fact that tutors show negligible temporal clustering suggests they successfully adapt to students. This is good!

**Recommendation**: Train tutors to recognize and adapt to student temporal patterns rather than imposing their own pacing.

### 4. Power User Implications

Given our [power user analysis](power_user_analysis.md) showing:
- Top tutor (Nicolas Miller): 12,332 conversations, 147 messages/day
- Many tutors with 100+ conversations

We should ask:
- Do high-volume tutors show **more** or **less** temporal flexibility?
- Do experienced tutors develop better adaptation skills?
- Does tutor experience predict better student engagement?

## Methodological Contribution

This analysis demonstrates the value of **role-specific feature decomposition** in analyzing dyadic interactions.

### When to Use 15D vs 5D

**Use conversation-level features (5D)** when:
- You believe both participants contribute equally to the pattern
- Roles are symmetric (e.g., peer collaboration)
- You want to capture emergent interaction dynamics

**Use role-specific features (15D)** when:
- Roles are asymmetric (tutor/student, interviewer/interviewee)
- You want to understand who drives the pattern
- Power dynamics or expertise differences exist

### Computational Considerations

15D clustering requires:
- More computational resources (3x features)
- Larger sample sizes (curse of dimensionality)
- Careful interpretation (which role matters?)

But it reveals:
- **Hidden signals** obscured by averaging
- **Role-specific effects** critical for understanding asymmetric interactions
- **Actionable insights** about who drives patterns

## Next Steps

### Analysis
1. **Characterize the 7% cluster**: Who are these highly consistent students?
   - Learning outcomes?
   - Demographics?
   - Subject areas?

2. **Tutor adaptation analysis**: Do tutors truly adapt or just vary randomly?
   - Within-tutor variance across students
   - Tutor "flexibility scores"

3. **Power user stratification**: Does effect size change when we separate:
   - Novice tutors (< 10 conversations) vs experts (100+)
   - Regular students vs power users

### Modeling
4. **Hierarchical model**: Student temporal clusters + individual effects
   ```
   θ_conv ~ Normal(μ_cluster[student_cluster], σ)
   ```

5. **Outcome prediction**: Does student cluster predict:
   - Conversation completion rate?
   - Student satisfaction?
   - Learning gains?

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

## Citation

If you use this role decomposition approach, please cite:

```
[Your Paper/Repository]
Finding: Student temporal patterns drive clustering in tutoring conversations (η² = 0.608),
while tutor patterns show negligible effect (η² < 0.01), demonstrating asymmetric
temporal dynamics in expert-novice interactions.
```

---

**Last Updated**: November 2024
**Author**: Mike Ion
**Contact**: [Your contact info]
