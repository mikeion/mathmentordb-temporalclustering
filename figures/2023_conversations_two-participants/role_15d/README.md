# 15D Role-Specific Clustering Analysis Figures

This directory contains visualizations for the role-decomposed (15D) temporal feature analysis.

## Key Finding

**Student temporal patterns drive clustering** (η²=0.411 for burst coefficient, η²=0.333 for timing consistency), while tutor patterns show negligible effects (η²<0.04).

---

## Figures

### [Figure 1: Effect Size Comparison](figure1_effect_size_comparison.png)
**Student vs Tutor vs Conversation-level features**

Compares effect sizes (η²) across three feature types:
- Student features (right): Strong effects (0.33-0.41)
- Tutor features (left): Negligible effects (<0.04)
- Conversation-level (middle): Moderate effects (0.06-0.10)

Demonstrates that student temporal behavior, not tutor behavior, differentiates conversation clusters.

---

### [Figure 2: Feature Profiles Heatmap](figure2_feature_profiles.png)
**Mean feature values by cluster**

Heatmap showing standardized mean values for all 15 features across 2 clusters:
- Cluster 0 (88.4%): "Natural Flow" - lower student burst/consistency
- Cluster 1 (11.6%): "Methodical Worker" - higher student burst/consistency

---

### [Figure 2b: UMAP Projection](figure2b_umap_projection.png)
**2D visualization of cluster separation**

UMAP projection of 15D feature space showing:
- Green: Natural Flow cluster (n=6,155)
- Orange: Methodical Worker cluster (n=809)
- Multiple spatial blobs visible, suggesting potential for finer-grained clustering

**Note**: Visual overlap indicates modest separation (silhouette=0.322). The hierarchical k=2 split may not capture all natural structure in the data.

---

### [Figure 3: Cluster Overview](figure3_cluster_overview.png)
**Cluster sizes and basic statistics**

Shows distribution and composition of the two clusters with counts and percentages.

---

### [Figure 4: Role Decomposition Diagram](figure4_role_decomposition.png)
**Conceptual diagram of 5D→15D feature decomposition**

Illustrates how 5 conversation-level temporal features (BC, CD, RA, MC, TC) are decomposed into:
- 5 conversation-level features
- 5 tutor-specific features
- 5 student-specific features
- = 15 total dimensions

---

## Technical Details

**Clustering Method**: Hierarchical (Ward linkage) with k=2
**Preprocessing**: Winsorization at 1st/99th percentiles to handle extreme outliers
**Dataset**: 6,964 tutor-student conversations
**Generated**: 2025-11-20
