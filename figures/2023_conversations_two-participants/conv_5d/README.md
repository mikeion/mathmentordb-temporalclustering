# 5D Conversation-Level Clustering Analysis

This directory contains the initial clustering analysis using **conversation-level temporal features** (5D), where each metric aggregates both tutor and student behavior.

## Key Finding

DBSCAN identified **989 outlier conversations (14.2%)** that differ from the main cluster (5,975 conversations, 85.8%). However, this approach does not reveal **who drives** temporal patterns (tutor vs student).

**For the main research finding**, see [../role_15d/](../role_15d/) which decomposes features by role and reveals that **students, not tutors**, drive temporal clustering.

---

## Figures

### [Figure 1: Optimal K Analysis](figure1_optimal_k.png)
Shows clustering quality metrics (silhouette, gap statistic) across different k values.

### [Figure 2: UMAP Clusters](figure2_umap_clusters.png)
2D UMAP projection showing DBSCAN results:
- Main cluster (green)
- Outliers (cluster -1)

### [Figure 3: Feature Comparison](figure3_feature_comparison.png)
Compares feature distributions between main cluster and outliers.

### [Figure 4: Summary Table](figure4_summary_table.png)
Statistical summary of clusters.

---

## Technical Details

**Features:** 5 conversation-level temporal metrics
- Burst Coefficient (BC)
- Cluster Density (CD)
- Response Acceleration (RA)
- Memory Coefficient (MC)
- Timing Consistency (TC)

**Method:** DBSCAN (density-based clustering)
**Result:** 1 main cluster + 989 outliers (no distinct second cluster)
**Limitation:** Cannot distinguish tutor vs student contributions

**Dataset:** 6,964 tutor-student conversations
**Generated:** 2025-11-20
