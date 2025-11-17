# Temporal Clustering Analysis

Discovery of temporal engagement patterns in tutoring conversations using role-specific temporal features.

## Key Findings

### 1. Students Drive Temporal Patterns (Not Tutors)

When we decompose temporal features by role (15D analysis), we discover:
- **Student features** explain **60.8%** of cluster variance (η²=0.608)
- **Tutor features** explain **<1%** of variance (η²<0.01)
- **Conversation-level features** (5D) obscure this signal by averaging roles

**Implication**: Temporal archetypes reflect student engagement strategies, not tutor pacing. Tutors adapt to students, not vice versa.

**[Read the detailed analysis: Role Decomposition Findings](docs/role_decomposition_findings.md)**

### 2. Two Student Temporal Archetypes

**Cluster 0** (93% of conversations): "Standard Engagement"
- Moderate burst coefficient, moderate consistency
- Typical student pacing patterns

**Cluster 1** (7% of conversations): "Steady & Consistent"
- Very low burst (anti-bursty, steady message flow)
- Very high timing consistency (η² = 0.478)
- Likely highly engaged or focused students

### 3. Methodological Contribution

This analysis demonstrates the value of **role-specific feature decomposition** in dyadic interactions:
- Conversation-level analysis (5D) loses **87% of signal strength** by averaging
- Role decomposition (15D) reveals asymmetric dynamics hidden in aggregate statistics
- Critical for understanding expert-novice, tutor-student, or any asymmetric interaction

## Overview

This pipeline analyzes conversation timing patterns to discover temporal engagement archetypes. It includes:
1. **5D analysis**: Conversation-level temporal features
2. **15D analysis**: Role-specific features (conversation + tutor + student)
3. Unsupervised clustering with statistical validation
4. Outcome comparison and interpretability analysis

## Pipeline Steps

### 01: Extract Features
Two approaches for feature extraction:

**01_extract_features.py** - 5D conversation-level features:
- Burst Coefficient (BC): Rhythm irregularity
- Cluster Density (CD): Temporal clumpiness
- Response Acceleration (RA): Pace change
- Memory Coefficient (MC): Autocorrelation
- Timing Consistency (TC): Stability

**01b_extract_role_features.py** - 15D role-specific features:
- 5 conversation-level (BC, CD, RA, MC, TC)
- 5 tutor-level (BC_tutor, CD_tutor, RA_tutor, MC_tutor, TC_tutor)
- 5 student-level (BC_student, CD_student, RA_student, MC_student, TC_student)

### 02: Cluster Analysis
Two clustering pipelines:

**02_cluster_analysis.py** - 5D clustering:
- Tests k=2-15 using 5 metrics (silhouette, Calinski-Harabasz, Davies-Bouldin, gap statistic, elbow)
- Compares methods (K-means, hierarchical, DBSCAN)
- Computes effect sizes

**02b_cluster_analysis_role.py** - 15D role-specific clustering:
- Same methodology as 02, but with role decomposition
- Reveals which role (tutor vs student) drives patterns

### 03-06: Analysis & Validation

**03_characterize_clusters.py** - Compute outcomes + sample conversations from each cluster
**04_compare_5d_vs_15d.py** - Compare conversation-level vs role-specific results
**05_create_figures_5d.py** - Visualization (5D results only)
**05b_create_figures_role.py** - Visualization for 15D role-specific results
**06_create_meeting_figures.py** - Private/meeting figures

**Utilities** (unnumbered):
- `analyze_power_users.py` - Verify top users are human (data quality check)

## Quick Start

**Recommended: Role-specific (15D) analysis**

```bash
# 1. Extract role-level features
python scripts/01b_extract_role_features.py --dataset mathconverse_full_dataset.parquet

# 2. Run clustering
python scripts/02b_cluster_analysis_role.py

# 3. Characterize clusters
python scripts/03_characterize_clusters.py

# 4. Compare to 5D results
python scripts/04_compare_5d_vs_15d.py

# 5. Create visualizations
python scripts/05b_create_figures_role.py
```

**Alternative: Conversation-level (5D) analysis only**

```bash
bash run_pipeline.sh mathconverse_full_dataset.parquet
# Or individually:
python scripts/01_extract_features.py
python scripts/02_cluster_analysis.py
python scripts/05_create_figures_5d.py
```

## Input Data Format

The pipeline expects a parquet file with conversation data:

**Required columns:**
- `conversation_id`: Unique identifier
- `timestamps`: List of message timestamps (ISO format or unix milliseconds)

**Optional columns:**
- `num_messages`: Message count (for success heuristic)
- Any additional metadata to preserve

**Example:**
```python
import pandas as pd

df = pd.DataFrame({
    'conversation_id': ['conv_001', 'conv_002'],
    'timestamps': [
        ['2024-01-01 10:00:00', '2024-01-01 10:02:30', ...],
        [1704110400000, 1704110550000, ...]
    ],
    'num_messages': [15, 8]
})

df.to_parquet('data/mathconverse_full_dataset.parquet')
```

## Output Files

### Data Outputs
- `data/conversation_features.parquet`: Features for each conversation
- `results/clustering_results.json`: Complete clustering analysis results

### Figure Outputs
- `figures/figure1_optimal_k.png/pdf`: Optimal k determination
- `figures/figure2_umap_clusters.png/pdf`: Cluster visualization
- `figures/figure3_feature_comparison.png/pdf`: Feature profiles by cluster
- `figures/figure4_summary_table.png/pdf`: Summary statistics

## Dependencies

Core requirements:
- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- scipy >= 1.9.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0

Optional:
- umap-learn >= 0.5.0 (for UMAP visualization)

## Configuration

### Dataset Selection

The pipeline supports multiple datasets. Available datasets in this repository:
- `mathconverse_full_dataset.parquet` (139MB, ~44k conversations) - **default**
- `2023_conversations_two-participants.parquet` (12MB, smaller subset)

To use a different dataset:
```bash
# Via shell script
bash run_pipeline.sh 2023_conversations_two-participants.parquet

# Via Python script directly
python scripts/01_extract_features.py --dataset 2023_conversations_two-participants.parquet
```

### Other Parameters

Default parameters can be modified in each script:

**01_extract_features.py:**
- Dataset selection via `--dataset` flag
- Minimum messages per conversation (currently: 3)

**02_cluster_analysis.py:**
- K-range for testing (default: 2-15)
- DBSCAN parameter ranges
- Hierarchical linkage methods

**03_create_figures.py:**
- Figure size and DPI
- Color schemes
- Plot style

## Expected Runtime

On a dataset with ~44,000 conversations:
- Feature extraction: ~2-3 minutes
- Cluster analysis: ~5-10 minutes
- Figure generation: ~30-60 seconds

Total: ~10-15 minutes

## Validation

The pipeline includes statistical validation:
- Silhouette score (cluster separation)
- Calinski-Harabasz index (variance ratio)
- Davies-Bouldin index (cluster compactness)
- Gap statistic (comparison to null)
- ANOVA F-tests and eta-squared effect sizes

Results are saved in `results/clustering_results.json` with:
- Optimal k recommendation and consensus
- Method comparison results
- Cluster statistics and effect sizes
- Individual conversation cluster assignments

## Troubleshooting

**Issue: "No module named 'umap'"**
- UMAP is optional. Script will skip UMAP visualization with a warning.
- To install: `pip install umap-learn`

**Issue: "Input file not found"**
- Check that the dataset exists in the `data/` directory
- Use `--dataset` flag to specify the correct filename
- The script will list available datasets if the file is not found

**Issue: Memory errors on large datasets**
- Reduce k_range in script 02 (e.g., range(2, 10) instead of range(2, 16))
- Process in batches or subsample data

## Citation

If you use this pipeline, please cite:

```
[Paper citation to be added]
```

## Contact

For questions or issues, contact [contact info to be added].
