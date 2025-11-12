# Temporal Clustering Pipeline

Minimal, reproducible pipeline for extracting temporal burstiness features from conversation data and identifying temporal archetypes through clustering.

## Overview

This pipeline takes raw conversation timing data and produces:
1. Five temporal burstiness features per conversation
2. Optimal cluster count determination using multiple metrics
3. Comparison of clustering methods (DBSCAN, hierarchical, K-means)
4. Publication-quality figures and statistical validation

## Pipeline Steps

### Step 1: Extract Features
Computes five temporal features from conversation timestamps:
- **Burst Coefficient (BC)**: Deviation from regular timing
- **Cluster Density (CD)**: Intensity of temporal bursts
- **Response Acceleration (RA)**: Rate of change in response timing
- **Memory Coefficient (MC)**: Influence of past interactions
- **Timing Consistency (TC)**: Predictability of intervals

### Step 2: Cluster Analysis
Determines optimal number of clusters and applies best method:
- Tests k=2 through k=15 using 5 metrics
- Compares DBSCAN, hierarchical, and K-means clustering
- Computes cluster statistics and effect sizes
- Validates with ANOVA and eta-squared

### Step 3: Create Figures
Generates publication-ready visualizations:
- Optimal k determination (6-panel plot)
- UMAP cluster projection
- Feature comparison by cluster
- Summary statistics table

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run entire pipeline (uses mathconverse_full_dataset.parquet by default)
bash run_pipeline.sh

# Or run with a specific dataset
bash run_pipeline.sh 2023_conversations_two-participants.parquet

# Or run steps individually
python scripts/01_extract_features.py  # default dataset
python scripts/01_extract_features.py --dataset 2023_conversations_two-participants.parquet
python scripts/02_cluster_analysis.py
python scripts/03_create_figures.py
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
