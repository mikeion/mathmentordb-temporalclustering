# Temporal Clustering Analysis

Analyzing message timing patterns in online tutoring conversations to understand student engagement strategies.

## What This Does

This pipeline analyzes **when** messages occur in tutor-student conversations (not what they say). It discovers distinct temporal engagement patterns and identifies which participant - student or tutor - drives the conversational rhythm.

**Input**: Conversation timestamps from ~7,000 tutoring sessions
**Output**: Temporal engagement clusters and statistical analysis of role-specific patterns

## Key Finding

**Students drive the temporal patterns in tutoring conversations, not tutors.**

- Student message timing explains 60.8% of variance between conversation types
- Tutor message timing explains <1%
- Two student engagement archetypes: "Natural Flow" (93%) and "Methodical Worker" (7%)

This suggests effective tutors adapt their pacing to student needs rather than imposing a fixed rhythm.

**[Full analysis and implications →](docs/role_decomposition_findings.md)**

## How It Works

The pipeline extracts temporal features from message timestamps, then uses unsupervised clustering to discover engagement patterns.

**Temporal features measured** (5 features):
- Burst Coefficient: Bursty vs steady message flow
- Cluster Density: How clumped messages are
- Response Acceleration: Pacing changes over time
- Memory Coefficient: Temporal autocorrelation
- Timing Consistency: Regularity of rhythm

**Two analysis modes**:
1. **5D**: Conversation-level features (averages both participants)
2. **15D**: Role-specific features (separate metrics for student, tutor, and conversation)

## Usage

### Full Pipeline (Recommended)

Run the role-specific (15D) analysis to discover who drives temporal patterns:

```bash
# 1. Extract temporal features by role
python scripts/01b_extract_role_features.py --dataset your_data.parquet

# 2. Cluster conversations
python scripts/02b_cluster_analysis_role.py

# 3. Characterize the clusters
python scripts/03_characterize_clusters.py

# 4. Generate visualizations
python scripts/05b_create_figures_role.py
```

### Alternative: Conversation-Level Analysis Only

For simpler analysis without role decomposition:

```bash
python scripts/01_extract_features.py --dataset your_data.parquet
python scripts/02_cluster_analysis.py
python scripts/05_create_figures_5d.py
```

### Input Data Format

Your input data should be a Parquet file with these columns:

**Required:**
- `conversation_id`: Unique conversation identifier
- `timestamps`: List of message timestamps (ISO format or Unix milliseconds)

**Optional:**
- `num_messages`: Message count
- `author_role`: Participant roles (needed for 15D analysis)
- Any other metadata to preserve

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

df.to_parquet('data/your_data.parquet')
```

### Output Files

**Data:**
- `data/processed/*_features.parquet` - Extracted temporal features
- `data/processed/*_role_features.parquet` - Role-specific features (15D)
- `results/*_clustering_results.json` - Cluster assignments and statistics

**Visualizations:**
- `figures/*/role_15d/` - Role decomposition figures
- `figures/*/5d/` - Conversation-level figures

## Requirements

```
Python 3.8+
pandas >= 1.5.0
numpy >= 1.23.0
scikit-learn >= 1.2.0
scipy >= 1.9.0
matplotlib >= 3.6.0
seaborn >= 0.12.0
umap-learn >= 0.5.0 (optional, for UMAP visualization)
```

Install with: `pip install -r requirements.txt`

## Citation

If you use this pipeline, please cite:

```
[Paper citation to be added]
```

## Contact

For questions or issues, contact [contact info to be added].
