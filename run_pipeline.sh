#!/bin/bash

set -e

# Default dataset
DATASET="${1:-mathconverse_full_dataset.parquet}"

echo "=========================================="
echo "Temporal Clustering Pipeline"
echo "=========================================="
echo "Dataset: $DATASET"
echo "=========================================="
echo ""

echo "[1/3] Extracting temporal features..."
python3 scripts/01_extract_features.py --dataset "$DATASET"
echo ""

echo "[2/3] Running cluster analysis..."
python3 scripts/02_cluster_analysis.py
echo ""

echo "[3/3] Creating figures..."
python3 scripts/03_create_figures.py
echo ""

DATASET_NAME="${DATASET%.parquet}"

echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Features: data/processed/${DATASET_NAME}_features.parquet"
echo "  - Clustering: results/${DATASET_NAME}_clustering_results.json"
echo "  - Figures: figures/${DATASET_NAME}/*.png, *.pdf"
