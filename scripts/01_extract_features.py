#!/usr/bin/env python3
"""
Extract Temporal Burstiness Features from Educational Conversations

This script computes 5 temporal features that capture learning dynamics:
- BC (Burst Coefficient): Irregularity of conversation rhythm
- CD (Cluster Density): How much activity clusters into intense bursts
- RA (Response Acceleration): Is conversation speeding up or slowing down
- MC (Memory Coefficient): Temporal correlation/rhythm
- TC (Timing Consistency): Stable vs erratic engagement

Input: data/raw/<dataset>.parquet (default: mathconverse_full_dataset.parquet)
Output: data/processed/<dataset_name>_features.parquet

Usage:
    python scripts/01_extract_features.py
    python scripts/01_extract_features.py --dataset 2023_conversations_two-participants.parquet
    python scripts/01_extract_features.py --dataset mathconverse_full_dataset.parquet
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# BURSTINESS FEATURE CALCULATIONS
# ==============================================================================

def calculate_burst_coefficient(timestamps):
    """
    BC = (σ - μ) / (σ + μ)

    Returns:
        BC ≈ +1: Very bursty (irregular timing)
        BC ≈  0: Random/Poisson process
        BC ≈ -1: Perfectly regular (like a metronome)
    """
    intervals = np.diff(timestamps)
    sigma = np.std(intervals)
    mu = np.mean(intervals)

    if sigma + mu == 0:
        return 0

    return float((sigma - mu) / (sigma + mu))


def calculate_cluster_density(timestamps, window_minutes=None):
    """
    CD = Var(window_counts) / Mean(window_counts)

    Uses adaptive window: 20% of message span, bounded [60s, 300s]
    This ensures the metric works for conversations of varying lengths.

    Returns:
        Higher = more concentrated bursts (downpour)
        Lower = evenly distributed (drizzle)
    """
    if len(timestamps) < 3:
        return 0.0

    # Calculate span
    span = timestamps[-1] - timestamps[0]
    if span < 1:
        return 0.0

    # Adaptive window: 20% of span, bounded [60s, 300s]
    if window_minutes is None:
        window_size = span * 0.2
        window_size = max(60, min(300, window_size))
    else:
        window_size = window_minutes * 60

    window_counts = []
    for i, t in enumerate(timestamps):
        window_end = t + window_size
        count = sum(1 for ts in timestamps[i:] if ts < window_end)
        window_counts.append(count)

    if len(window_counts) == 0 or np.mean(window_counts) == 0:
        return 0

    return float(np.var(window_counts) / np.mean(window_counts))


def calculate_response_acceleration(timestamps):
    """
    RA = slope of linear fit to intervals over time

    Returns:
        Negative = speeding up (breakthrough)
        Positive = slowing down (frustration)
        ~0 = constant pace
    """
    intervals = np.diff(timestamps)

    if len(intervals) < 2:
        return 0

    time_indices = np.arange(len(intervals))
    slope, _ = np.polyfit(time_indices, intervals, 1)

    return float(slope)


def calculate_memory_coefficient(timestamps):
    """
    MC = Pearson correlation between successive intervals

    Returns:
        MC > 0: Long times follow long times (rhythmic)
        MC < 0: Alternating pattern (reactive)
        MC ≈ 0: No temporal memory (random)
    """
    intervals = np.diff(timestamps)

    if len(intervals) < 2:
        return 0

    T_i = intervals[:-1]
    T_i_plus_1 = intervals[1:]

    if np.std(T_i) == 0 or np.std(T_i_plus_1) == 0:
        return 0

    correlation = np.corrcoef(T_i, T_i_plus_1)[0, 1]

    return 0 if np.isnan(correlation) else float(correlation)


def calculate_timing_consistency(timestamps):
    """
    TC = 1 / (1 + CV) where CV = σ/μ

    Returns:
        TC ≈ 1: Very consistent (sustained focus)
        TC ≈ 0.5: Moderate variation
        TC ≈ 0: Highly erratic
    """
    intervals = np.diff(timestamps)

    if len(intervals) == 0 or np.mean(intervals) == 0:
        return 0

    cv = np.std(intervals) / np.mean(intervals)

    return float(1 / (1 + cv))


def extract_burstiness_features(timestamps):
    """
    Extract all 5 temporal features.

    Args:
        timestamps: Array of Unix timestamps (seconds since epoch)

    Returns:
        Dictionary with 5 features, or None if < 3 messages
    """
    if len(timestamps) < 3:
        return None

    return {
        'burst_coefficient': calculate_burst_coefficient(timestamps),
        'cluster_density': calculate_cluster_density(timestamps),
        'response_acceleration': calculate_response_acceleration(timestamps),
        'memory_coefficient': calculate_memory_coefficient(timestamps),
        'timing_consistency': calculate_timing_consistency(timestamps)
    }


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Extract temporal burstiness features from conversation data'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='mathconverse_full_dataset.parquet',
        help='Dataset filename in data/ directory (default: mathconverse_full_dataset.parquet)'
    )
    args = parser.parse_args()

    # Paths (relative to script location)
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    # Input from raw, output to processed with dataset-specific name
    input_file = raw_dir / args.dataset
    dataset_name = args.dataset.replace('.parquet', '')
    output_file = processed_dir / f"{dataset_name}_features.parquet"

    # Check input exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.error(f"Please place {args.dataset} in the data/raw/ directory")
        logger.error("\nAvailable datasets:")
        for f in sorted(raw_dir.glob("*.parquet")):
            if not f.name.endswith('_features.parquet'):
                logger.error(f"  - {f.name}")
        return 1

    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {input_file.name}")
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded {len(df):,} messages across {df['conversation_id'].nunique():,} conversations")

    # Convert timestamps to Unix time (seconds since epoch)
    logger.info("Converting timestamps...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract features per conversation
    logger.info("Extracting burstiness features...")

    results = []
    conv_ids = df['conversation_id'].unique()

    for i, conv_id in enumerate(conv_ids):
        if (i + 1) % 5000 == 0:
            logger.info(f"  Progress: {i+1:,}/{len(conv_ids):,} ({(i+1)/len(conv_ids)*100:.1f}%)")

        # Get conversation messages sorted by time
        conv_df = df[df['conversation_id'] == conv_id].sort_values('timestamp')

        if len(conv_df) < 3:
            continue

        # Extract timestamps as Unix time (seconds)
        timestamps = conv_df['timestamp'].astype(np.int64) / 1e9
        timestamps = timestamps.values

        # Compute features
        features = extract_burstiness_features(timestamps)

        if features is None:
            continue

        features['conversation_id'] = conv_id
        results.append(features)

    logger.info(f"Extracted features for {len(results):,} conversations")

    # Create DataFrame
    features_df = pd.DataFrame(results)

    # Reorder columns (conversation_id first)
    cols = ['conversation_id', 'burst_coefficient', 'cluster_density',
            'response_acceleration', 'memory_coefficient', 'timing_consistency']
    features_df = features_df[cols]

    # Summary statistics
    logger.info("\nFeature Statistics:")
    logger.info(features_df.describe().round(3).to_string())

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_file, index=False)

    logger.info(f"\nSaved: Features saved to {output_file}")
    logger.info(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"  Conversations: {len(features_df):,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
