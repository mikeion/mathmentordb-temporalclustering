#!/usr/bin/env python3
"""
Role-Level Temporal Feature Extraction

This script computes temporal features separately for tutor and student roles:
- BC_tutor, BC_student: Burstiness by role
- CD_tutor, CD_student: Cluster density by role
- RA_tutor, RA_student: Response acceleration by role
- MC_tutor, MC_student: Memory coefficient by role
- TC_tutor, TC_student: Timing consistency by role

Plus the original conversation-level features (BC, CD, RA, MC, TC).

Total: 15 features per conversation (5 conversation + 10 role-specific)

Input: data/raw/<dataset>.parquet
Output: data/processed/<dataset_name>_role_features.parquet

Usage:
    python scripts/01b_extract_role_features.py
    python scripts/01b_extract_role_features.py --dataset 2023_conversations_two-participants.parquet
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
# TEMPORAL FEATURE CALCULATIONS (same as 01_extract_features.py)
# ==============================================================================

def calculate_burst_coefficient(timestamps):
    """BC = (σ - μ) / (σ + μ) for inter-event times"""
    if len(timestamps) < 2:
        return 0.0

    intervals = np.diff(timestamps)
    if len(intervals) == 0:
        return 0.0

    sigma = np.std(intervals)
    mu = np.mean(intervals)

    if sigma + mu == 0:
        return 0.0

    return float((sigma - mu) / (sigma + mu))


def calculate_cluster_density(timestamps, window_minutes=5):
    """CD = Var(counts) / Mean(counts) in sliding windows"""
    if len(timestamps) < 2:
        return 0.0

    window_size = window_minutes * 60
    start_time = timestamps.min()
    end_time = timestamps.max()
    duration = end_time - start_time

    if duration < window_size:
        return 0.0

    # Sliding window counts
    window_counts = []
    current_time = start_time

    while current_time <= end_time - window_size:
        count = np.sum((timestamps >= current_time) &
                      (timestamps < current_time + window_size))
        window_counts.append(count)
        current_time += window_size / 2  # 50% overlap

    if len(window_counts) < 2:
        return 0.0

    mean_count = np.mean(window_counts)
    if mean_count == 0:
        return 0.0

    return float(np.var(window_counts) / mean_count)


def calculate_response_acceleration(timestamps):
    """RA = slope of inter-event times (negative = speeding up)"""
    if len(timestamps) < 3:
        return 0.0

    intervals = np.diff(timestamps)
    if len(intervals) < 2:
        return 0.0

    # Linear regression: intervals ~ time
    x = np.arange(len(intervals))
    slope = np.polyfit(x, intervals, 1)[0]

    return float(slope)


def calculate_memory_coefficient(timestamps):
    """MC = Pearson correlation of successive inter-event times"""
    if len(timestamps) < 3:
        return 0.0

    intervals = np.diff(timestamps)
    if len(intervals) < 2:
        return 0.0

    # Correlation between t[i] and t[i+1]
    x = intervals[:-1]
    y = intervals[1:]

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    correlation = np.corrcoef(x, y)[0, 1]

    if np.isnan(correlation):
        return 0.0

    return float(correlation)


def calculate_timing_consistency(timestamps):
    """TC = 1 / (1 + CV) where CV = σ/μ"""
    if len(timestamps) < 2:
        return 0.0

    intervals = np.diff(timestamps)
    if len(intervals) == 0:
        return 0.0

    mu = np.mean(intervals)
    sigma = np.std(intervals)

    if mu == 0:
        return 0.0

    cv = sigma / mu

    return float(1 / (1 + cv))


def extract_temporal_features(timestamps):
    """Extract all 5 temporal features from a sequence of timestamps"""
    if len(timestamps) < 2:
        return {
            'burst_coefficient': 0.0,
            'cluster_density': 0.0,
            'response_acceleration': 0.0,
            'memory_coefficient': 0.0,
            'timing_consistency': 0.0
        }

    return {
        'burst_coefficient': calculate_burst_coefficient(timestamps),
        'cluster_density': calculate_cluster_density(timestamps),
        'response_acceleration': calculate_response_acceleration(timestamps),
        'memory_coefficient': calculate_memory_coefficient(timestamps),
        'timing_consistency': calculate_timing_consistency(timestamps)
    }


# ==============================================================================
# ROLE-SPECIFIC FEATURE EXTRACTION
# ==============================================================================

def extract_role_features(conversation_df):
    """
    Extract temporal features separately for tutor and student.

    Returns dict with:
    - Conversation-level: BC, CD, RA, MC, TC
    - Tutor-level: BC_tutor, CD_tutor, RA_tutor, MC_tutor, TC_tutor
    - Student-level: BC_student, CD_student, RA_student, MC_student, TC_student
    """
    # Full conversation timestamps
    all_timestamps = conversation_df['unix_timestamp'].values

    # Tutor timestamps
    tutor_df = conversation_df[conversation_df['role'] == 'tutor']
    tutor_timestamps = tutor_df['unix_timestamp'].values if len(tutor_df) > 0 else np.array([])

    # Student timestamps
    student_df = conversation_df[conversation_df['role'] == 'student']
    student_timestamps = student_df['unix_timestamp'].values if len(student_df) > 0 else np.array([])

    # Extract features for each
    conv_features = extract_temporal_features(all_timestamps)
    tutor_features = extract_temporal_features(tutor_timestamps)
    student_features = extract_temporal_features(student_timestamps)

    # Combine into single feature dict
    features = {}

    # Conversation-level (original features)
    for key, val in conv_features.items():
        features[key] = val

    # Role-specific features
    for key, val in tutor_features.items():
        features[f'{key}_tutor'] = val

    for key, val in student_features.items():
        features[f'{key}_student'] = val

    return features


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Extract role-specific temporal burstiness features'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='mathconverse_full_dataset.parquet',
        help='Dataset filename in data/ directory (default: mathconverse_full_dataset.parquet)'
    )
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    # Input/output files
    input_file = raw_dir / args.dataset
    dataset_name = args.dataset.replace('.parquet', '')
    output_file = processed_dir / f"{dataset_name}_role_features.parquet"

    # Check input exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.error(f"Please place {args.dataset} in the data/raw/ directory")
        logger.error("\nAvailable datasets:")
        for f in sorted(raw_dir.glob("*.parquet")):
            if not f.name.endswith('_features.parquet'):
                logger.error(f"  - {f.name}")
        return 1

    # Ensure output directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {input_file.name}")
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded {len(df):,} messages across {df['conversation_id'].nunique():,} conversations")

    # Convert timestamps
    logger.info("Converting timestamps...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['unix_timestamp'] = df['timestamp'].astype(np.int64) / 1e9

    # Create role column if it doesn't exist
    if 'role' not in df.columns:
        logger.info("Creating 'role' column from student/helper indicators...")

        if 'student' in df.columns and 'helper' in df.columns:
            # Convert to numeric if needed
            df['student'] = pd.to_numeric(df['student'], errors='coerce').fillna(0)
            df['helper'] = pd.to_numeric(df['helper'], errors='coerce').fillna(0)

            # Create role column
            df['role'] = 'unknown'
            df.loc[df['student'] == 1, 'role'] = 'student'
            df.loc[df['helper'] == 1, 'role'] = 'tutor'

            logger.info(f"  Students: {(df['role'] == 'student').sum():,} messages")
            logger.info(f"  Tutors: {(df['role'] == 'tutor').sum():,} messages")
            logger.info(f"  Unknown: {(df['role'] == 'unknown').sum():,} messages")
        else:
            logger.error("Dataset must have either 'role' or both 'student'/'helper' columns")
            logger.error(f"Available columns: {df.columns.tolist()}")
            return 1

    # Filter conversations with enough messages
    min_messages = 3
    conversation_sizes = df.groupby('conversation_id').size()
    valid_conversations = conversation_sizes[conversation_sizes >= min_messages].index
    df = df[df['conversation_id'].isin(valid_conversations)]

    logger.info(f"Filtered to {len(valid_conversations):,} conversations with ≥{min_messages} messages")

    # Extract features for each conversation
    logger.info("Extracting role-specific temporal features...")

    features_list = []
    total_conversations = len(valid_conversations)

    for i, conv_id in enumerate(valid_conversations):
        if (i + 1) % 5000 == 0:
            logger.info(f"  Progress: {i+1:,}/{total_conversations:,} ({100*(i+1)/total_conversations:.1f}%)")

        conv_df = df[df['conversation_id'] == conv_id].sort_values('unix_timestamp')

        # Extract role-specific features
        features = extract_role_features(conv_df)
        features['conversation_id'] = conv_id

        features_list.append(features)

    # Create DataFrame
    features_df = pd.DataFrame(features_list)

    # Reorder columns: conversation_id first, then conversation-level, then role-specific
    conversation_cols = ['conversation_id']
    conv_level_cols = ['burst_coefficient', 'cluster_density', 'response_acceleration',
                       'memory_coefficient', 'timing_consistency']
    tutor_cols = [f'{col}_tutor' for col in conv_level_cols]
    student_cols = [f'{col}_student' for col in conv_level_cols]

    features_df = features_df[conversation_cols + conv_level_cols + tutor_cols + student_cols]

    # Summary statistics
    logger.info(f"Extracted features for {len(features_df):,} conversations")
    logger.info("\nFeature Statistics:")
    logger.info(features_df.describe().to_string())

    # Save
    features_df.to_parquet(output_file, index=False)

    logger.info(f"\nSaved: Role-specific features saved to {output_file}")
    logger.info(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"  Conversations: {len(features_df):,}")
    logger.info(f"  Total features: {len(features_df.columns) - 1}")  # -1 for conversation_id

    return 0


if __name__ == "__main__":
    sys.exit(main())
