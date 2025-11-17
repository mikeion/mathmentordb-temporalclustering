#!/usr/bin/env python3
"""
Characterize Temporal Clusters

Step 2: Compute outcomes by cluster (conversation length, message counts, etc.)
Step 3: Sample conversations from each cluster for manual inspection

Usage:
    python scripts/characterize_clusters.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    # Load clustering results
    results_file = project_dir / "results" / "2023_conversations_two-participants_role_clustering_results.json"

    if not results_file.exists():
        print(f"ERROR: {results_file} not found")
        print("Run 02b_cluster_analysis_role.py first!")
        return 1

    with open(results_file) as f:
        results = json.load(f)

    print("="*80)
    print("CLUSTER CHARACTERIZATION")
    print("="*80 + "\n")

    # Create cluster assignment lookup
    cluster_assignments = {item['conversation_id']: item['cluster']
                          for item in results['cluster_assignments']}

    print(f"Loaded cluster assignments for {len(cluster_assignments):,} conversations")
    print(f"Number of clusters: {results['metadata']['n_clusters']}\n")

    # Load raw data
    raw_file = project_dir / "data" / "raw" / "2023_conversations_two-participants.parquet"

    if not raw_file.exists():
        print(f"ERROR: {raw_file} not found")
        return 1

    print(f"Loading raw data from {raw_file.name}...")
    df = pd.read_parquet(raw_file)
    print(f"Loaded {len(df):,} messages\n")

    # Add cluster assignments
    df['cluster'] = df['conversation_id'].map(cluster_assignments)

    # Filter to conversations that were clustered
    df = df[df['cluster'].notna()]

    print("="*80)
    print("STEP 2: CLUSTER OUTCOMES COMPARISON")
    print("="*80 + "\n")

    # Compute outcomes by cluster
    outcomes = []

    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]

        # Conversation-level metrics
        conv_groups = cluster_df.groupby('conversation_id')

        outcome = {
            'cluster': int(cluster_id),
            'n_conversations': cluster_df['conversation_id'].nunique(),
            'percentage': cluster_df['conversation_id'].nunique() / len(cluster_assignments) * 100,

            # Message counts
            'mean_messages': float(conv_groups.size().mean()),
            'median_messages': float(conv_groups.size().median()),
            'std_messages': float(conv_groups.size().std()),

            # Conversation duration (if timestamp available)
            'mean_duration_minutes': None,
            'median_duration_minutes': None,
        }

        # Try to compute duration
        if 'timestamp' in cluster_df.columns:
            cluster_df['timestamp'] = pd.to_datetime(cluster_df['timestamp'])
            durations = conv_groups['timestamp'].apply(lambda x: (x.max() - x.min()).total_seconds() / 60)
            outcome['mean_duration_minutes'] = float(durations.mean())
            outcome['median_duration_minutes'] = float(durations.median())

        outcomes.append(outcome)

    # Display outcomes
    print("Cluster Outcomes:\n")
    for outcome in outcomes:
        print(f"Cluster {outcome['cluster']} ({outcome['n_conversations']:,} conversations, {outcome['percentage']:.1f}%):")
        print(f"  Messages: mean={outcome['mean_messages']:.1f}, median={outcome['median_messages']:.1f}, std={outcome['std_messages']:.1f}")
        if outcome['mean_duration_minutes']:
            print(f"  Duration: mean={outcome['mean_duration_minutes']:.1f} min, median={outcome['median_duration_minutes']:.1f} min")
        print()

    # Statistical tests
    print("Statistical Comparison:")
    from scipy.stats import mannwhitneyu

    cluster_0_msgs = df[df['cluster'] == 0].groupby('conversation_id').size()
    cluster_1_msgs = df[df['cluster'] == 1].groupby('conversation_id').size()

    u_stat, p_value = mannwhitneyu(cluster_0_msgs, cluster_1_msgs)
    print(f"  Message count difference: U={u_stat:.0f}, p={p_value:.4f}")

    if outcomes[0]['mean_duration_minutes']:
        cluster_0_dur = df[df['cluster'] == 0].groupby('conversation_id')['timestamp'].apply(
            lambda x: (x.max() - x.min()).total_seconds() / 60)
        cluster_1_dur = df[df['cluster'] == 1].groupby('conversation_id')['timestamp'].apply(
            lambda x: (x.max() - x.min()).total_seconds() / 60)

        u_stat_dur, p_value_dur = mannwhitneyu(cluster_0_dur, cluster_1_dur)
        print(f"  Duration difference: U={u_stat_dur:.0f}, p={p_value_dur:.4f}")

    print()

    print("="*80)
    print("STEP 3: SAMPLE CONVERSATIONS FROM EACH CLUSTER")
    print("="*80 + "\n")

    # Sample 5 conversations from each cluster
    n_samples = 5

    for cluster_id in sorted(df['cluster'].unique()):
        cluster_convs = df[df['cluster'] == cluster_id]['conversation_id'].unique()

        # Sample randomly
        np.random.seed(42)
        sample_convs = np.random.choice(cluster_convs, size=min(n_samples, len(cluster_convs)), replace=False)

        print(f"CLUSTER {cluster_id} SAMPLE CONVERSATIONS:")
        print(f"(Randomly selected {len(sample_convs)} out of {len(cluster_convs):,} conversations)\n")

        for i, conv_id in enumerate(sample_convs, 1):
            conv_df = df[df['conversation_id'] == conv_id].sort_values('timestamp')

            # Conversation metadata
            n_msgs = len(conv_df)
            duration = (conv_df['timestamp'].max() - conv_df['timestamp'].min()).total_seconds() / 60

            print(f"  [{i}] Conversation {conv_id}")
            print(f"      {n_msgs} messages over {duration:.1f} minutes")
            print(f"      Messages:")

            # Show first 5 messages
            for j, (_, msg) in enumerate(conv_df.head(5).iterrows(), 1):
                role = "Student" if msg.get('student') == 1 else "Tutor" if msg.get('helper') == 1 else "Unknown"
                content = msg.get('content', msg.get('text', '[no content]'))

                # Truncate long messages
                if len(content) > 80:
                    content = content[:80] + "..."

                print(f"        {j}. [{role}] {content}")

            if n_msgs > 5:
                print(f"        ... ({n_msgs - 5} more messages)")

            print()

        print("-"*80 + "\n")

    # Save characterization results
    output_file = project_dir / "results" / "2023_conversations_two-participants_cluster_characterization.json"

    characterization = {
        'cluster_outcomes': outcomes,
        'sample_conversations': []
    }

    # Add sampled conversations to output
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_convs = df[df['cluster'] == cluster_id]['conversation_id'].unique()
        np.random.seed(42)
        sample_convs = np.random.choice(cluster_convs, size=min(n_samples, len(cluster_convs)), replace=False)

        for conv_id in sample_convs:
            conv_df = df[df['conversation_id'] == conv_id].sort_values('timestamp')

            characterization['sample_conversations'].append({
                'conversation_id': float(conv_id),
                'cluster': int(cluster_id),
                'n_messages': len(conv_df),
                'duration_minutes': float((conv_df['timestamp'].max() - conv_df['timestamp'].min()).total_seconds() / 60),
                'messages': [
                    {
                        'role': 'student' if msg.get('student') == 1 else 'tutor' if msg.get('helper') == 1 else 'unknown',
                        'content': msg.get('content', msg.get('text', '[no content]')),
                        'timestamp': str(msg['timestamp'])
                    }
                    for _, msg in conv_df.iterrows()
                ]
            })

    with open(output_file, 'w') as f:
        json.dump(characterization, f, indent=2)

    print(f"Saved cluster characterization to {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
