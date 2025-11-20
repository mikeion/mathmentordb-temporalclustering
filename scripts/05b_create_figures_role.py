#!/usr/bin/env python3
"""
Create visualizations for 15D role-specific clustering results.

This script generates figures that demonstrate the key finding:
student temporal patterns drive clustering (η²=0.608), not tutor patterns (η²<0.01).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Try to import UMAP, but make it optional
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed. UMAP visualization will be skipped.")

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

def load_results(dataset_name='2023_conversations_two-participants'):
    """Load clustering results and features"""
    results_file = Path('results') / f'{dataset_name}_role_clustering_results.json'
    features_file = Path('data') / 'processed' / f'{dataset_name}_role_features.parquet'

    with open(results_file) as f:
        results = json.load(f)

    features_df = pd.read_parquet(features_file)

    # Add cluster assignments
    cluster_assignments = {item['conversation_id']: item['cluster']
                          for item in results['cluster_assignments']}
    features_df['cluster'] = features_df['conversation_id'].map(cluster_assignments)

    return results, features_df

def create_effect_size_comparison(results, output_dir):
    """
    Figure 1: Effect size comparison showing student vs tutor vs conversation features

    This is the KEY figure demonstrating that students drive temporal patterns.
    """
    effect_sizes = results['effect_sizes']

    # Organize features by role
    student_features = {k: v for k, v in effect_sizes.items() if '_student' in k}
    tutor_features = {k: v for k, v in effect_sizes.items() if '_tutor' in k}
    conv_features = {k: v for k, v in effect_sizes.items()
                    if '_student' not in k and '_tutor' not in k}

    # Prepare data for plotting
    data = []
    for feature, stats in student_features.items():
        if not np.isnan(stats.get('eta_squared', float('nan'))):
            data.append({
                'feature': feature.replace('_student', ''),
                'eta_squared': stats['eta_squared'],
                'role': 'Student'
            })

    for feature, stats in tutor_features.items():
        if not np.isnan(stats.get('eta_squared', float('nan'))):
            data.append({
                'feature': feature.replace('_tutor', ''),
                'eta_squared': stats['eta_squared'],
                'role': 'Tutor'
            })

    for feature, stats in conv_features.items():
        if not np.isnan(stats.get('eta_squared', float('nan'))):
            data.append({
                'feature': feature,
                'eta_squared': stats['eta_squared'],
                'role': 'Conversation'
            })

    df = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group bars by feature
    features_ordered = ['burst_coefficient', 'cluster_density', 'response_acceleration',
                       'memory_coefficient', 'timing_consistency']

    x = np.arange(len(features_ordered))
    width = 0.25

    for i, role in enumerate(['Student', 'Tutor', 'Conversation']):
        role_data = df[df['role'] == role]
        values = [role_data[role_data['feature'] == f]['eta_squared'].values[0]
                 if len(role_data[role_data['feature'] == f]) > 0 else 0
                 for f in features_ordered]

        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=role, alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # Only label meaningful values
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Temporal Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Effect Size (η²)', fontsize=12, fontweight='bold')
    ax.set_title('Effect Sizes by Role: Students Drive Temporal Patterns',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', ' ').title() for f in features_ordered],
                       rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add interpretation box
    textstr = 'Key Finding:\n• Student features: η² > 0.4 (strong effect)\n• Tutor features: η² < 0.01 (negligible)\n• Students drive temporal patterns!'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'figure1_effect_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure1_effect_size_comparison.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: figure1_effect_size_comparison (PNG + PDF)")

def create_feature_profiles(results, features_df, output_dir):
    """
    Figure 2: Feature profiles by cluster (heatmap)

    Shows mean feature values for each cluster across all 15 dimensions.
    """
    # Get feature columns (exclude metadata)
    feature_cols = [col for col in features_df.columns
                   if col not in ['conversation_id', 'cluster']]

    # Compute mean feature values per cluster
    cluster_means = features_df.groupby('cluster')[feature_cols].mean()

    # Organize features by category for better visualization
    conv_features = [f for f in feature_cols if '_student' not in f and '_tutor' not in f]
    tutor_features = [f for f in feature_cols if '_tutor' in f]
    student_features = [f for f in feature_cols if '_student' in f]

    ordered_features = conv_features + tutor_features + student_features
    cluster_means = cluster_means[ordered_features]

    # Normalize each feature to [0, 1] for better color comparison
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(cluster_means_norm.T,
               annot=cluster_means.T,  # Show actual values
               fmt='.3f',
               cmap='RdYlGn',
               center=0.5,
               cbar_kws={'label': 'Normalized Value'},
               xticklabels=[f'Cluster {i}' for i in cluster_means.index],
               yticklabels=[f.replace('_', ' ').title() for f in ordered_features],
               ax=ax)

    ax.set_title('Feature Profiles by Cluster', fontsize=14, fontweight='bold', pad=20)

    # Add category labels on right side
    n_conv = len(conv_features)
    n_tutor = len(tutor_features)
    n_student = len(student_features)

    ax.text(1.15, n_conv/2, 'Conversation', transform=ax.get_yaxis_transform(),
           ha='left', va='center', fontsize=10, fontweight='bold', rotation=-90)
    ax.text(1.15, n_conv + n_tutor/2, 'Tutor', transform=ax.get_yaxis_transform(),
           ha='left', va='center', fontsize=10, fontweight='bold', rotation=-90)
    ax.text(1.15, n_conv + n_tutor + n_student/2, 'Student', transform=ax.get_yaxis_transform(),
           ha='left', va='center', fontsize=10, fontweight='bold', rotation=-90)

    plt.tight_layout()

    # Save
    plt.savefig(output_dir / 'figure2_feature_profiles.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure2_feature_profiles.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: figure2_feature_profiles (PNG + PDF)")

def create_umap_visualization(features_df, output_dir):
    """
    Figure 2b: UMAP projection showing cluster separation

    Projects 15D role-specific features into 2D using UMAP to visualize
    how well the two clusters separate in feature space.
    """
    if not HAS_UMAP:
        print("⚠ Skipping UMAP visualization (umap-learn not installed)")
        return

    # Get feature columns (exclude metadata)
    feature_cols = [col for col in features_df.columns
                   if col not in ['conversation_id', 'cluster']]

    X = features_df[feature_cols].values
    clusters = features_df['cluster'].values

    # UMAP projection
    print("  Computing UMAP projection (this may take a minute)...")
    reducer = umap.UMAP(n_components=2, random_state=42,
                       n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get cluster labels
    unique_clusters = sorted(np.unique(clusters))
    colors = sns.color_palette("Set2", len(unique_clusters))

    # Cluster names matching our findings
    cluster_names = {
        0: "Natural Flow",
        1: "Methodical Worker"
    }

    # Plot each cluster
    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        n_points = mask.sum()
        pct = 100 * n_points / len(clusters)

        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  c=[colors[i]], label=f'{cluster_name} (n={n_points:,}, {pct:.1f}%)',
                  alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

    ax.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title('15D Role-Specific Features: UMAP Projection by Cluster',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3)

    # Add interpretation box
    textstr = 'Cluster separation based on:\n• Student temporal patterns (η²=0.608)\n• 15D role-decomposed features'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save
    plt.savefig(output_dir / 'figure2b_umap_projection.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure2b_umap_projection.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: figure2b_umap_projection (PNG + PDF)")

def create_cluster_size_comparison(results, features_df, output_dir):
    """
    Figure 3: Cluster size and composition

    Shows distribution of conversations across clusters and basic statistics.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Cluster sizes (pie chart)
    cluster_counts = features_df['cluster'].value_counts().sort_index()
    colors = sns.color_palette("Set2", len(cluster_counts))

    wedges, texts, autotexts = ax1.pie(cluster_counts.values,
                                        labels=[f'Cluster {i}' for i in cluster_counts.index],
                                        autopct='%1.1f%%',
                                        colors=colors,
                                        startangle=90)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    ax1.set_title('Cluster Distribution', fontsize=14, fontweight='bold')

    # Right panel: Silhouette scores and cluster statistics
    method = results['metadata']['method']
    silhouette = results['method_comparison'][method]['silhouette']
    optimal_k = results['metadata'].get('k', len(cluster_counts))

    stats_text = f"""Clustering Quality

Method: {method.capitalize()}
Optimal k: {optimal_k}
Silhouette: {silhouette:.3f}

Cluster Sizes:
"""

    for cluster_id, count in cluster_counts.items():
        pct = 100 * count / cluster_counts.sum()
        stats_text += f"  Cluster {cluster_id}: {count:,} ({pct:.1f}%)\n"

    # Add effect size highlight
    effect_sizes = results['effect_sizes']
    top_student_feature = max(
        [(k, v['eta_squared']) for k, v in effect_sizes.items()
         if '_student' in k and not np.isnan(v.get('eta_squared', float('nan')))],
        key=lambda x: x[1]
    )

    stats_text += f"\nTop Discriminative Feature:\n  {top_student_feature[0]}\n  η² = {top_student_feature[1]:.3f}"

    ax2.text(0.1, 0.5, stats_text,
            transform=ax2.transAxes,
            fontsize=11,
            verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax2.axis('off')

    plt.suptitle('Cluster Overview: 15D Role-Specific Clustering',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    plt.savefig(output_dir / 'figure3_cluster_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure3_cluster_overview.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: figure3_cluster_overview (PNG + PDF)")

def create_role_decomposition_diagram(results, output_dir):
    """
    Figure 4: Conceptual diagram showing role decomposition

    Illustrates how 5D conversation features are decomposed into 15D role-specific features.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Title
    fig.suptitle('Role Decomposition: 5D → 15D Feature Space',
                fontsize=16, fontweight='bold', y=0.98)

    # Draw boxes for each level
    # Level 1: Conversation-level (5D)
    conv_box = plt.Rectangle((0.2, 0.7), 0.6, 0.15,
                             fill=True, facecolor='lightblue',
                             edgecolor='black', linewidth=2)
    ax.add_patch(conv_box)
    ax.text(0.5, 0.775, '5D Conversation Features',
           ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.74, 'BC, CD, RA, MC, TC',
           ha='center', va='center', fontsize=10, style='italic')

    # Arrow down
    ax.arrow(0.5, 0.7, 0, -0.08, head_width=0.03, head_length=0.02,
            fc='black', ec='black', linewidth=2)

    # Level 2: Role-specific (15D) - Three boxes
    # Tutor box (left)
    tutor_box = plt.Rectangle((0.05, 0.35), 0.25, 0.2,
                              fill=True, facecolor='lightcoral',
                              edgecolor='black', linewidth=2)
    ax.add_patch(tutor_box)
    ax.text(0.175, 0.5, '5D Tutor Features',
           ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(0.175, 0.45, 'BC_tutor\nCD_tutor\nRA_tutor\nMC_tutor\nTC_tutor',
           ha='center', va='center', fontsize=8)
    ax.text(0.175, 0.37, 'η² < 0.01',
           ha='center', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Conversation box (middle)
    conv_box2 = plt.Rectangle((0.375, 0.35), 0.25, 0.2,
                              fill=True, facecolor='lightblue',
                              edgecolor='black', linewidth=2)
    ax.add_patch(conv_box2)
    ax.text(0.5, 0.5, '5D Conv Features',
           ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.45, 'BC\nCD\nRA\nMC\nTC',
           ha='center', va='center', fontsize=8)
    ax.text(0.5, 0.37, 'η² ≈ 0.07',
           ha='center', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Student box (right)
    student_box = plt.Rectangle((0.7, 0.35), 0.25, 0.2,
                                fill=True, facecolor='lightgreen',
                                edgecolor='black', linewidth=2)
    ax.add_patch(student_box)
    ax.text(0.825, 0.5, '5D Student Features',
           ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(0.825, 0.45, 'BC_student\nCD_student\nRA_student\nMC_student\nTC_student',
           ha='center', va='center', fontsize=8)
    ax.text(0.825, 0.37, 'η² > 0.4',
           ha='center', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Arrows from conversation to role-specific
    ax.arrow(0.35, 0.6, -0.15, -0.05, head_width=0.02, head_length=0.015,
            fc='gray', ec='gray', linewidth=1.5, linestyle='--')
    ax.arrow(0.5, 0.6, 0, -0.05, head_width=0.02, head_length=0.015,
            fc='gray', ec='gray', linewidth=1.5, linestyle='--')
    ax.arrow(0.65, 0.6, 0.15, -0.05, head_width=0.02, head_length=0.015,
            fc='gray', ec='gray', linewidth=1.5, linestyle='--')

    # Key finding box at bottom
    finding_text = """Key Finding: Students Drive Temporal Patterns

• Student features explain 60.8% of cluster variance (η² = 0.608)
• Tutor features explain < 1% of variance (η² < 0.01)
• Conversation-level features (5D) wash out the student signal by averaging

Implication: Temporal archetypes reflect student engagement strategies,
not tutor pacing. Tutors adapt to student rhythms."""

    finding_box = plt.Rectangle((0.1, 0.05), 0.8, 0.22,
                               fill=True, facecolor='wheat',
                               edgecolor='black', linewidth=2)
    ax.add_patch(finding_box)
    ax.text(0.5, 0.16, finding_text,
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    # Save
    plt.savefig(output_dir / 'figure4_role_decomposition.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure4_role_decomposition.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: figure4_role_decomposition (PNG + PDF)")

def main():
    """Generate all role-specific figures"""
    # Configuration
    dataset_name = '2023_conversations_two-participants'
    output_dir = Path('figures') / dataset_name / 'role_15d'

    print(f"\n{'='*60}")
    print(f"CREATING 15D ROLE-SPECIFIC FIGURES")
    print(f"{'='*60}\n")
    print(f"Dataset: {dataset_name}")
    print(f"Output directory: {output_dir}\n")

    # Load data
    print("Loading results and features...")
    results, features_df = load_results(dataset_name)
    print(f"✓ Loaded {len(features_df):,} conversations with cluster assignments\n")

    # Create figures
    print("Generating figures...")
    print("-" * 60)

    create_effect_size_comparison(results, output_dir)
    create_feature_profiles(results, features_df, output_dir)
    create_umap_visualization(features_df, output_dir)
    create_cluster_size_comparison(results, features_df, output_dir)
    create_role_decomposition_diagram(results, output_dir)

    print("-" * 60)
    print(f"\n✓ All figures saved to: {output_dir}/")
    print("\nFigures created:")
    print("  • figure1_effect_size_comparison: Student vs Tutor vs Conversation η²")
    print("  • figure2_feature_profiles: Mean feature values per cluster (heatmap)")
    print("  • figure2b_umap_projection: 2D UMAP projection showing cluster separation")
    print("  • figure3_cluster_overview: Cluster sizes and statistics")
    print("  • figure4_role_decomposition: Conceptual diagram of 5D→15D decomposition")
    print(f"\n{'='*60}\n")

if __name__ == '__main__':
    main()
