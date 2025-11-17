#!/usr/bin/env python3
"""
Create Publication-Quality Figures

Generates visualizations from clustering results:
1. Optimal k determination (5 metrics)
2. UMAP cluster visualization
3. Feature comparison by cluster
4. Summary table

Input: results/clustering_results.json, data/processed/*_features.parquet
Output: figures/*.png, figures/*.pdf

Usage:
    python scripts/03_create_figures.py
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Try to import UMAP, but make it optional
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Plot style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11


def plot_optimal_k(optimal_k_data, output_dir):
    """
    6-panel plot showing all 5 metrics for determining optimal k.
    """
    logger.info("Creating Figure 1: Optimal k determination...")

    df = pd.DataFrame(optimal_k_data)
    k_values = df['k'].values

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Elbow (Inertia)
    ax = axes[0, 0]
    ax.plot(k_values, df['inertia'], 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia (WCSS)')
    ax.set_title('Elbow Method', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Silhouette
    ax = axes[0, 1]
    ax.plot(k_values, df['silhouette'], 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score (higher is better)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    best_k = k_values[np.argmax(df['silhouette'])]
    ax.axvline(x=best_k, color='green', linestyle='--', alpha=0.5,
               label=f'k={best_k}')
    ax.legend()

    # Calinski-Harabasz
    ax = axes[0, 2]
    ax.plot(k_values, df['calinski_harabasz'], 'mo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Calinski-Harabasz Index')
    ax.set_title('Calinski-Harabasz (higher is better)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Davies-Bouldin
    ax = axes[1, 0]
    ax.plot(k_values, df['davies_bouldin'], 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Davies-Bouldin Index')
    ax.set_title('Davies-Bouldin (lower is better)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Gap Statistic
    ax = axes[1, 1]
    ax.errorbar(k_values, df['gap_statistic'], yerr=df['gap_error'],
                fmt='co-', linewidth=2, markersize=8, capsize=5)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Gap Statistic')
    ax.set_title('Gap Statistic', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Summary
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = "OPTIMAL k SUMMARY\n" + "="*35 + "\n\n"
    summary_text += f"Silhouette (max): k = {k_values[np.argmax(df['silhouette'])]}\n"
    summary_text += f"Calinski-H (max): k = {k_values[np.argmax(df['calinski_harabasz'])]}\n"
    summary_text += f"Davies-B (min):   k = {k_values[np.argmin(df['davies_bouldin'])]}\n"
    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Optimal Number of Clusters Determination',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / 'figure1_optimal_k.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure1_optimal_k.pdf', bbox_inches='tight')
    logger.info(f"Saved: {output_file.name}")
    plt.close()


def plot_umap_clusters(features_df, labels, output_dir):
    """
    2D UMAP projection colored by cluster.
    """
    if not HAS_UMAP:
        logger.warning("Skipping UMAP (umap-learn not installed)")
        return

    logger.info("Creating Figure 2: UMAP cluster visualization...")

    feature_cols = ['burst_coefficient', 'cluster_density',
                    'response_acceleration', 'memory_coefficient',
                    'timing_consistency']

    X = features_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # UMAP projection
    reducer = umap.UMAP(n_components=2, random_state=42,
                       n_neighbors=30, min_dist=0.3, spread=1.0)
    embedding = reducer.fit_transform(X_scaled)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use distinct colors for each cluster
    unique_labels = np.unique(labels)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            # Outliers in gray
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      c='lightgray', s=15, alpha=0.3,
                      edgecolors='none', label=f'Outliers (n={mask.sum()})')
        else:
            # Regular clusters in color
            color = plt.cm.tab10(label % 10)
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      c=[color], s=25, alpha=0.7,
                      edgecolors='black', linewidth=0.5,
                      label=f'Cluster {label} (n={mask.sum()})')

    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title('Temporal Clusters (UMAP Projection)',
                 fontsize=14, fontweight='bold')

    ax.legend(markerscale=2, loc='best')
    plt.tight_layout()

    output_file = output_dir / 'figure2_umap_clusters.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure2_umap_clusters.pdf', bbox_inches='tight')
    logger.info(f"Saved: {output_file.name}")
    plt.close()


def plot_feature_comparison(cluster_stats, output_dir):
    """
    Bar plots showing mean feature values per cluster.
    """
    logger.info("Creating Figure 3: Feature comparison by cluster...")

    features = ['burst_coefficient', 'cluster_density', 'response_acceleration',
                'memory_coefficient', 'timing_consistency']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]

        cluster_ids = []
        values = []
        sizes = []

        for cluster_key in sorted(cluster_stats.keys(),
                                 key=lambda x: int(x.split('_')[1])):
            stats = cluster_stats[cluster_key]
            cluster_ids.append(f"C{stats['cluster_id']}")
            values.append(stats['mean_features'][feature])
            sizes.append(stats['size'])

        value_range = max(values) - min(values)
        if value_range > 0:
            colors = plt.cm.RdYlGn([(v - min(values)) / value_range
                                   for v in values])
        else:
            colors = ['gray'] * len(values)

        bars = ax.barh(cluster_ids, values, color=colors, edgecolor='black')

        ax.set_xlabel(f'{feature.replace("_", " ").title()}')
        ax.set_title(feature.replace('_', ' ').title(), fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

        # Add size annotations
        for i, (bar, size) in enumerate(zip(bars, sizes)):
            ax.text(0.02, i, f"n={size:,}", va='center',
                   fontsize=8, color='gray')

    fig.delaxes(axes[5])

    plt.suptitle('Feature Comparison Across Clusters',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / 'figure3_feature_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure3_feature_comparison.pdf', bbox_inches='tight')
    logger.info(f"Saved: {output_file.name}")
    plt.close()


def create_summary_table(cluster_stats, effect_sizes, output_dir):
    """
    Publication-ready summary table.
    """
    logger.info("Creating Figure 4: Summary table...")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    for cluster_key in sorted(cluster_stats.keys(),
                             key=lambda x: int(x.split('_')[1])):
        stats = cluster_stats[cluster_key]
        row = [
            f"C{stats['cluster_id']}",
            f"{stats['size']:,}",
            f"{stats['percentage']:.1f}%",
            f"{stats['mean_features']['burst_coefficient']:+.2f}",
            f"{stats['mean_features']['cluster_density']:.2f}",
            f"{stats['mean_features']['response_acceleration']:+.1f}",
        ]
        table_data.append(row)

    columns = ['Cluster', 'Size', '% Total', 'BC', 'CD', 'RA']

    table = ax.table(cellText=table_data, colLabels=columns,
                    cellLoc='center', loc='center',
                    colColours=['lightgray']*len(columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Cluster Summary Table', fontsize=14,
                 fontweight='bold', pad=20)

    output_file = output_dir / 'figure4_summary_table.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure4_summary_table.pdf', bbox_inches='tight')
    logger.info(f"Saved: {output_file.name}")
    plt.close()


def plot_method_comparison(method_comparison, optimal_k, output_dir):
    """
    Create a comparison figure showing silhouette scores for different clustering methods.
    """
    logger.info("Creating Method Comparison figure...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Bar chart comparing methods
    methods = []
    silhouettes = []
    n_clusters = []
    colors_map = {'kmeans': '#4472C4', 'hierarchical': '#ED7D31', 'dbscan': '#70AD47'}
    colors = []

    if method_comparison.get('kmeans'):
        methods.append('K-means')
        silhouettes.append(method_comparison['kmeans']['silhouette'])
        n_clusters.append(method_comparison['kmeans']['n_clusters'])
        colors.append(colors_map['kmeans'])

    if method_comparison.get('hierarchical'):
        methods.append('Hierarchical')
        silhouettes.append(method_comparison['hierarchical']['silhouette'])
        n_clusters.append(method_comparison['hierarchical']['n_clusters'])
        colors.append(colors_map['hierarchical'])

    if method_comparison.get('dbscan') and method_comparison['dbscan']:
        methods.append('DBSCAN')
        silhouettes.append(method_comparison['dbscan']['silhouette'])
        n_clusters.append(int(method_comparison['dbscan']['n_clusters']))
        colors.append(colors_map['dbscan'])

    bars = ax1.barh(methods, silhouettes, color=colors, edgecolor='black', linewidth=1.5)

    # Highlight the recommended method
    recommended = method_comparison.get('recommended', 'kmeans')
    for i, method in enumerate(methods):
        if method.lower() == recommended:
            bars[i].set_linewidth(3)
            bars[i].set_edgecolor('red')

    ax1.set_xlabel('Silhouette Score', fontsize=13, fontweight='bold')
    ax1.set_title('Clustering Method Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, max(silhouettes) * 1.15)

    # Add value labels and cluster counts
    for i, (sil, k) in enumerate(zip(silhouettes, n_clusters)):
        ax1.text(sil + 0.005, i, f'{sil:.3f}\n(k={k})',
                va='center', fontsize=10, fontweight='bold')

    ax1.grid(axis='x', alpha=0.3)

    # Panel 2: Summary table
    ax2.axis('tight')
    ax2.axis('off')

    table_data = []
    for method, sil, k in zip(methods, silhouettes, n_clusters):
        row = [method, f'{k}', f'{sil:.4f}']
        table_data.append(row)

    # Add recommendation marker
    rec_idx = [m.lower() for m in methods].index(recommended)
    table_data[rec_idx][0] = f'{table_data[rec_idx][0]} ★'

    columns = ['Method', 'Clusters', 'Silhouette']
    table = ax2.table(cellText=table_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colColours=['lightgray']*3)

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

    # Highlight recommended row
    for i in range(len(columns)):
        table[(rec_idx + 1, i)].set_facecolor('#FFE699')
        table[(rec_idx + 1, i)].set_text_props(weight='bold')

    ax2.set_title(f'Selected Method: {recommended.upper()} (★)\n'
                  f'Optimal k = {optimal_k}',
                  fontsize=13, fontweight='bold', pad=20)

    plt.suptitle('Clustering Method Selection', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_file = output_dir / 'figure0_method_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure0_method_comparison.pdf', bbox_inches='tight')
    logger.info(f"Saved: {output_file.name}")
    plt.close()


def main():
    """Main execution."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    processed_dir = project_dir / "data" / "processed"
    results_dir = project_dir / "results"

    # Find the most recent features file
    feature_files = list(processed_dir.glob("*_features.parquet"))
    if not feature_files:
        logger.error(f"No feature files found in {processed_dir}")
        logger.error("Run 01_extract_features.py first")
        return 1

    features_file = max(feature_files, key=lambda p: p.stat().st_mtime)
    dataset_name = features_file.stem.replace('_features', '')
    logger.info(f"Using features from: {features_file.name}")

    # Look for corresponding results file
    results_file = results_dir / f"{dataset_name}_clustering_results.json"
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        logger.error("Run 02_cluster_analysis.py first")
        return 1

    # Create dataset-specific output directory
    output_dir = project_dir / "figures" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Load features for UMAP
    features_df = pd.read_parquet(features_file)

    # Extract cluster labels
    cluster_assignments = {item['conversation_id']: item['cluster']
                          for item in results['cluster_assignments']}
    labels = features_df['conversation_id'].map(cluster_assignments).values

    # Generate figures
    plot_method_comparison(results['method_comparison'],
                          results['metadata']['n_clusters'], output_dir)
    plot_optimal_k(results['optimal_k_analysis'], output_dir)
    plot_umap_clusters(features_df, labels, output_dir)
    plot_feature_comparison(results['cluster_statistics'], output_dir)
    create_summary_table(results['cluster_statistics'],
                        results['effect_sizes'], output_dir)

    logger.info(f"\nAll figures saved to {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
