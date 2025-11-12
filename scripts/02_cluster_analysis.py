#!/usr/bin/env python3
"""
Temporal Clustering Analysis

This script:
1. Determines optimal number of clusters (k=2-15) using 5 metrics
2. Compares DBSCAN vs hierarchical vs K-means
3. Applies final clustering with best method
4. Computes cluster statistics and effect sizes
5. Saves results for visualization

Input: data/processed/*_features.parquet
Output: results/clustering_results.json

Usage:
    python scripts/02_cluster_analysis.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist
from scipy.stats import f_oneway
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: OPTIMAL K DETERMINATION
# ==============================================================================

def compute_gap_statistic(X, k_range, n_refs=10, random_state=42):
    """
    Gap statistic: Compare within-cluster dispersion to random data.
    """
    gaps, errors = [], []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)

        # Actual within-cluster dispersion
        Wk = sum(np.min(cdist(X[labels == i], [kmeans.cluster_centers_[i]], 'euclidean')**2)
                 for i in range(k))

        # Reference datasets (uniform random)
        ref_dispersions = []
        for _ in range(n_refs):
            random_data = np.random.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
            kmeans_ref = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels_ref = kmeans_ref.fit_predict(random_data)
            Wk_ref = sum(np.min(cdist(random_data[labels_ref == i],
                                     [kmeans_ref.cluster_centers_[i]], 'euclidean')**2)
                        for i in range(k))
            ref_dispersions.append(np.log(Wk_ref))

        gap = np.mean(ref_dispersions) - np.log(Wk)
        error = np.std(ref_dispersions) * np.sqrt(1 + 1/n_refs)

        gaps.append(gap)
        errors.append(error)

    return gaps, errors


def determine_optimal_k(X_scaled, k_range=range(2, 16)):
    """
    Use 5 different metrics to determine optimal k.
    """
    logger.info("\n" + "="*80)
    logger.info("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
    logger.info("="*80)

    results = {'k': [], 'inertia': [], 'silhouette': [], 'calinski_harabasz': [],
               'davies_bouldin': [], 'gap_statistic': [], 'gap_error': []}

    logger.info(f"\nTesting k={min(k_range)} to k={max(k_range)-1}...")

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        results['k'].append(k)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(X_scaled, labels))
        results['calinski_harabasz'].append(calinski_harabasz_score(X_scaled, labels))
        results['davies_bouldin'].append(davies_bouldin_score(X_scaled, labels))

    # Gap statistic
    logger.info("Computing gap statistic (this may take a minute)...")
    gaps, gap_errors = compute_gap_statistic(X_scaled, k_range, n_refs=10)
    results['gap_statistic'] = gaps
    results['gap_error'] = gap_errors

    # Determine optimal k from each method
    best_sil_k = k_range[np.argmax(results['silhouette'])]
    best_ch_k = k_range[np.argmax(results['calinski_harabasz'])]
    best_db_k = k_range[np.argmin(results['davies_bouldin'])]

    # Gap statistic: first k where gap[k] >= gap[k+1] - error[k+1]
    optimal_gap_k = None
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i+1] - gap_errors[i+1]:
            optimal_gap_k = k_range[i]
            break

    logger.info("\nOptimal k by method:")
    logger.info(f"  Silhouette score: k = {best_sil_k}")
    logger.info(f"  Calinski-Harabasz: k = {best_ch_k}")
    logger.info(f"  Davies-Bouldin: k = {best_db_k}")
    if optimal_gap_k:
        logger.info(f"  Gap statistic: k = {optimal_gap_k}")

    # Consensus
    votes = [best_sil_k, best_ch_k, best_db_k]
    if optimal_gap_k:
        votes.append(optimal_gap_k)

    from collections import Counter
    consensus_k = Counter(votes).most_common(1)[0][0]

    logger.info(f"\nCONSENSUS: k = {consensus_k} ({Counter(votes)[consensus_k]}/{len(votes)} methods agree)")

    return pd.DataFrame(results), consensus_k


# ==============================================================================
# SECTION 2: METHOD COMPARISON
# ==============================================================================

def compare_clustering_methods(X_scaled, optimal_k=6):
    """
    Compare DBSCAN vs hierarchical vs K-means.
    """
    logger.info("\n" + "="*80)
    logger.info("COMPARING CLUSTERING METHODS")
    logger.info("="*80)

    comparison = {}

    # Test DBSCAN
    logger.info("\n1. DBSCAN parameter search...")
    logger.info("   Testing eps=[0.3-0.8] × min_samples=[30-100]")

    dbscan_results = []
    for eps in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for min_samples in [30, 50, 70, 100]:
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(X_scaled)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_outliers = np.sum(labels == -1)

            silhouette = None
            if n_clusters > 1:
                mask = labels != -1
                if mask.sum() > n_clusters:
                    try:
                        silhouette = silhouette_score(X_scaled[mask], labels[mask])
                    except:
                        pass

            dbscan_results.append({
                'eps': eps, 'min_samples': min_samples,
                'n_clusters': n_clusters, 'n_outliers': n_outliers,
                'outlier_pct': n_outliers / len(labels) * 100,
                'silhouette': silhouette
            })

    dbscan_df = pd.DataFrame(dbscan_results)
    dbscan_good = dbscan_df[
        (dbscan_df['n_clusters'] >= optimal_k - 1) &
        (dbscan_df['n_clusters'] <= optimal_k + 1) &
        (dbscan_df['outlier_pct'] < 30)
    ]

    if len(dbscan_good) > 0:
        best_dbscan = dbscan_good.nlargest(1, 'silhouette').iloc[0]
        logger.info(f"   Best config: eps={best_dbscan['eps']}, min_samples={best_dbscan['min_samples']}")
        logger.info(f"   Result: {best_dbscan['n_clusters']} clusters, silhouette={best_dbscan['silhouette']:.3f}")
        comparison['dbscan'] = best_dbscan.to_dict()
    else:
        logger.info(f"   No viable DBSCAN configs found (data too dense)")
        comparison['dbscan'] = None

    # Test Hierarchical
    logger.info("\n2. Hierarchical clustering (Ward linkage)...")
    hier = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    hier_labels = hier.fit_predict(X_scaled)
    hier_sil = silhouette_score(X_scaled, hier_labels)

    logger.info(f"   Silhouette: {hier_sil:.3f}")
    comparison['hierarchical'] = {
        'n_clusters': optimal_k,
        'linkage': 'ward',
        'silhouette': float(hier_sil)
    }

    # Test K-means
    logger.info("\n3. K-means...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_sil = silhouette_score(X_scaled, kmeans_labels)

    logger.info(f"   Silhouette: {kmeans_sil:.3f}")
    comparison['kmeans'] = {
        'n_clusters': optimal_k,
        'silhouette': float(kmeans_sil),
        'inertia': float(kmeans.inertia_)
    }

    # Recommend best method
    if comparison['dbscan'] and comparison['dbscan']['silhouette'] > max(hier_sil, kmeans_sil):
        best_method = 'dbscan'
    elif hier_sil > kmeans_sil + 0.02:
        best_method = 'hierarchical'
    else:
        best_method = 'kmeans'

    logger.info(f"\nRECOMMENDATION: Use {best_method}")
    comparison['recommended'] = best_method

    return comparison


# ==============================================================================
# SECTION 3: FINAL CLUSTERING
# ==============================================================================

def apply_final_clustering(X_scaled, method='kmeans', k=6):
    """
    Apply chosen clustering method.
    """
    logger.info("\n" + "="*80)
    logger.info(f"APPLYING FINAL CLUSTERING: {method.upper()} with k={k}")
    logger.info("="*80)

    if method == 'kmeans':
        clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X_scaled)
    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = clusterer.fit_predict(X_scaled)
    else:  # dbscan (shouldn't reach here typically)
        clusterer = DBSCAN(eps=0.5, min_samples=50)
        labels = clusterer.fit_predict(X_scaled)

    logger.info(f"Clustered {len(labels):,} conversations into {len(set(labels))} groups")

    return labels


# ==============================================================================
# SECTION 4: CLUSTER CHARACTERIZATION & VALIDATION
# ==============================================================================

def compute_cluster_statistics(df, labels, feature_cols):
    """
    Compute statistics and effect sizes for each cluster.
    """
    logger.info("\n" + "="*80)
    logger.info("CLUSTER STATISTICS & VALIDATION")
    logger.info("="*80)

    df['cluster'] = labels

    cluster_stats = {}

    for cluster_id in sorted(set(labels)):
        cluster_df = df[df['cluster'] == cluster_id]

        stats = {
            'cluster_id': int(cluster_id),
            'size': len(cluster_df),
            'percentage': len(cluster_df) / len(df) * 100,
            'mean_features': {}
        }

        for col in feature_cols:
            stats['mean_features'][col] = float(cluster_df[col].mean())

        cluster_stats[f'Cluster_{cluster_id}'] = stats

    # Compute effect sizes (eta-squared) for each feature
    logger.info("\nEffect sizes (η²) for each feature:")

    effect_sizes = {}
    for col in feature_cols:
        groups = [df[df['cluster'] == c][col].values for c in sorted(set(labels))]
        f_stat, p_value = f_oneway(*groups)

        # Eta-squared
        grand_mean = df[col].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_total = sum((df[col] - grand_mean)**2)
        eta_squared = ss_between / ss_total

        effect_sizes[col] = {
            'eta_squared': float(eta_squared),
            'f_statistic': float(f_stat),
            'p_value': float(p_value)
        }

        logger.info(f"  {col}: η² = {eta_squared:.3f} (F={f_stat:.1f}, p={p_value:.2e})")

    return cluster_stats, effect_sizes


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    processed_dir = project_dir / "data" / "processed"

    # Find the most recent features file
    feature_files = list(processed_dir.glob("*_features.parquet"))

    if not feature_files:
        logger.error(f"No feature files found in {processed_dir}")
        logger.error("Run 01_extract_features.py first!")
        return 1

    # Use the most recently modified features file
    input_file = max(feature_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using features from: {input_file.name}")

    # Extract dataset name from features file
    dataset_name = input_file.stem.replace('_features', '')
    output_file = project_dir / "results" / f"{dataset_name}_clustering_results.json"

    # Load features
    logger.info(f"Loading features from {input_file.name}")
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded {len(df):,} conversations")

    feature_cols = ['burst_coefficient', 'cluster_density', 'response_acceleration',
                    'memory_coefficient', 'timing_consistency']

    X = df[feature_cols].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 1: Determine optimal k
    optimal_k_df, optimal_k = determine_optimal_k(X_scaled)

    # Step 2: Compare methods
    method_comparison = compare_clustering_methods(X_scaled, optimal_k=optimal_k)

    # Step 3: Apply final clustering
    best_method = method_comparison['recommended']
    labels = apply_final_clustering(X_scaled, method=best_method, k=optimal_k)

    # Step 4: Compute statistics
    cluster_stats, effect_sizes = compute_cluster_statistics(df, labels, feature_cols)

    # Save results
    results = {
        'metadata': {
            'n_conversations': len(df),
            'n_clusters': optimal_k,
            'method': best_method,
            'feature_columns': feature_cols
        },
        'optimal_k_analysis': optimal_k_df.to_dict(orient='records'),
        'method_comparison': method_comparison,
        'cluster_statistics': cluster_stats,
        'effect_sizes': effect_sizes,
        'cluster_assignments': [
            {'conversation_id': float(cid) if isinstance(cid, (int, float)) else cid,
             'cluster': int(label)}
            for cid, label in zip(df['conversation_id'], labels)
        ]
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved: Results saved to {output_file}")
    logger.info(f"  Optimal k: {optimal_k}")
    logger.info(f"  Method: {best_method}")
    logger.info(f"  Clusters: {len(cluster_stats)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
