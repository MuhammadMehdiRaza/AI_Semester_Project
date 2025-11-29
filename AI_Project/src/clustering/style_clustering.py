"""
DBSCAN Clustering for Coding Style Analysis
Clusters code files by coding style to detect plagiarism anomalies.

Anomaly Detection Logic:
- Students with similar code but DIFFERENT styles = likely plagiarism
- Students with similar code and SIMILAR styles = could be common patterns
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class StyleClusterer:
    """
    DBSCAN-based clustering for coding style analysis.
    
    Extracts style features from code and clusters programmers by their
    coding patterns. Detects anomalies where similar code comes from
    different style clusters.
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 2):
        """
        Initialize DBSCAN clusterer.
        
        Args:
            eps: Maximum distance between samples in a cluster
            min_samples: Minimum samples to form a cluster
        """
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = None
        self.scaler = StandardScaler()
        self.tsne = None
        self.labels_ = None
        self.style_features_ = None
        self.file_names_ = None
        self.tsne_embedding_ = None
        
    def extract_style_features(self, json_data: Dict) -> np.ndarray:
        """
        Extract style-related features from a single file's JSON data.
        
        Style features include:
        - Average function length (LOC / num_functions)
        - Naming conventions (identifier patterns)
        - Cyclomatic complexity preferences
        - Import patterns
        - Nesting depth preferences
        """
        features = []
        
        # 1. Average function length
        loc = json_data.get('loc', 0)
        num_funcs = json_data.get('num_functions', 1) or 1
        avg_func_length = loc / num_funcs
        features.append(avg_func_length)
        
        # 2. Complexity preferences
        avg_cc = json_data.get('avg_cc', 0)
        max_cc = json_data.get('max_cc', 0)
        features.append(avg_cc)
        features.append(max_cc)
        
        # 3. Import count (library usage style)
        imports = json_data.get('imports', [])
        features.append(len(imports))
        
        # 4. Function count preference
        features.append(num_funcs)
        
        # 5. Average arguments per function
        funcs = json_data.get('functions', [])
        if funcs:
            avg_args = sum(f.get('args', 0) for f in funcs) / len(funcs)
        else:
            avg_args = 0
        features.append(avg_args)
        
        # 6. Identifier diversity (unique identifiers / total)
        top_idents = json_data.get('top_idents', [])
        total_idents = sum(count for _, count in top_idents) if top_idents else 0
        unique_idents = len(top_idents)
        ident_diversity = unique_idents / total_idents if total_idents > 0 else 0
        features.append(ident_diversity)
        
        # 7. Node type diversity (AST complexity)
        node_hist = json_data.get('node_hist', {})
        total_nodes = sum(node_hist.values()) if node_hist else 0
        unique_nodes = len(node_hist)
        node_diversity = unique_nodes / total_nodes if total_nodes > 0 else 0
        features.append(node_diversity)
        
        # 8. Comment density (comments / LOC) - approximated from node histogram
        comment_nodes = node_hist.get('Expr', 0)  # String expressions often are docstrings
        comment_density = comment_nodes / loc if loc > 0 else 0
        features.append(comment_density)
        
        # 9. Control flow complexity (loops + conditionals ratio)
        loops = node_hist.get('For', 0) + node_hist.get('While', 0)
        conditionals = node_hist.get('If', 0)
        control_flow_ratio = (loops + conditionals) / total_nodes if total_nodes > 0 else 0
        features.append(control_flow_ratio)
        
        # 10. Function definition ratio
        func_defs = node_hist.get('FunctionDef', 0)
        func_ratio = func_defs / total_nodes if total_nodes > 0 else 0
        features.append(func_ratio)
        
        return np.array(features)
    
    def load_and_extract_features(self, json_dir: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load JSON files and extract style features from each.
        
        Args:
            json_dir: Directory containing per-file JSON data
            
        Returns:
            (features_array, file_names)
        """
        json_path = Path(json_dir)
        json_files = sorted(json_path.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {json_dir}")
        
        features = []
        file_names = []
        
        for jf in json_files:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            style_feats = self.extract_style_features(data)
            features.append(style_feats)
            file_names.append(jf.stem)
        
        return np.array(features), file_names
    
    def fit(self, json_dir: str) -> 'StyleClusterer':
        """
        Fit DBSCAN clustering on style features.
        
        Args:
            json_dir: Directory containing per-file JSON data
            
        Returns:
            self
        """
        print(f"Loading style features from {json_dir}...")
        features, file_names = self.load_and_extract_features(json_dir)
        
        self.style_features_ = features
        self.file_names_ = file_names
        
        print(f"Extracted {features.shape[1]} style features from {len(file_names)} files")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit DBSCAN
        print(f"Fitting DBSCAN (eps={self.eps}, min_samples={self.min_samples})...")
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = self.dbscan.fit_predict(features_scaled)
        
        # Count clusters
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)
        
        print(f"Found {n_clusters} clusters, {n_noise} noise points")
        
        # Compute silhouette score if more than 1 cluster
        if n_clusters > 1:
            non_noise_mask = self.labels_ != -1
            if non_noise_mask.sum() > 1:
                score = silhouette_score(features_scaled[non_noise_mask], 
                                        self.labels_[non_noise_mask])
                print(f"Silhouette score: {score:.4f}")
        
        return self
    
    def compute_tsne(self, perplexity: int = 5, random_state: int = 42) -> np.ndarray:
        """
        Compute t-SNE embedding for visualization.
        
        Args:
            perplexity: t-SNE perplexity (should be < n_samples)
            random_state: Random seed
            
        Returns:
            2D embedding array
        """
        if self.style_features_ is None:
            raise ValueError("Must call fit() first")
        
        n_samples = len(self.style_features_)
        perplexity = min(perplexity, n_samples - 1)
        
        print(f"Computing t-SNE embedding (perplexity={perplexity})...")
        
        features_scaled = self.scaler.transform(self.style_features_)
        
        self.tsne = TSNE(n_components=2, perplexity=perplexity, 
                        random_state=random_state, n_iter=1000)
        self.tsne_embedding_ = self.tsne.fit_transform(features_scaled)
        
        return self.tsne_embedding_
    
    def detect_anomalies(self, similarity_threshold: float = 0.7,
                        features_df: pd.DataFrame = None) -> List[Dict]:
        """
        Detect plagiarism anomalies: similar code from different style clusters.
        
        Args:
            similarity_threshold: Minimum similarity to consider as potential plagiarism
            features_df: DataFrame with pairwise features including 'file1', 'file2', 
                        'canonical_similarity', and 'label'
                        
        Returns:
            List of anomaly dictionaries
        """
        if self.labels_ is None:
            raise ValueError("Must call fit() first")
        
        if features_df is None:
            print("No features DataFrame provided, skipping anomaly detection")
            return []
        
        # Create file -> cluster mapping
        file_to_cluster = {name: label for name, label in 
                          zip(self.file_names_, self.labels_)}
        
        anomalies = []
        
        for _, row in features_df.iterrows():
            file1 = row.get('file1', '').replace('.json', '')
            file2 = row.get('file2', '').replace('.json', '')
            similarity = row.get('canonical_similarity', 0)
            
            # Skip if below similarity threshold
            if similarity < similarity_threshold:
                continue
            
            cluster1 = file_to_cluster.get(file1, -99)
            cluster2 = file_to_cluster.get(file2, -99)
            
            # Anomaly: High similarity but different clusters
            if cluster1 != cluster2 and cluster1 != -99 and cluster2 != -99:
                anomalies.append({
                    'file1': file1,
                    'file2': file2,
                    'similarity': similarity,
                    'cluster1': int(cluster1),
                    'cluster2': int(cluster2),
                    'anomaly_type': 'different_style_high_similarity',
                    'risk_level': 'HIGH' if similarity > 0.85 else 'MEDIUM'
                })
        
        print(f"Detected {len(anomalies)} anomalies (similar code, different styles)")
        return anomalies
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get summary of clusters with file assignments.
        
        Returns:
            DataFrame with file names and cluster assignments
        """
        if self.labels_ is None:
            raise ValueError("Must call fit() first")
        
        data = {
            'file': self.file_names_,
            'cluster': self.labels_
        }
        
        if self.tsne_embedding_ is not None:
            data['tsne_x'] = self.tsne_embedding_[:, 0]
            data['tsne_y'] = self.tsne_embedding_[:, 1]
        
        return pd.DataFrame(data)
    
    def save_visualization(self, output_path: str, title: str = "Code Style Clusters (t-SNE)"):
        """
        Save t-SNE visualization with cluster coloring.
        
        Args:
            output_path: Path to save the plot
            title: Plot title
        """
        if not HAS_MATPLOTLIB:
            print("WARNING: Matplotlib not available, cannot save visualization")
            return
        
        if self.tsne_embedding_ is None:
            self.compute_tsne()
        
        print(f"Generating t-SNE visualization...")
        
        plt.figure(figsize=(12, 8))
        
        # Color by cluster
        unique_labels = set(self.labels_)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(sorted(unique_labels), colors):
            mask = self.labels_ == label
            
            if label == -1:
                # Noise points
                plt.scatter(self.tsne_embedding_[mask, 0], 
                           self.tsne_embedding_[mask, 1],
                           c='gray', marker='x', s=100, alpha=0.5,
                           label='Noise/Outliers')
            else:
                plt.scatter(self.tsne_embedding_[mask, 0], 
                           self.tsne_embedding_[mask, 1],
                           c=[color], marker='o', s=100, alpha=0.7,
                           label=f'Cluster {label}')
        
        # Add file labels
        for i, name in enumerate(self.file_names_):
            short_name = name[:15] + '...' if len(name) > 15 else name
            plt.annotate(short_name, 
                        (self.tsne_embedding_[i, 0], self.tsne_embedding_[i, 1]),
                        fontsize=7, alpha=0.7,
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {output_path}")
    
    def save_anomaly_visualization(self, anomalies: List[Dict], 
                                   output_path: str,
                                   title: str = "Plagiarism Anomalies Detection"):
        """
        Visualize anomalies on t-SNE plot.
        
        Args:
            anomalies: List of anomaly dictionaries
            output_path: Path to save the plot
            title: Plot title
        """
        if not HAS_MATPLOTLIB:
            print("WARNING: Matplotlib not available")
            return
        
        if self.tsne_embedding_ is None:
            self.compute_tsne()
        
        plt.figure(figsize=(12, 8))
        
        # Plot all points
        plt.scatter(self.tsne_embedding_[:, 0], 
                   self.tsne_embedding_[:, 1],
                   c='lightblue', marker='o', s=80, alpha=0.5,
                   label='Normal')
        
        # Highlight anomalous files
        anomalous_files = set()
        for a in anomalies:
            anomalous_files.add(a['file1'])
            anomalous_files.add(a['file2'])
        
        for i, name in enumerate(self.file_names_):
            if name in anomalous_files:
                plt.scatter(self.tsne_embedding_[i, 0], 
                           self.tsne_embedding_[i, 1],
                           c='red', marker='*', s=200, alpha=0.9,
                           zorder=5)
        
        # Draw lines between anomalous pairs
        file_to_idx = {name: i for i, name in enumerate(self.file_names_)}
        for a in anomalies:
            idx1 = file_to_idx.get(a['file1'])
            idx2 = file_to_idx.get(a['file2'])
            if idx1 is not None and idx2 is not None:
                plt.plot([self.tsne_embedding_[idx1, 0], self.tsne_embedding_[idx2, 0]],
                        [self.tsne_embedding_[idx1, 1], self.tsne_embedding_[idx2, 1]],
                        'r--', linewidth=2, alpha=0.5)
        
        # Add legend
        plt.scatter([], [], c='red', marker='*', s=200, label='Suspicious')
        plt.plot([], [], 'r--', linewidth=2, label='High Similarity Pair')
        
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved anomaly visualization to {output_path}")
    
    def save(self, path: str):
        """Save clusterer state to file."""
        state = {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'labels_': self.labels_,
            'style_features_': self.style_features_,
            'file_names_': self.file_names_,
            'tsne_embedding_': self.tsne_embedding_,
            'scaler': self.scaler
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved clusterer to {path}")
    
    def load(self, path: str) -> 'StyleClusterer':
        """Load clusterer state from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.eps = state['eps']
        self.min_samples = state['min_samples']
        self.labels_ = state['labels_']
        self.style_features_ = state['style_features_']
        self.file_names_ = state['file_names_']
        self.tsne_embedding_ = state['tsne_embedding_']
        self.scaler = state['scaler']
        
        print(f"Loaded clusterer from {path}")
        return self


def main():
    """Run DBSCAN clustering on code files."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DBSCAN clustering for code style analysis")
    parser.add_argument("--json-dir", default="src/preprocess/data/processed/files",
                       help="Directory containing per-file JSON data")
    parser.add_argument("--features-csv", default="src/feature_selection/artifacts/features.csv",
                       help="Path to pairwise features CSV (for anomaly detection)")
    parser.add_argument("--output-dir", default="src/clustering/artifacts",
                       help="Output directory for results")
    parser.add_argument("--eps", type=float, default=1.0,
                       help="DBSCAN eps parameter")
    parser.add_argument("--min-samples", type=int, default=2,
                       help="DBSCAN min_samples parameter")
    parser.add_argument("--similarity-threshold", type=float, default=0.6,
                       help="Minimum similarity for anomaly detection")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize and fit clusterer
    clusterer = StyleClusterer(eps=args.eps, min_samples=args.min_samples)
    
    try:
        clusterer.fit(args.json_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run preprocessing first:")
        print("  python src/preprocess/extract_perfile.py <code_dir>")
        return
    
    # Compute t-SNE
    clusterer.compute_tsne()
    
    # Save cluster summary
    summary = clusterer.get_cluster_summary()
    summary_path = output_dir / "cluster_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved cluster summary to {summary_path}")
    
    # Save visualization
    viz_path = output_dir / "style_clusters_tsne.png"
    clusterer.save_visualization(str(viz_path))
    
    # Detect anomalies if features CSV exists
    if Path(args.features_csv).exists():
        print(f"\nLoading pairwise features from {args.features_csv}...")
        features_df = pd.read_csv(args.features_csv)
        
        anomalies = clusterer.detect_anomalies(
            similarity_threshold=args.similarity_threshold,
            features_df=features_df
        )
        
        if anomalies:
            # Save anomalies
            anomalies_df = pd.DataFrame(anomalies)
            anomalies_path = output_dir / "anomalies.csv"
            anomalies_df.to_csv(anomalies_path, index=False)
            print(f"Saved anomalies to {anomalies_path}")
            
            # Visualize anomalies
            anomaly_viz_path = output_dir / "anomalies_visualization.png"
            clusterer.save_anomaly_visualization(anomalies, str(anomaly_viz_path))
    
    # Save clusterer state
    clusterer_path = output_dir / "style_clusterer.pkl"
    clusterer.save(str(clusterer_path))
    
    print("\n" + "="*60)
    print("CLUSTERING COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - cluster_summary.csv")
    print(f"  - style_clusters_tsne.png")
    print(f"  - style_clusterer.pkl")
    if Path(args.features_csv).exists():
        print(f"  - anomalies.csv (if any found)")
        print(f"  - anomalies_visualization.png (if any found)")


if __name__ == "__main__":
    main()
