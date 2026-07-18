"""
Unit tests for unsupervised learning algorithms: PCA, K-Means, DBSCAN.
"""

import pytest
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unsupervised.pca import PCA
from src.unsupervised.kmeans import KMeans
from src.unsupervised.dbscan import DBSCAN
from src.experiments.unsupervised_analysis import run_unsupervised_pipeline


class TestPCA:
    """Tests for PCA implementation."""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple 2D data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        return X
    
    @pytest.fixture
    def high_dim_data(self):
        """Generate high-dimensional data."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        return X
    
    def test_pca_initialization(self):
        """Test PCA initialization."""
        pca = PCA(n_components=2)
        assert pca.n_components == 2
        assert pca.components_ is None
    
    def test_pca_invalid_n_components(self):
        """Test that invalid n_components raises error."""
        with pytest.raises(ValueError):
            PCA(n_components=0)
        with pytest.raises(ValueError):
            PCA(n_components=-1)
    
    def test_pca_fit_shape(self, simple_data):
        """Test that fit produces correct output shapes."""
        pca = PCA(n_components=2)
        pca.fit(simple_data)
        
        assert pca.components_.shape == (2, 2)
        assert pca.explained_variance_.shape == (2,)
        assert pca.explained_variance_ratio_.shape == (2,)
        assert pca.mean_.shape == (2,)
    
    def test_pca_variance_sum(self, high_dim_data):
        """Test that explained variance ratio sums to at most 1."""
        pca = PCA(n_components=5)
        pca.fit(high_dim_data)
        
        # Sum of ratios should be < 1.0 (not all variance captured)
        assert np.sum(pca.explained_variance_ratio_) < 1.0
        # But should be significant
        assert np.sum(pca.explained_variance_ratio_) > 0.5
    
    def test_pca_transform_shape(self, simple_data):
        """Test that transform produces correct output shape."""
        pca = PCA(n_components=1)
        pca.fit(simple_data)
        X_transformed = pca.transform(simple_data)
        
        assert X_transformed.shape == (100, 1)
    
    def test_pca_fit_transform(self, simple_data):
        """Test that fit_transform is equivalent to fit then transform."""
        pca1 = PCA(n_components=2)
        pca2 = PCA(n_components=2)
        
        X_ft = pca1.fit_transform(simple_data)
        X_ft2 = pca2.fit(simple_data).transform(simple_data)
        
        np.testing.assert_array_almost_equal(X_ft, X_ft2)
    
    def test_pca_n_components_exceeds_features(self, simple_data):
        """Test that requesting more components than features raises error."""
        pca = PCA(n_components=3)  # data has only 2 features
        with pytest.raises(ValueError):
            pca.fit(simple_data)
    
    def test_pca_invalid_input_shape(self, simple_data):
        """Test that 1D input raises error."""
        pca = PCA(n_components=1)
        with pytest.raises(ValueError):
            pca.fit(simple_data.ravel())
    
    def test_pca_transform_before_fit(self, simple_data):
        """Test that transform before fit raises error."""
        pca = PCA(n_components=2)
        with pytest.raises(ValueError):
            pca.transform(simple_data)
    
    def test_pca_determinism(self, simple_data):
        """Test that PCA is deterministic."""
        pca1 = PCA(n_components=2)
        pca2 = PCA(n_components=2)
        
        pca1.fit(simple_data)
        pca2.fit(simple_data)
        
        np.testing.assert_array_almost_equal(pca1.components_, pca2.components_)
        np.testing.assert_array_almost_equal(pca1.explained_variance_ratio_, 
                                            pca2.explained_variance_ratio_)


class TestKMeans:
    """Tests for K-Means implementation."""
    
    @pytest.fixture
    def blob_data(self):
        """Generate blob data for clustering."""
        X, y = make_blobs(n_samples=100, n_features=2, centers=3, 
                         random_state=42, cluster_std=0.6)
        return X, y
    
    def test_kmeans_initialization(self):
        """Test K-Means initialization."""
        kmeans = KMeans(n_clusters=3, random_state=42)
        assert kmeans.n_clusters == 3
        assert kmeans.centroids_ is None
    
    def test_kmeans_invalid_n_clusters(self):
        """Test that invalid n_clusters raises error."""
        with pytest.raises(ValueError):
            KMeans(n_clusters=0)
        with pytest.raises(ValueError):
            KMeans(n_clusters=-1)
    
    def test_kmeans_fit_shapes(self, blob_data):
        """Test that fit produces correct output shapes."""
        X, y = blob_data
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        assert kmeans.centroids_.shape == (3, 2)
        assert kmeans.labels_.shape == (100,)
        assert isinstance(kmeans.inertia_, float)
    
    def test_kmeans_all_labels_valid(self, blob_data):
        """Test that all labels are in valid range."""
        X, y = blob_data
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        assert np.all(kmeans.labels_ >= 0)
        assert np.all(kmeans.labels_ < 3)
    
    def test_kmeans_inertia_positive(self, blob_data):
        """Test that inertia is positive."""
        X, y = blob_data
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        assert kmeans.inertia_ >= 0
    
    def test_kmeans_too_many_clusters(self, blob_data):
        """Test that requesting more clusters than samples raises error."""
        X, y = blob_data
        kmeans = KMeans(n_clusters=150, random_state=42)
        with pytest.raises(ValueError):
            kmeans.fit(X)
    
    def test_kmeans_invalid_input_shape(self, blob_data):
        """Test that 1D input raises error."""
        X, y = blob_data
        kmeans = KMeans(n_clusters=3, random_state=42)
        with pytest.raises(ValueError):
            kmeans.fit(X.ravel())
    
    def test_kmeans_determinism(self, blob_data):
        """Test that K-Means is deterministic with same random_state."""
        X, y = blob_data
        
        kmeans1 = KMeans(n_clusters=3, random_state=42)
        kmeans1.fit(X)
        
        kmeans2 = KMeans(n_clusters=3, random_state=42)
        kmeans2.fit(X)
        
        # Labels might be permuted, but clustering should be similar
        ari = adjusted_rand_score(kmeans1.labels_, kmeans2.labels_)
        assert ari > 0.95  # Very high agreement
    
    def test_kmeans_single_cluster(self, blob_data):
        """Test K-Means with k=1."""
        X, y = blob_data
        kmeans = KMeans(n_clusters=1, random_state=42)
        kmeans.fit(X)
        
        assert kmeans.centroids_.shape == (1, 2)
        assert np.all(kmeans.labels_ == 0)
    
    def test_kmeans_inertia_decreases(self, blob_data):
        """Test that inertia increases with more clusters."""
        X, y = blob_data
        
        inertias = []
        for k in range(1, 5):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Inertia should monotonically decrease as k increases
        for i in range(len(inertias) - 1):
            assert inertias[i] >= inertias[i+1]


class TestDBSCAN:
    """Tests for DBSCAN implementation."""
    
    @pytest.fixture
    def blob_data(self):
        """Generate blob data for clustering."""
        X, y = make_blobs(n_samples=100, n_features=2, centers=3,
                         random_state=42, cluster_std=0.6)
        return X, y
    
    @pytest.fixture
    def noisy_data(self):
        """Generate data with noise."""
        X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
        return X, y
    
    def test_dbscan_initialization(self):
        """Test DBSCAN initialization."""
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        assert dbscan.eps == 0.5
        assert dbscan.min_samples == 5
        assert dbscan.labels_ is None
    
    def test_dbscan_invalid_eps(self):
        """Test that invalid eps raises error."""
        with pytest.raises(ValueError):
            DBSCAN(eps=0, min_samples=5)
        with pytest.raises(ValueError):
            DBSCAN(eps=-1, min_samples=5)
    
    def test_dbscan_invalid_min_samples(self):
        """Test that invalid min_samples raises error."""
        with pytest.raises(ValueError):
            DBSCAN(eps=0.5, min_samples=0)
        with pytest.raises(ValueError):
            DBSCAN(eps=0.5, min_samples=-1)
    
    def test_dbscan_fit_shape(self, blob_data):
        """Test that fit produces correct output shape."""
        X, y = blob_data
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(X)
        
        assert dbscan.labels_.shape == (100,)
    
    def test_dbscan_labels_contain_noise(self, blob_data):
        """Test that labels can contain -1 for noise."""
        X, y = blob_data
        dbscan = DBSCAN(eps=0.05, min_samples=10)  # Very small eps
        dbscan.fit(X)
        
        # With very small eps, we expect some noise points
        assert -1 in dbscan.labels_
    
    def test_dbscan_no_noise_with_large_eps(self, blob_data):
        """Test that large eps produces no noise points."""
        X, y = blob_data
        dbscan = DBSCAN(eps=10.0, min_samples=2)  # Very large eps
        dbscan.fit(X)
        
        # With very large eps, there should be no noise points (-1)
        assert np.all(dbscan.labels_ >= 0)
        # All points should be clustered (no -1 labels)
        assert -1 not in dbscan.labels_
    
    def test_dbscan_invalid_input_shape(self, blob_data):
        """Test that 1D input raises error."""
        X, y = blob_data
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        with pytest.raises(ValueError):
            dbscan.fit(X.ravel())
    
    def test_dbscan_determinism(self, blob_data):
        """Test that DBSCAN is deterministic."""
        X, y = blob_data
        
        dbscan1 = DBSCAN(eps=0.5, min_samples=5)
        dbscan1.fit(X)
        
        dbscan2 = DBSCAN(eps=0.5, min_samples=5)
        dbscan2.fit(X)
        
        # Labels might be permuted, but clustering should be identical
        np.testing.assert_array_equal(dbscan1.labels_, dbscan2.labels_)
    
    def test_dbscan_connectivity(self, blob_data):
        """Test that DBSCAN respects eps parameter."""
        X, y = blob_data
        
        # Large eps should produce fewer clusters
        dbscan_small_eps = DBSCAN(eps=0.3, min_samples=5)
        dbscan_small_eps.fit(X)
        n_clusters_small = len(set(dbscan_small_eps.labels_)) - (1 if -1 in dbscan_small_eps.labels_ else 0)
        
        # Smaller eps should produce more clusters or more noise
        dbscan_large_eps = DBSCAN(eps=1.0, min_samples=5)
        dbscan_large_eps.fit(X)
        n_clusters_large = len(set(dbscan_large_eps.labels_)) - (1 if -1 in dbscan_large_eps.labels_ else 0)
        
        # With larger eps, we expect fewer clusters (more merging)
        assert n_clusters_large <= n_clusters_small + 1


class TestIntegration:
    """Integration tests for all unsupervised algorithms."""
    
    def test_pipeline_on_iris_like_data(self):
        """Test complete pipeline on blob data."""
        X, y = make_blobs(n_samples=150, n_features=4, centers=3,
                         random_state=42, cluster_std=0.7)
        
        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        assert X_pca.shape == (150, 2)
        assert np.sum(pca.explained_variance_ratio_) < 1.0
        assert np.sum(pca.explained_variance_ratio_) > 0.7
        
        # K-Means
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        ari_kmeans = adjusted_rand_score(y, kmeans.labels_)
        assert ari_kmeans > 0.7  # Should cluster well
        
        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(X)
        ari_dbscan = adjusted_rand_score(y, dbscan.labels_)
        assert ari_dbscan > 0.5  # Should be reasonable

    def test_pipeline_outputs_and_figures(self, tmp_path):
        X, y = make_blobs(n_samples=40, n_features=3, centers=3, random_state=42)
        X = StandardScaler().fit_transform(X)
        result = run_unsupervised_pipeline(X, y, "small test", figures_dir=tmp_path)

        assert result["components_90"] >= 1
        assert 1 <= result["optimal_k"] <= 10
        assert result["best_eps"] > 0
        assert 0 <= result["dbscan_noise_fraction"] <= 1
        assert len(list(tmp_path.glob("*.png"))) == 6


def test_edge_case_validation():
    with pytest.raises(ValueError):
        PCA(1).fit(np.ones((3, 2)))
    with pytest.raises(ValueError):
        PCA(1).fit(np.array([[1.0, np.nan], [2.0, 3.0]]))
    with pytest.raises(ValueError):
        KMeans(1).fit(np.empty((0, 2)))
    with pytest.raises(ValueError):
        DBSCAN(0.5, 2).fit(np.array([[1.0, np.inf]]))


def test_dbscan_matches_expected_border_reassignment():
    X = np.array([[0.0], [0.1], [0.2], [1.0]])
    model = DBSCAN(eps=0.11, min_samples=3).fit(X)
    np.testing.assert_array_equal(model.labels_, np.array([0, 0, 0, -1]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
