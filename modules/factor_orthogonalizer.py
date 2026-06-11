"""
Factor Orthogonalizer Module - Advanced Optimization
Implements Gram-Schmidt orthogonalization to remove multicollinearity
between trading factors
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class FactorOrthogonalizer:
    """Orthogonalize trading factors using Gram-Schmidt process"""
    
    def __init__(self):
        # Factor definitions
        self.factor_names = [
            'momentum',
            'mean_reversion', 
            'trend_following',
            'volatility',
            'volume',
            'sentiment'
        ]
        
        # Factor correlations (initial estimates)
        self.factor_correlations = {}
        
        # Orthogonalized factor values
        self.orthogonal_factors = None
        
        # Factor loadings
        self.factor_loadings = None
    
    def calculate_factor_correlation_matrix(self, factor_values: np.ndarray) -> np.ndarray:
        """
        Calculate correlation matrix between factors
        
        Args:
            factor_values: 2D array of shape (n_samples, n_factors)
        
        Returns:
            Correlation matrix of shape (n_factors, n_factors)
        """
        n_factors = factor_values.shape[1]
        correlation_matrix = np.zeros((n_factors, n_factors))
        
        for i in range(n_factors):
            for j in range(n_factors):
                if i == j:
                    correlation_matrix[i][j] = 1.0
                else:
                    # Calculate Pearson correlation
                    cov = np.cov(factor_values[:, i], factor_values[:, j])[0][1]
                    std_i = np.std(factor_values[:, i])
                    std_j = np.std(factor_values[:, j])
                    
                    if std_i > 0 and std_j > 0:
                        correlation_matrix[i][j] = cov / (std_i * std_j)
                    else:
                        correlation_matrix[i][j] = 0.0
        
        return correlation_matrix
    
    def gram_schmidt_orthogonalize(self, factor_values: np.ndarray) -> np.ndarray:
        """
        Apply Gram-Schmidt orthogonalization to factor values
        
        Args:
            factor_values: 2D array of shape (n_samples, n_factors)
        
        Returns:
            Orthogonalized factor values
        """
        n_samples, n_factors = factor_values.shape
        orthogonal = np.zeros_like(factor_values)
        
        # Initialize with first factor
        orthogonal[:, 0] = factor_values[:, 0]
        
        # Apply Gram-Schmidt process
        for i in range(1, n_factors):
            orthogonal[:, i] = factor_values[:, i]
            
            # Subtract projections onto previous orthogonal factors
            for j in range(i):
                # Calculate projection coefficient
                dot_product = np.dot(orthogonal[:, i], orthogonal[:, j])
                norm_squared = np.dot(orthogonal[:, j], orthogonal[:, j])
                
                if norm_squared > 0:
                    projection_coeff = dot_product / norm_squared
                    orthogonal[:, i] -= projection_coeff * orthogonal[:, j]
            
            # Normalize
            norm = np.linalg.norm(orthogonal[:, i])
            if norm > 0:
                orthogonal[:, i] /= norm
        
        return orthogonal
    
    def orthogonalize_factors(self, factor_values: np.ndarray) -> Dict:
        """
        Main orthogonalization function
        
        Args:
            factor_values: 2D array of shape (n_samples, n_factors)
        
        Returns:
            Dictionary with orthogonalized factors and statistics
        """
        # Calculate correlation matrix before orthogonalization
        correlation_before = self.calculate_factor_correlation_matrix(factor_values)
        
        # Apply Gram-Schmidt orthogonalization
        orthogonal = self.gram_schmidt_orthogonalize(factor_values)
        
        # Calculate correlation matrix after orthogonalization
        correlation_after = self.calculate_factor_correlation_matrix(orthogonal)
        
        # Calculate factor loadings (how much each original factor contributes)
        factor_loadings = np.zeros((len(self.factor_names), len(self.factor_names)))
        for i in range(len(self.factor_names)):
            for j in range(len(self.factor_names)):
                if i >= j:
                    # Calculate loading as correlation between original and orthogonal
                    if np.std(factor_values[:, i]) > 0 and np.std(orthogonal[:, j]) > 0:
                        factor_loadings[i][j] = np.corrcoef(factor_values[:, i], orthogonal[:, j])[0][1]
                    else:
                        factor_loadings[i][j] = 0.0
        
        # Calculate information retention
        variance_before = np.sum(np.var(factor_values, axis=0))
        variance_after = np.sum(np.var(orthogonal, axis=0))
        information_retention = variance_after / variance_before if variance_before > 0 else 1.0
        
        return {
            'orthogonal_factors': orthogonal,
            'correlation_before': correlation_before,
            'correlation_after': correlation_after,
            'factor_loadings': factor_loadings,
            'information_retention': information_retention,
            'reduction_in_multicollinearity': self._calculate_multicollinearity_reduction(
                correlation_before, correlation_after
            )
        }
    
    def _calculate_multicollinearity_reduction(self, corr_before: np.ndarray, 
                                               corr_after: np.ndarray) -> float:
        """Calculate reduction in multicollinearity"""
        # Use sum of squared off-diagonal correlations as measure
        sum_sq_before = np.sum(corr_before ** 2) - np.trace(corr_before ** 2)
        sum_sq_after = np.sum(corr_after ** 2) - np.trace(corr_after ** 2)
        
        reduction = 1 - (sum_sq_after / sum_sq_before) if sum_sq_before > 0 else 0
        return max(0, min(1, reduction))
    
    def calculate_vif(self, factor_values: np.ndarray) -> np.ndarray:
        """
        Calculate Variance Inflation Factor (VIF) for each factor
        
        VIF > 10 indicates severe multicollinearity
        
        Args:
            factor_values: 2D array of shape (n_samples, n_factors)
        
        Returns:
            Array of VIF values
        """
        n_factors = factor_values.shape[1]
        vif_values = np.zeros(n_factors)
        
        for i in range(n_factors):
            # Regress factor i on all other factors
            X = np.delete(factor_values, i, axis=1)
            y = factor_values[:, i]
            
            # Calculate R-squared
            if np.linalg.matrix_rank(X) >= X.shape[1]:
                XTX_inv = np.linalg.pinv(X.T @ X)
                beta = XTX_inv @ X.T @ y
                y_hat = X @ beta
                ss_res = np.sum((y - y_hat) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                vif_values[i] = 1 / (1 - r_squared)
            else:
                vif_values[i] = np.inf
        
        return vif_values
    
    def filter_high_vif_factors(self, factor_values: np.ndarray, 
                                 vif_threshold: float = 10.0) -> Tuple[np.ndarray, List[int]]:
        """
        Filter out factors with high VIF
        
        Args:
            factor_values: 2D array of shape (n_samples, n_factors)
            vif_threshold: Maximum acceptable VIF
        
        Returns:
            Tuple of (filtered factors, indices of removed factors)
        """
        vif = self.calculate_vif(factor_values)
        high_vif_indices = np.where(vif > vif_threshold)[0]
        
        # Remove high VIF factors
        mask = np.ones(factor_values.shape[1], dtype=bool)
        mask[high_vif_indices] = False
        filtered_factors = factor_values[:, mask]
        
        return filtered_factors, high_vif_indices.tolist()
    
    def get_orthogonal_factor_weights(self, factor_values: np.ndarray,
                                       target_weights: Optional[Dict] = None) -> Dict:
        """
        Calculate optimal weights for orthogonalized factors
        
        Args:
            factor_values: 2D array of shape (n_samples, n_factors)
            target_weights: Optional target weights
        
        Returns:
            Dictionary with optimized weights
        """
        n_factors = factor_values.shape[1]
        
        # Calculate factor performance (using Sharpe-like metric)
        factor_means = np.mean(factor_values, axis=0)
        factor_stds = np.std(factor_values, axis=0)
        factor_scores = factor_means / factor_stds if np.all(factor_stds > 0) else np.zeros(n_factors)
        
        # Normalize scores to weights
        total_score = np.sum(np.abs(factor_scores))
        if total_score > 0:
            weights = factor_scores / total_score
        else:
            weights = np.ones(n_factors) / n_factors
        
        # Apply target weights if provided
        if target_weights:
            for i, name in enumerate(self.factor_names):
                if i < n_factors and name in target_weights:
                    weights[i] = weights[i] * target_weights[name]
            
            # Re-normalize
            total_weight = np.sum(np.abs(weights))
            if total_weight > 0:
                weights = weights / total_weight
        
        return {
            'factor_weights': weights,
            'factor_scores': factor_scores,
            'factor_names': self.factor_names[:n_factors],
            'total_exposure': np.sum(np.abs(weights))
        }
