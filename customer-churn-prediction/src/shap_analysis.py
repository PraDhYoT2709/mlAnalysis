"""
SHAP Analysis for Customer Churn Prediction
This module provides model explainability using SHAP (SHapley Additive exPlanations)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

class ChurnSHAPAnalyzer:
    def __init__(self, model, X_train, X_test, feature_names):
        """
        Initialize SHAP analyzer
        
        Args:
            model: Trained ML model
            X_train: Training features
            X_test: Test features  
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
        # Initialize SHAP explainer
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer based on model type"""
        model_name = type(self.model).__name__.lower()
        
        print(f"Initializing SHAP explainer for {model_name}...")
        
        try:
            if 'xgb' in model_name or 'gradient' in model_name or 'random' in model_name:
                # Tree-based models
                self.explainer = shap.TreeExplainer(self.model)
                print("Using TreeExplainer")
            elif 'logistic' in model_name or 'linear' in model_name:
                # Linear models
                self.explainer = shap.LinearExplainer(self.model, self.X_train)
                print("Using LinearExplainer")
            else:
                # General explainer (slower but works for any model)
                self.explainer = shap.Explainer(self.model, self.X_train)
                print("Using general Explainer")
                
        except Exception as e:
            print(f"Error initializing specific explainer: {e}")
            print("Falling back to general Explainer...")
            self.explainer = shap.Explainer(self.model.predict, self.X_train)
    
    def calculate_shap_values(self, X=None, max_samples=1000):
        """
        Calculate SHAP values for the dataset
        
        Args:
            X: Data to explain (defaults to test set)
            max_samples: Maximum number of samples to analyze (for performance)
        """
        if X is None:
            X = self.X_test
        
        # Limit samples for performance
        if len(X) > max_samples:
            print(f"Limiting analysis to {max_samples} samples for performance")
            X = X.sample(n=max_samples, random_state=42)
        
        print("Calculating SHAP values...")
        
        try:
            self.shap_values = self.explainer(X)
            print(f"SHAP values calculated for {len(X)} samples")
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            # Fallback method
            try:
                self.shap_values = self.explainer.shap_values(X)
                if isinstance(self.shap_values, list):
                    # For binary classification, take positive class
                    self.shap_values = self.shap_values[1]
                print(f"SHAP values calculated using fallback method")
            except Exception as e2:
                print(f"Fallback method also failed: {e2}")
                return None
        
        return self.shap_values
    
    def plot_summary(self, save_path=None, max_display=20):
        """
        Create SHAP summary plot showing feature importance and impact
        
        Args:
            save_path: Path to save the plot
            max_display: Maximum number of features to display
        """
        if self.shap_values is None:
            print("SHAP values not calculated. Run calculate_shap_values() first.")
            return
        
        plt.figure(figsize=(12, 8))
        
        try:
            if hasattr(self.shap_values, 'values'):
                # New SHAP version format
                shap.summary_plot(
                    self.shap_values.values, 
                    self.shap_values.data, 
                    feature_names=self.feature_names,
                    max_display=max_display,
                    show=False
                )
            else:
                # Old SHAP format
                shap.summary_plot(
                    self.shap_values, 
                    self.X_test,
                    feature_names=self.feature_names,
                    max_display=max_display,
                    show=False
                )
        except Exception as e:
            print(f"Error creating summary plot: {e}")
            return
        
        plt.title('SHAP Summary Plot - Feature Impact on Churn Prediction', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_bar(self, save_path=None, max_display=15):
        """
        Create SHAP bar plot showing mean absolute feature importance
        
        Args:
            save_path: Path to save the plot
            max_display: Maximum number of features to display
        """
        if self.shap_values is None:
            print("SHAP values not calculated. Run calculate_shap_values() first.")
            return
        
        plt.figure(figsize=(10, 8))
        
        try:
            if hasattr(self.shap_values, 'values'):
                shap.plots.bar(self.shap_values, max_display=max_display, show=False)
            else:
                # Calculate mean absolute SHAP values manually
                mean_shap = np.abs(self.shap_values).mean(axis=0)
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': mean_shap
                }).sort_values('importance', ascending=True).tail(max_display)
                
                plt.barh(range(len(feature_importance)), feature_importance['importance'])
                plt.yticks(range(len(feature_importance)), feature_importance['feature'])
                plt.xlabel('Mean |SHAP value|')
                plt.title('Feature Importance (Mean Absolute SHAP Values)')
        except Exception as e:
            print(f"Error creating bar plot: {e}")
            return
        
        plt.title('SHAP Feature Importance - Mean Absolute Impact', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP bar plot saved to {save_path}")
        
        plt.show()
    
    def plot_waterfall(self, sample_idx=0, save_path=None):
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            sample_idx: Index of the sample to explain
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            print("SHAP values not calculated. Run calculate_shap_values() first.")
            return
        
        plt.figure(figsize=(12, 8))
        
        try:
            if hasattr(self.shap_values, 'values'):
                shap.plots.waterfall(self.shap_values[sample_idx], show=False)
            else:
                # Manual waterfall plot creation
                sample_shap = self.shap_values[sample_idx]
                sample_data = self.X_test.iloc[sample_idx] if hasattr(self.X_test, 'iloc') else self.X_test[sample_idx]
                
                # Create waterfall-like plot
                sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:15]  # Top 15 features
                
                plt.figure(figsize=(12, 8))
                colors = ['red' if x < 0 else 'blue' for x in sample_shap[sorted_idx]]
                
                plt.barh(range(len(sorted_idx)), sample_shap[sorted_idx], color=colors)
                plt.yticks(range(len(sorted_idx)), 
                          [f"{self.feature_names[i]}" for i in sorted_idx])
                plt.xlabel('SHAP value')
                plt.title(f'SHAP Explanation for Sample {sample_idx}')
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        except Exception as e:
            print(f"Error creating waterfall plot: {e}")
            return
        
        plt.title(f'SHAP Waterfall Plot - Individual Prediction Explanation (Sample {sample_idx})', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP waterfall plot saved to {save_path}")
        
        plt.show()
    
    def plot_force(self, sample_idx=0, save_path=None):
        """
        Create SHAP force plot for a single prediction
        
        Args:
            sample_idx: Index of the sample to explain
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            print("SHAP values not calculated. Run calculate_shap_values() first.")
            return
        
        try:
            if hasattr(self.shap_values, 'values'):
                # New SHAP version
                shap.plots.force(
                    self.shap_values[sample_idx],
                    matplotlib=True,
                    show=False
                )
            else:
                # Old SHAP version
                shap.force_plot(
                    self.explainer.expected_value,
                    self.shap_values[sample_idx],
                    self.X_test.iloc[sample_idx] if hasattr(self.X_test, 'iloc') else self.X_test[sample_idx],
                    feature_names=self.feature_names,
                    matplotlib=True,
                    show=False
                )
        except Exception as e:
            print(f"Error creating force plot: {e}")
            return
        
        plt.title(f'SHAP Force Plot - Prediction Breakdown (Sample {sample_idx})', 
                 fontsize=14, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP force plot saved to {save_path}")
        
        plt.show()
    
    def plot_partial_dependence(self, feature_name, save_path=None):
        """
        Create partial dependence plot for a specific feature
        
        Args:
            feature_name: Name of the feature to analyze
            save_path: Path to save the plot
        """
        if feature_name not in self.feature_names:
            print(f"Feature '{feature_name}' not found in feature names")
            return
        
        feature_idx = self.feature_names.index(feature_name)
        
        try:
            plt.figure(figsize=(10, 6))
            shap.plots.partial_dependence(
                feature_name, self.model.predict, self.X_train, 
                ice=False, model_expected_value=True, feature_expected_value=True,
                show=False
            )
            
            plt.title(f'Partial Dependence Plot - {feature_name}', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Partial dependence plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating partial dependence plot: {e}")
    
    def get_feature_importance_ranking(self):
        """
        Get feature importance ranking based on mean absolute SHAP values
        
        Returns:
            DataFrame with features ranked by importance
        """
        if self.shap_values is None:
            print("SHAP values not calculated. Run calculate_shap_values() first.")
            return None
        
        try:
            if hasattr(self.shap_values, 'values'):
                shap_vals = self.shap_values.values
            else:
                shap_vals = self.shap_values
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            
            # Create ranking DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'mean_abs_shap': mean_abs_shap,
                'rank': range(1, len(self.feature_names) + 1)
            }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
            
            importance_df['rank'] = range(1, len(importance_df) + 1)
            
            return importance_df
            
        except Exception as e:
            print(f"Error calculating feature importance ranking: {e}")
            return None
    
    def analyze_feature_interactions(self, feature1, feature2, save_path=None):
        """
        Analyze interaction between two features
        
        Args:
            feature1: Name of first feature
            feature2: Name of second feature
            save_path: Path to save the plot
        """
        if feature1 not in self.feature_names or feature2 not in self.feature_names:
            print("One or both features not found in feature names")
            return
        
        try:
            plt.figure(figsize=(10, 8))
            
            # Create interaction plot
            shap.plots.scatter(
                self.shap_values[:, self.feature_names.index(feature1)],
                color=self.shap_values[:, self.feature_names.index(feature2)],
                show=False
            )
            
            plt.title(f'Feature Interaction: {feature1} vs {feature2}', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Feature interaction plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating feature interaction plot: {e}")
    
    def generate_explanation_report(self, top_n_features=10):
        """
        Generate a comprehensive explanation report
        
        Args:
            top_n_features: Number of top features to include in the report
            
        Returns:
            Dictionary containing explanation insights
        """
        if self.shap_values is None:
            print("SHAP values not calculated. Run calculate_shap_values() first.")
            return None
        
        # Get feature importance ranking
        importance_df = self.get_feature_importance_ranking()
        
        if importance_df is None:
            return None
        
        # Generate report
        report = {
            'model_type': type(self.model).__name__,
            'total_features': len(self.feature_names),
            'samples_analyzed': len(self.shap_values),
            'top_features': importance_df.head(top_n_features).to_dict('records'),
            'feature_insights': {}
        }
        
        # Add insights for top features
        for idx, row in importance_df.head(top_n_features).iterrows():
            feature_name = row['feature']
            feature_idx = self.feature_names.index(feature_name)
            
            if hasattr(self.shap_values, 'values'):
                feature_shap = self.shap_values.values[:, feature_idx]
            else:
                feature_shap = self.shap_values[:, feature_idx]
            
            report['feature_insights'][feature_name] = {
                'mean_abs_impact': row['mean_abs_shap'],
                'rank': row['rank'],
                'positive_impact_samples': np.sum(feature_shap > 0),
                'negative_impact_samples': np.sum(feature_shap < 0),
                'max_positive_impact': np.max(feature_shap),
                'max_negative_impact': np.min(feature_shap)
            }
        
        return report
    
    def create_comprehensive_analysis(self, save_dir=None):
        """
        Create a comprehensive SHAP analysis with all plots
        
        Args:
            save_dir: Directory to save all plots
        """
        import os
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        print("Creating comprehensive SHAP analysis...")
        
        # Summary plot
        summary_path = os.path.join(save_dir, 'shap_summary.png') if save_dir else None
        self.plot_summary(save_path=summary_path)
        
        # Bar plot
        bar_path = os.path.join(save_dir, 'shap_bar.png') if save_dir else None
        self.plot_bar(save_path=bar_path)
        
        # Waterfall plot for first sample
        waterfall_path = os.path.join(save_dir, 'shap_waterfall.png') if save_dir else None
        self.plot_waterfall(sample_idx=0, save_path=waterfall_path)
        
        # Feature importance ranking
        importance_df = self.get_feature_importance_ranking()
        if importance_df is not None and save_dir:
            importance_path = os.path.join(save_dir, 'feature_importance_ranking.csv')
            importance_df.to_csv(importance_path, index=False)
            print(f"Feature importance ranking saved to {importance_path}")
        
        # Generate explanation report
        report = self.generate_explanation_report()
        if report and save_dir:
            import json
            report_path = os.path.join(save_dir, 'shap_explanation_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"SHAP explanation report saved to {report_path}")
        
        print("Comprehensive SHAP analysis completed!")
        return report

def main():
    """Example usage of ChurnSHAPAnalyzer"""
    print("ChurnSHAPAnalyzer class initialized successfully!")
    print("Use this class with your trained models to generate SHAP explanations.")

if __name__ == "__main__":
    main()