"""
Machine Learning Models for Customer Churn Prediction
This module implements multiple ML algorithms with hyperparameter tuning and evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    def __init__(self, random_state=42):
        """
        Initialize the Churn Predictor with multiple models
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.results = {}
        self.feature_importance = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models with default parameters"""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss'
            )
        }
    
    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        """
        Handle class imbalance using various techniques
        
        Args:
            X_train: Training features
            y_train: Training target
            method: 'smote', 'undersample', or 'class_weight'
            
        Returns:
            Balanced X_train, y_train
        """
        print(f"Handling class imbalance using {method}...")
        
        # Check original distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"Original distribution: {dict(zip(unique, counts))}")
        
        if method == 'smote':
            smote = SMOTE(random_state=self.random_state)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=self.random_state)
            X_balanced, y_balanced = undersampler.fit_resample(X_train, y_train)
        else:  # class_weight - handled in model parameters
            return X_train, y_train
        
        # Check new distribution
        unique, counts = np.unique(y_balanced, return_counts=True)
        print(f"Balanced distribution: {dict(zip(unique, counts))}")
        
        return X_balanced, y_balanced
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            model_name: Name of the model to tune
            X_train: Training features
            y_train: Training target
            cv: Cross-validation folds
            
        Returns:
            Best model after tuning
        """
        print(f"Tuning hyperparameters for {model_name}...")
        
        # Define parameter grids
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'scale_pos_weight': [1, 3, 5]  # For imbalanced classes
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return self.models[model_name]
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=self.models[model_name],
            param_grid=param_grids[model_name],
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_models(self, X_train, y_train, X_test, y_test, 
                    use_hyperparameter_tuning=True, balance_data=True):
        """
        Train all models and evaluate their performance
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            use_hyperparameter_tuning: Whether to tune hyperparameters
            balance_data: Whether to balance the training data
        """
        print("Starting model training...")
        
        # Handle class imbalance if requested
        if balance_data:
            X_train_balanced, y_train_balanced = self.handle_class_imbalance(
                X_train, y_train, method='smote'
            )
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Train each model
        for model_name in self.models.keys():
            print(f"\n{'='*50}")
            print(f"Training {model_name.replace('_', ' ').title()}")
            print(f"{'='*50}")
            
            # Hyperparameter tuning
            if use_hyperparameter_tuning:
                best_model = self.hyperparameter_tuning(
                    model_name, X_train_balanced, y_train_balanced
                )
                self.best_models[model_name] = best_model
            else:
                # Train with default parameters
                model = self.models[model_name]
                model.fit(X_train_balanced, y_train_balanced)
                self.best_models[model_name] = model
            
            # Evaluate model
            self._evaluate_model(model_name, X_test, y_test)
            
            # Store feature importance
            self._extract_feature_importance(model_name, X_train.columns)
    
    def _evaluate_model(self, model_name, X_test, y_test):
        """Evaluate a trained model and store results"""
        model = self.best_models[model_name]
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Print results
        print(f"\n{model_name.replace('_', ' ').title()} Results:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    def _extract_feature_importance(self, model_name, feature_names):
        """Extract and store feature importance"""
        model = self.best_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_[0])
        else:
            print(f"Cannot extract feature importance for {model_name}")
            return
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance[model_name] = feature_importance_df
        
        print(f"\nTop 10 Important Features for {model_name.replace('_', ' ').title()}:")
        print("-" * 60)
        print(feature_importance_df.head(10).to_string(index=False))
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        comparison_df = pd.DataFrame()
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_df[model_name.replace('_', ' ').title()] = [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score'],
                metrics['roc_auc']
            ]
        
        comparison_df.index = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        print(comparison_df.round(4))
        
        # Find best model for each metric
        print("\nBest Models by Metric:")
        print("-" * 30)
        for metric in comparison_df.index:
            best_model = comparison_df.loc[metric].idxmax()
            best_score = comparison_df.loc[metric].max()
            print(f"{metric}: {best_model} ({best_score:.4f})")
        
        return comparison_df
    
    def plot_model_comparison(self, save_path=None):
        """Plot model comparison visualization"""
        comparison_df = pd.DataFrame()
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_df[model_name.replace('_', ' ').title()] = [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score'],
                metrics['roc_auc']
            ]
        
        comparison_df.index = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Bar plot
        ax1 = axes[0, 0]
        comparison_df.plot(kind='bar', ax=ax1)
        ax1.set_title('Metrics Comparison')
        ax1.set_ylabel('Score')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # Heatmap
        ax2 = axes[0, 1]
        sns.heatmap(comparison_df, annot=True, cmap='YlOrRd', ax=ax2, fmt='.3f')
        ax2.set_title('Performance Heatmap')
        
        # ROC Curves
        ax3 = axes[1, 0]
        for model_name, result in self.results.items():
            if 'y_pred_proba' in result:
                # We need the actual test labels to plot ROC curve
                # This is a placeholder - you'd pass y_test to this method
                pass
        ax3.set_title('ROC Curves')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        
        # Feature Importance Comparison (top 10 features)
        ax4 = axes[1, 1]
        if self.feature_importance:
            # Get top 5 features from the best performing model
            best_model = comparison_df.loc['ROC-AUC'].idxmax().lower().replace(' ', '_')
            if best_model in self.feature_importance:
                top_features = self.feature_importance[best_model].head(10)
                ax4.barh(range(len(top_features)), top_features['importance'])
                ax4.set_yticks(range(len(top_features)))
                ax4.set_yticklabels(top_features['feature'])
                ax4.set_title(f'Top Features - {best_model.replace("_", " ").title()}')
                ax4.set_xlabel('Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrices(self, save_path=None):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def save_models(self, save_dir):
        """Save all trained models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.best_models.items():
            model_path = os.path.join(save_dir, f"{model_name}_model.joblib")
            joblib.dump(model, model_path)
            print(f"Model {model_name} saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a saved model"""
        return joblib.load(model_path)
    
    def predict_churn_probability(self, model_name, X):
        """Predict churn probability for new data"""
        if model_name not in self.best_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.best_models.keys())}")
        
        model = self.best_models[model_name]
        probabilities = model.predict_proba(X)[:, 1]
        
        return probabilities
    
    def get_model_summary(self):
        """Get summary of all trained models"""
        summary = {
            'models_trained': list(self.best_models.keys()),
            'best_model': None,
            'results': self.results
        }
        
        # Find best model based on ROC-AUC
        best_roc_auc = 0
        best_model = None
        
        for model_name, result in self.results.items():
            roc_auc = result['metrics']['roc_auc']
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model = model_name
        
        summary['best_model'] = best_model
        summary['best_roc_auc'] = best_roc_auc
        
        return summary

def main():
    """Example usage of ChurnPredictor"""
    # This would typically be called after loading and preprocessing data
    print("ChurnPredictor class initialized successfully!")
    print("Use this class with your preprocessed data to train and evaluate models.")

if __name__ == "__main__":
    main()