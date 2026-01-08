"""
ETL and Model Training Pipeline for Healthcare Readmission Prediction
This module handles data extraction, transformation, and model training for predicting
30-day hospital readmission rates.
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading from various sources."""
    
    def __init__(self, data_path: str = 'data/'):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data integrity.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        logger.info("Validating data...")
        
        # Check for empty dataframe
        if df.empty:
            logger.error("DataFrame is empty")
            return False
        
        # Check for required columns (customize based on your data)
        required_columns = [
            'patient_id', 'age', 'gender', 'admission_type',
            'length_of_stay', 'readmitted'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
        
        logger.info("Data validation completed")
        return True


class DataTransformer:
    """Handles data transformation and feature engineering."""
    
    def __init__(self):
        """Initialize DataTransformer."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('mean', 'median', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values using {strategy} strategy")
        df_copy = df.copy()
        
        missing_counts = df_copy.isnull().sum()
        if missing_counts.any():
            logger.info(f"Missing values:\n{missing_counts[missing_counts > 0]}")
            
            for col in df_copy.columns:
                if df_copy[col].isnull().any():
                    if strategy == 'mean' and df_copy[col].dtype in ['int64', 'float64']:
                        df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                    elif strategy == 'median' and df_copy[col].dtype in ['int64', 'float64']:
                        df_copy[col].fillna(df_copy[col].median(), inplace=True)
                    else:
                        df_copy[col].fillna(df_copy[col].mode()[0] if not df_copy[col].mode().empty else 'Unknown', inplace=True)
        
        logger.info("Missing values handled")
        return df_copy
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the encoders or use existing ones
            
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical features")
        df_copy = df.copy()
        
        categorical_columns = df_copy.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_columns:
            if col == 'patient_id':  # Skip patient ID
                continue
            
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_copy[col] = self.label_encoders[col].fit_transform(df_copy[col].astype(str))
                logger.info(f"Encoded {col}: {len(self.label_encoders[col].classes_)} unique values")
            else:
                if col in self.label_encoders:
                    df_copy[col] = self.label_encoders[col].transform(df_copy[col].astype(str))
        
        return df_copy
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for the model.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        logger.info("Creating derived features")
        df_copy = df.copy()
        
        # Example feature engineering (customize based on your domain knowledge)
        if 'age' in df_copy.columns:
            df_copy['age_group'] = pd.cut(df_copy['age'], bins=[0, 30, 50, 70, 100], labels=[1, 2, 3, 4])
            df_copy['age_group'] = df_copy['age_group'].astype(int)
        
        if 'length_of_stay' in df_copy.columns:
            df_copy['long_stay'] = (df_copy['length_of_stay'] > df_copy['length_of_stay'].median()).astype(int)
        
        # Add interaction features if relevant columns exist
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Created features. Total features: {len(numeric_columns)}")
        
        return df_copy
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X: Feature matrix
            fit: Whether to fit the scaler or use existing one
            
        Returns:
            Scaled feature matrix
        """
        logger.info("Scaling features")
        
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def transform(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Execute complete transformation pipeline.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers or use existing ones
            
        Returns:
            Tuple of (transformed features array, processed dataframe)
        """
        logger.info("Starting transformation pipeline")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Create derived features
        df = self.create_features(df)
        
        # Separate features and target
        target_col = 'readmitted'
        if target_col in df.columns:
            X = df.drop([target_col, 'patient_id'], axis=1, errors='ignore')
            self.feature_names = X.columns.tolist()
        else:
            X = df.drop(['patient_id'], axis=1, errors='ignore')
            self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scale_features(X.values, fit=fit)
        
        logger.info("Transformation pipeline completed")
        return X_scaled, df


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """
        Train logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained LogisticRegression model
        """
        logger.info("Training Logistic Regression model")
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(X_train, y_train)
        self.models['LogisticRegression'] = model
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Train random forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained RandomForestClassifier model
        """
        logger.info("Training Random Forest model")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['RandomForest'] = model
        return model
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
        """
        Train gradient boosting model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained GradientBoostingClassifier model
        """
        logger.info("Training Gradient Boosting model")
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.models['GradientBoosting'] = model
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train all available models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training all models")
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        logger.info(f"Training completed. Total models: {len(self.models)}")
        return self.models
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str = "Model") -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'precision': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision'],
            'recall': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall']
        }
        
        self.metrics[model_name] = metrics
        logger.info(f"{model_name} Metrics: {metrics}")
        
        return metrics
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics for all models
        """
        logger.info("Evaluating all models")
        
        for model_name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
        
        # Select best model based on ROC-AUC score
        best_model_name = max(self.metrics, key=lambda x: self.metrics[x]['roc_auc'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        logger.info(f"Best model: {best_model_name} with ROC-AUC: {self.metrics[best_model_name]['roc_auc']:.4f}")
        
        return self.metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save best model to disk.
        
        Args:
            filepath: Path to save the model
        """
        logger.info(f"Saving best model ({self.best_model_name}) to {filepath}")
        
        model_dir = Path(filepath).parent
        model_dir.mkdir(exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        logger.info("Model saved successfully")


class Pipeline:
    """Main ETL and training pipeline."""
    
    def __init__(self, data_path: str = 'data/', models_path: str = 'models/'):
        """
        Initialize Pipeline.
        
        Args:
            data_path: Path to data directory
            models_path: Path to models directory
        """
        self.data_path = data_path
        self.models_path = models_path
        self.loader = DataLoader(data_path)
        self.transformer = DataTransformer()
        self.trainer = ModelTrainer()
        
        Path(models_path).mkdir(exist_ok=True)
    
    def run(self, data_filepath: str, test_size: float = 0.2, 
            save_model: bool = True) -> Dict[str, Any]:
        """
        Run complete ETL and training pipeline.
        
        Args:
            data_filepath: Path to input data CSV
            test_size: Fraction of data to use for testing
            save_model: Whether to save the trained model
            
        Returns:
            Dictionary containing pipeline results
        """
        logger.info("=" * 80)
        logger.info("Starting Healthcare Readmission Prediction Pipeline")
        logger.info("=" * 80)
        
        try:
            # Step 1: Load data
            logger.info("\nStep 1: Loading Data")
            df = self.loader.load_data(data_filepath)
            
            if not self.loader.validate_data(df):
                raise ValueError("Data validation failed")
            
            # Step 2: Transform data
            logger.info("\nStep 2: Transforming Data")
            X, df_transformed = self.transformer.transform(df, fit=True)
            
            # Extract target variable
            y = df_transformed['readmitted'].values if 'readmitted' in df_transformed.columns else None
            
            if y is None:
                raise ValueError("Target variable 'readmitted' not found in data")
            
            # Step 3: Train-test split
            logger.info("\nStep 3: Splitting Data")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            logger.info(f"Training set: {X_train.shape[0]} samples")
            logger.info(f"Test set: {X_test.shape[0]} samples")
            
            # Step 4: Train models
            logger.info("\nStep 4: Training Models")
            self.trainer.train_all_models(X_train, y_train)
            
            # Step 5: Evaluate models
            logger.info("\nStep 5: Evaluating Models")
            metrics = self.trainer.evaluate_all_models(X_test, y_test)
            
            # Step 6: Save model
            if save_model:
                logger.info("\nStep 6: Saving Model")
                model_path = os.path.join(self.models_path, 'best_model.pkl')
                self.trainer.save_model(model_path)
            
            # Prepare results
            results = {
                'timestamp': datetime.now().isoformat(),
                'data_shape': df.shape,
                'train_shape': X_train.shape,
                'test_shape': X_test.shape,
                'features': self.transformer.feature_names,
                'best_model': self.trainer.best_model_name,
                'metrics': self.trainer.metrics,
                'model_path': os.path.join(self.models_path, 'best_model.pkl') if save_model else None
            }
            
            # Save results to JSON
            results_path = os.path.join(self.models_path, 'results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            logger.info("\n" + "=" * 80)
            logger.info("Pipeline Completed Successfully!")
            logger.info("=" * 80)
            
            return results
        
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main execution function."""
    # Initialize pipeline
    pipeline = Pipeline(
        data_path='data/',
        models_path='models/'
    )
    
    # Run pipeline
    # Note: Update 'data/readmission_data.csv' with your actual data file path
    results = pipeline.run(
        data_filepath='data/readmission_data.csv',
        test_size=0.2,
        save_model=True
    )
    
    # Print results summary
    print("\n" + "=" * 80)
    print("PIPELINE RESULTS SUMMARY")
    print("=" * 80)
    print(f"Best Model: {results['best_model']}")
    print(f"\nModel Metrics:")
    for model_name, metrics in results['metrics'].items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
