import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import joblib
import warnings
warnings.filterwarnings('ignore')


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Apply logarithmic transformation to skewed features to make their distribution more normal.
    
    This is particularly useful for features like house prices or habitable surface area
    that often follow a right-skewed distribution.
    
    Parameters:
        features (Optional[List[str]]): List of feature names to apply transformation to.
        offset (float): Small value to add to avoid log(0). Default is 1.
    """
    def __init__(self, features=None, offset=1):
        self.features = features
        self.offset = offset

    def fit(self, X, y=None):
        """
        Fit the transformer (no actual fitting needed, returns self).
        
        Args:
            X: Input features DataFrame
            y: Target values (optional)
        
        Returns:
            self: Returns the transformer instance
        """
        return self
    
    def transform(self, X):
        """
        Apply logarithmic transformation to specified features.
        
        Args:
            X: Input features DataFrame
        
        Returns:
            Transformed DataFrame with logarithmic features
        """
        X_trans = X.copy()
        if self.features:
            for feature in self.features:
                if feature in X_trans.columns:
                    # Apply log(1 + x) transformation to handle zero values
                    X_trans[feature] = np.log1p(X_trans[feature])
        return X_trans

    def inverse_transform(self, X):
        """
        Reverse the logarithmic transformation.
            
        Args:
            X: Transformed features DataFrame
            
        Returns:
            Original scale features DataFrame
        """
        X_inv = X.copy()
        if self.features:
            for feature in self.features:
                if feature in X_inv.columns:
                    # Apply exponential minus one to reverse log1p
                    X_inv[feature] = np.expm1(X_inv[feature])
        return X_inv


def prepare_data(
        data_path: str,
        target_column: str = 'price',
        test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Prepare data for house price prediction modeling.
    
    This function:
    1. Loads the cleaned dataset
    2. Separates features and target
    3. Splits data into training and test sets
    4. Applies log transformation to the target variable (house prices)
    
    Args:
        data_path: Path to the cleaned dataset CSV file
        target_column: Name of the target price column
        test_size: Proportion of data to use for testing
    
    Returns:
        Tuple containing:
        - X_train: Training features
        - X_test: Test features
        - y_train: Original training target
        - y_test: Original test target
        - y_train_log: Log-transformed training target
        - y_test_log: Log-transformed test target
    """
    # Load cleaned data
    df = pd.read_csv(data_path)
    print(f'Loaded data with shape: {df.shape}')

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f'Train set shape: {X_train.shape}, Test set shape: {X_test.shape}')

    # Apply log transformation to target (house prices)
    # This helps handle the typical right-skew in price distributions
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    return X_train, X_test, y_train, y_test, y_train_log, y_test_log


def identify_skewed_features(X, skew_threshold=0.5):
    """
    Identify features with high skewness that might benefit from log transformation.
    
    Args:
        X: Features DataFrame
        skew_threshold: Threshold for considering a feature as skewed
    
    Returns:
        List of column names with high skewness
    """
    nemeric_features = X.select_dtypes(include=['int64','float64']).columns
    skewed_features = []
    for col in nemeric_features :
        skewness = X[col].skew()
        if abs(skewness) > skew_threshold :
            skewed_features.append(col)
            print(f"Feature '{col}' has skewness: {skewness:.2f}")

    return skewed_features

def creat_preprocessing_pipeline(
        numeric_features:list,
        skewned_features:list =None,
        use_robust_scaler:bool=True
) -> ColumnTransformer :
    """
    Create a preprocessing pipeline for feature transformation.
    
    Args:
        numeric_features: List of numeric feature names
        skewed_features: List of features to apply log transformation
        use_robust_scaler: Whether to use RobustScaler (better for outliers)
    
    Returns:
        ColumnTransformer with preprocessing steps
    """
    # Choose scaler
    scaler = RobustScaler() if use_robust_scaler else StandardScaler()

    # creat transfromers list
    transformers = []

    # Add log transformer for skewed features if provided
    if skewned_features:
        log_transformer = LogTransformer(features = skewned_features)
        transformers.append('log',log_transformer,skewned_features)

    # Add scaler for all numeric features
    transformers.append('scaler',scaler,numeric_features)

    # Creat and return the column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder= 'passthrough' #keep other columns as is
    )

    return preprocessor

def train_models(
        X_train:pd.DataFrame,
        X_test:pd.DataFrame,
        y_train:pd.Series,
        y_test:pd.Series,
        preprocessor:ColumnTransformer = None
) ->dict:
    """
    Train multiple regression models and compare their performance.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target (log-transformed)
        y_test: Test target (log-transformed)
        preprocessor: Preprocessing pipeline
    
    Returns:
        Dictionary with model results
    """
    # define models to test
    models = {
        'Linear Regression':LinearRegression(),
        'Decision Tree':DecisionTreeRegressor(max_depth=10,random_state=42),
        'Random Forest':RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f'\nTraining{name}...')

        # Creat pipeline if preprocessor provided
        if preprocessor:
            Pipeline = Pipeline([
                ('preprocessor',preprocessor),
                ('model',model)
            ])
            model_to_fit = Pipeline
        else:
            model_to_fit = model

        # Train model
        model_to_fit.fit(X_train,y_train)

        # make prediction
        y_pred = model_to_fit.predict(X_test)

        # Calculate metrics(in log space)
        mae_log = mean_absolute_error(y_test,y_pred)
        rmse_log = np.sqrt(mean_squared_error(y_test,y_pred))
        r2 = r2_score(y_test,y_pred)

        # convert predictions back to original scale for interpretability
        y_pred_original = np.expm1(y_pred)
        y_test_original = np.expm1(y_test)
        mae_original = mean_absolute_error(y_test_original,y_pred_original)
        rmse_original = np.sqrt(mean_squared_error(y_test_original,y_pred_original))

        # Cross-validation
        cv_scores = cross_val_score(
            model_to_fit,X_train,y_train,
            cv=5,scoring='neg_mean_absolute_error'
        )

        results[name] = {
            'model':model_to_fit,
            'mae_log':mae_log,
            'rmse_log':rmse_log,
            'mae_original':mae_original,
            'rmse_original':rmse_original,
            'r2':r2,
            'cv_mae': -cv_scores.mean(),
            'predictions':y_pred
        }

def visualize_results(results:dict,y_test: pd.Series, save_dir:str = 'plots') :
    """
    Create visualizations for model evaluation.
    
    Args:
        results: Dictionary with model results
        y_test: Test target values (log-transformed)
        save_dir: Directory to save plots
    """
    