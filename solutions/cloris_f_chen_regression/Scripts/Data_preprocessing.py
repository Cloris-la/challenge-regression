import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from scipy import stats
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
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """
        Fit the transformer (no actual fitting needed, returns self).
        
        Args:
            X: Input features DataFrame
            y: Target values (optional)
        
        Returns:
            self: Returns the transformer instance
        """
        # Store feature names if X is a DataFrame
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self
    
    def transform(self, X):
        """
        Apply logarithmic transformation to specified features.
        
        Args:
            X: Input features DataFrame
        
        Returns:
            Transformed DataFrame with logarithmic features
        """
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names_in_ if self.feature_names_in_ else None)
        else:
            X_df = X.copy()
            
        if self.features:
            for feature in self.features:
                if feature in X_df.columns:
                    # Apply log(1 + x) transformation to handle zero values
                    X_df[feature] = np.log1p(X_df[feature])
                    
        # Return as numpy array for sklearn compatibility
        return X_df.values if isinstance(X, np.ndarray) else X_df

    def inverse_transform(self, X):
        """
        Reverse the logarithmic transformation.
            
        Args:
            X: Transformed features DataFrame
            
        Returns:
            Original scale features DataFrame
        """
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names_in_ if self.feature_names_in_ else None)
        else:
            X_df = X.copy()
            
        if self.features:
            for feature in self.features:
                if feature in X_df.columns:
                    # Apply exponential minus one to reverse log1p
                    X_df[feature] = np.expm1(X_df[feature])
                    
        return X_df.values if isinstance(X, np.ndarray) else X_df
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.
        
        Args:
            input_features: Input feature names
            
        Returns:
            Output feature names
        """
        if input_features is not None:
            return input_features
        elif self.feature_names_in_ is not None:
            return self.feature_names_in_
        else:
            return None


def analyze_dataset(df, target_column='price'):
    """
    Analyze the dataset and print summary statistics.
    
    Args:
        df: DataFrame to analyze
        target_column: Name of the target column
    
    Returns:
        Dictionary with feature information
    """
    print('\n' + '='*60)
    print('DATASET ANALYSIS')
    print('='*60)

    # Print basic information of dataset
    print(f'\nDataset shape: {df.shape}')
    print(f'Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB')

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify types of features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # These are your already encoded categorical features
    encoded_categorical_features = [
        'postCode', 'province_encoded', 'type_encoded', 'subtype_encoded', 'locality_encoded'
    ]
    
    # Filter out encoded categorical from numeric features
    encoded_categorical_features = [col for col in encoded_categorical_features if col in X.columns]
    true_numeric_features = [
        col for col in numeric_features if col not in encoded_categorical_features
    ]

    # Identify binary and count features
    binary_features = []
    count_features = []
    
    for col in true_numeric_features:
        unique_vals = X[col].nunique()
        if unique_vals == 2 and set(X[col].unique()).issubset({0, 1}):
            binary_features.append(col)
        else:
            count_features.append(col)

    print(f"\nFeature types:")
    print(f"  - Total features: {len(X.columns)}")
    print(f"  - Numeric features: {len(numeric_features)}")
    print(f"  - True numeric features: {len(true_numeric_features)}")
    print(f"  - Binary features: {len(binary_features)}")
    print(f"  - Count features: {len(count_features)}")
    print(f"  - Categorical features: {len(categorical_features)}")
    print(f"  - Encoded categorical features: {len(encoded_categorical_features)}")

    if categorical_features:
        print(f"\nOriginal categorical features:")
        for feature in categorical_features:
            print(f"  - {feature} (unique values: {X[feature].nunique()})")
    
    if encoded_categorical_features:
        print(f"\nEncoded categorical features:")
        for feature in encoded_categorical_features:
            if feature in X.columns:
                print(f"  - {feature} (unique values: {X[feature].nunique()})")

    # Target statistics
    print(f"\nTarget variable '{target_column}' statistics:")
    print(f"  - Min: €{y.min():,.0f}")
    print(f"  - Max: €{y.max():,.0f}")
    print(f"  - Mean: €{y.mean():,.0f}")
    print(f"  - Median: €{y.median():,.0f}")
    print(f"  - Std: €{y.std():,.0f}")

    return {
        'numeric_features': numeric_features,
        'true_numeric_features': true_numeric_features,
        'binary_features': binary_features,
        'count_features': count_features,
        'categorical_features': categorical_features,
        'encoded_categorical_features': encoded_categorical_features,
        'all_features': X.columns.tolist()
    }


def identify_skewed_features(X, skew_threshold=0.5):
    """
    Identify features with high skewness that might benefit from log transformation.
    
    Args:
        X: Features DataFrame
        skew_threshold: Threshold for considering a feature as skewed
    
    Returns:
        List of column names with high skewness
    """
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    skewed_features = []

    print(f"\nChecking skewness (threshold = {skew_threshold}):")
    
    for col in numeric_features:
        # Skip binary features
        if X[col].nunique() == 2:
            continue

        skewness = X[col].skew()
        if abs(skewness) > skew_threshold:
            skewed_features.append(col)
            print(f"  - Feature '{col}' has skewness: {skewness:.2f}")
    
    if not skewed_features:
        print("  - No highly skewed features found")

    return skewed_features


def create_preprocessing_pipeline(
        true_numeric_features: list,
        categorical_features: list,
        encoded_categorical_features: list,
        skewed_features: list = None,
        use_robust_scaler: bool = True
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for feature transformation.
    
    Args:
        true_numeric_features: List of true numeric feature names
        categorical_features: List of categorical feature names
        encoded_categorical_features: List of already encoded categorical features
        skewed_features: List of features to apply log transformation
        use_robust_scaler: Whether to use RobustScaler (better for outliers)
    
    Returns:
        ColumnTransformer with preprocessing steps
    """
    # Choose scaler
    scaler = RobustScaler() if use_robust_scaler else StandardScaler()

    # Create transformers list
    transformers = []

    # Add log transformer for skewed features if provided
    if skewed_features:
        valid_skewed_features = [
            f for f in skewed_features if f in true_numeric_features
        ]
        if valid_skewed_features:
            log_transformer = LogTransformer(features=valid_skewed_features)
            transformers.append(('log', log_transformer, valid_skewed_features))

    # Scaling for real numeric features
    if true_numeric_features:
        transformers.append(('scaler', scaler, true_numeric_features))

    # One-hot encode for category features (if any)
    if categorical_features:
        onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformers.append(('onehot', onehot_transformer, categorical_features))

    # Create and return the column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough',  # Keep other columns as is (like encoded categorical)
        verbose_feature_names_out=False  # Simplify feature names
    )

    return preprocessor


def visualize_preprocessing_effects(y_train, y_train_log, save_dir="plots"):
    """
    Visualize the effects of preprocessing on the target variable.
    
    Args:
        y_train: Original target values
        y_train_log: Log-transformed target values
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original distribution
    axes[0, 0].hist(y_train, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Price Distribution - Original', fontsize=14)
    axes[0, 0].set_xlabel('Price (€)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(y_train.mean(), color='red', linestyle='--', 
                       label=f'Mean: €{y_train.mean():,.0f}')
    axes[0, 0].axvline(y_train.median(), color='green', linestyle='--',
                       label=f'Median: €{y_train.median():,.0f}')
    axes[0, 0].legend()

    # Log-transformed distribution
    axes[0, 1].hist(y_train_log, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
    axes[0, 1].set_title('Price Distribution - Log Transformed', fontsize=14)
    axes[0, 1].set_xlabel('Log(Price + 1)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(y_train_log.mean(), color='red', linestyle='--',
                       label=f'Mean: {y_train_log.mean():.2f}')
    axes[0, 1].axvline(y_train_log.median(), color='green', linestyle='--',
                       label=f'Median: {y_train_log.median():.2f}')
    axes[0, 1].legend()

    # Q-Q plot for original
    stats.probplot(y_train, dist='norm', plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot - Original Prices')
    axes[1, 0].grid(True, alpha=0.3)

    # Q-Q plot for transformed
    stats.probplot(y_train_log, dist='norm', plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot - Log Transformed Prices')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'preprocessing_effects.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\nPreprocessing visualization saved to {save_dir}/preprocessing_effects.png')


def prepare_data(
        data_path: str,
        target_column: str = 'price',
        test_size: float = 0.2,
        use_robust_scaler: bool = True,
        apply_log_to_features: bool = True,
        visualize: bool = True
) -> dict:
    """
    Prepare data for house price prediction modeling.
    
    This function:
    1. Loads the cleaned dataset
    2. Analyzes the dataset
    3. Separates features and target
    4. Splits data into training and test sets
    5. Creates preprocessing pipeline
    6. Applies log transformation to the target variable
    
    Args:
        data_path: Path to the cleaned dataset CSV file
        target_column: Name of the target price column
        test_size: Proportion of data to use for testing
        use_robust_scaler: Whether to use RobustScaler
        apply_log_to_features: Whether to apply log transformation to skewed features
        visualize: Whether to create visualizations
    
    Returns:
        Dictionary containing all prepared data and metadata
    """
    print("="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    # Load cleaned data
    print("\n1. Loading data...")
    df = pd.read_csv(data_path)
    print(f"   ✓ Loaded data with shape: {df.shape}")

    # Analyze dataset
    print("\n2. Analyzing dataset...")
    feature_info = analyze_dataset(df, target_column)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split data into training and test sets
    print(f"\n3. Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )
    print(f"   ✓ Train set shape: {X_train.shape}")
    print(f"   ✓ Test set shape: {X_test.shape}")
    
    # Identify skewed features
    print("\n4. Identifying skewed features...")
    skewed_features = identify_skewed_features(X_train) if apply_log_to_features else []
    
    # Create preprocessing pipeline
    print("\n5. Creating preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline(
        true_numeric_features=feature_info['true_numeric_features'],
        categorical_features=feature_info['categorical_features'],
        encoded_categorical_features=feature_info['encoded_categorical_features'],
        skewed_features=skewed_features,
        use_robust_scaler=use_robust_scaler
    )
    
    # Fit and transform features
    print("\n6. Applying preprocessing to features...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after transformation
    try:
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [str(name) for name in feature_names]  # Ensure all are strings
    except:
        # Fallback if get_feature_names_out doesn't work
        feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
    
    # Convert back to DataFrame for easier handling
    X_train_processed = pd.DataFrame(X_train_processed, index=X_train.index, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, index=X_test.index, columns=feature_names)
    
    print(f"   ✓ Feature preprocessing completed")
    print(f"   ✓ Processed shape: {X_train_processed.shape}")

    # Apply log transformation to target (house prices)
    print("\n7. Applying log transformation to target variable...")
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    print(f"   ✓ Target transformation completed")
    
    # Visualize preprocessing effects
    if visualize:
        print("\n8. Creating visualizations...")
        visualize_preprocessing_effects(
            y_train, 
            y_train_log,
            save_dir=r'D:\Projects\Bouman9projects\challenge-regression\solutions\cloris_f_chen_regression\plots'
            )
    
    # Prepare return dictionary
    prepared_data = {
        # Processed data
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_log': y_train_log,
        'y_test_log': y_test_log,
        
        # Metadata
        'feature_names': feature_names,
        'feature_info': feature_info,
        'skewed_features': skewed_features,
        'preprocessor': preprocessor,
        
        # Configuration
        'config': {
            'test_size': test_size,
            'use_robust_scaler': use_robust_scaler,
            'apply_log_to_features': apply_log_to_features
        }
    }
    
    print("\n" + "="*60)
    print("DATA PREPROCESSING COMPLETED!")
    print("="*60)
    
    return prepared_data


def save_prepared_data(prepared_data, save_dir=r'D:\Projects\Bouman9projects\challenge-regression\solutions\cloris_f_chen_regression\Scripts'):
    """
    Save the prepared data to disk.
    
    Args:
        prepared_data: Dictionary with all prepared data
        save_path: Path to save the data
    """
    # make sure the folder exists
    os.makedirs(save_dir, exist_ok=True) 

    prepared_data_path = os.path.join(save_dir, 'prepared_data.pkl')
    preprocessor_path = os.path.join(save_dir, 'preprocessor.pkl')

    joblib.dump(prepared_data, prepared_data_path)
    print(f"\nPrepared data saved to '{prepared_data_path}'")

    joblib.dump(prepared_data['preprocessor'], preprocessor_path)
    print(f"Preprocessor saved to '{preprocessor_path}'")


def main():
    """
    Main function to run data preprocessing.
    """
    # Configuration
    data_path = r'D:\Projects\Bouman9projects\challenge-regression\solutions\cloris_f_chen_regression\data\cleaned_dataset.csv'
    
    # Prepare data with default settings
    prepared_data = prepare_data(
        data_path=data_path,
        target_column='price',
        test_size=0.2,
        use_robust_scaler=True,
        apply_log_to_features=True,
        visualize=True
    )
    
    # Save prepared data
    save_prepared_data(prepared_data)
    
    # Print summary
    print("\nPREPROCESSING SUMMARY:")
    print(f"  - Training samples: {len(prepared_data['y_train'])}")
    print(f"  - Test samples: {len(prepared_data['y_test'])}")
    print(f"  - Number of features: {prepared_data['X_train'].shape[1]}")
    print(f"  - Skewed features transformed: {len(prepared_data['skewed_features'])}")
    
    return prepared_data


if __name__ == "__main__":
    prepared_data = main()