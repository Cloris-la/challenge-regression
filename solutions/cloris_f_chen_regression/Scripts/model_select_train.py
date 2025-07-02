from Data_preprocessing import LogTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import joblib
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optuna for hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# Linear models
from sklearn.linear_model import (
    LinearRegression, Ridge, BayesianRidge
)

# Tree-based models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)

# Other models
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# Define all output path
BASE_DIR = "D:/Projects/Bouman9projects/challenge-regression/solutions/cloris_f_chen_regression"
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# Define model dictionary
def get_model_dictionary():
    """
    Get a comprehensive dictionary of regression models to test.
    
    Args:
        include_advanced: Whether to include XGBoost, LightGBM, CatBoost
    
    Returns:
        Dictionary of model names and instances
    """
    models = {
        # Linear models
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Bayesian Ridge': BayesianRidge(),
        
        # Tree-based models
        'Decision Tree': DecisionTreeRegressor(
            max_depth=10, min_samples_split=20, random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=10,
            random_state=42, n_jobs=-1
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=100, max_depth=10, min_samples_split=10,
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42
        ),
        'AdaBoost': AdaBoostRegressor(
            n_estimators=100, learning_rate=1.0, random_state=42
        ),
        
        # Other models
        'KNN-5': KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1),
        'KNN-10': KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1),
        'XGBoost': XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=100, depth=6, learning_rate=0.1,
            random_state=42, verbose=False,
            allow_writing_files=False
        )
        }
    
    return models


def train_single_model(name, model, X_train, y_train, X_test, y_test, cv_folds=5):
    """
    Train and evaluate a single model.
    
    Args:
        name: Model name
        model: Model instance
        X_train, y_train: Training data
        X_test, y_test: Test data
        cv_folds: Number of cross-validation folds
    
    Returns:
        Dictionary with evaluation results
    """
    print(f'\nTraining {name}...')
    start_time = time.time()

    try:
        # Train model
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics (in log space)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)

        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)

        # Convert predictions back to original scale for interpretability
        y_test_original = np.expm1(y_test)
        y_pred_original = np.expm1(y_pred_test)
        mae_original = mean_absolute_error(y_test_original, y_pred_original)
        rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        
        # Calculate residuals
        residuals = y_test_original - y_pred_original

        # Cross-validation (skip for slow models)
        cv_scores = None
        cv_mae = None
        if train_time < 5 and cv_folds > 0:
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv_folds, scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            cv_scores = np.sqrt(-cv_scores)
            cv_mae = -cross_val_score(
                model, X_train, y_train,
                cv=cv_folds, scoring='neg_mean_absolute_error',
                n_jobs=-1
            ).mean()

        results = {
            'model': model,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'y_test_original': y_test_original,
            'y_pred_original': y_pred_original,
            'mae_original': mae_original,
            'rmse_original': rmse_original,
            'residuals': residuals,
            'y_pred_train': y_pred_train,
            'predictions': y_pred_test,
            'cv_scores': cv_scores,
            'cv_mae': cv_mae,
            'train_time': train_time
        }

        # Print results
        print("="*50)
        print(f"Model: {name}")

        print("\n🔹 Training Results (log space):")
        print(f"  MAE : {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  R²  : {train_r2:.4f}")

        print("\n🔹 Test Results (log space):")
        print(f"  MAE : {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  R²  : {test_r2:.4f}")

        print("\n🔹 Test Results (original scale):")
        print(f"  MAE : €{mae_original:,.0f}")
        print(f"  RMSE: €{rmse_original:,.0f}")

        if cv_scores is not None:
            print("\n🔹 Cross-Validation:")
            print(f"  CV RMSE: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"  CV MAE : {cv_mae:.4f}")

        print(f"\n🔹 Training Time: {train_time:.2f} seconds")
        print("="*50)
        
        return results
        
    except Exception as e:
        print(f"  ✗ Error training {name}: {str(e)}")
        return None


def train_models(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        cv_folds: int = 5,
) -> dict:
    """
    Train multiple regression models and compare their performance.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target (log-transformed)
        y_test: Test target (log-transformed)
        cv_folds: Number of cross-validation folds
    
    Returns:
        Dictionary with model results
    """
    print('\n' + '='*60)
    print('MODEL TRAINING AND EVALUATION')
    print('='*60)

    # Get models
    models = get_model_dictionary()
    print(f"\nTotal models to train: {len(models)}")

    # Train all models
    results = {}
    for i, (name, model) in enumerate(models.items(), 1):
        print(f'\n[{i}/{len(models)}]', end='')
        result = train_single_model(name, model, X_train, y_train, X_test, y_test, cv_folds)
        if result:
            results[name] = result
    
    return results


def find_best_model(results, metric='mae_original'):
    """
    Find the best model based on specified metric.

    Args:
        results: Dictionary of model results
        metric: Metric to use for comparison

    Returns:
        Tuple of (best_model_name, best_model_results)
    """
    if not results:
        return None, None
    
    # Sort by specified metric (original scale)
    sorted_models = sorted(results.items(), key=lambda x: x[1][metric])
    best_model_name = sorted_models[0][0]
    best_model_result = sorted_models[0][1]

    print('\n' + '='*60)
    print('MODEL COMPARISON')
    print('='*60)
    print(f"\n{'Rank':<5} {'Model':<25} {'MAE (€)':<12} {'RMSE (€)':<12} {'R²':<8} {'Time (s)':<10}")
    print("-"*80)

    # Print all model results
    for idx, (name, res) in enumerate(sorted_models, 1):
        print(f"{idx:<5} {name:<25} €{res['mae_original']:<11,.0f} €{res['rmse_original']:<11,.0f} "
              f"{res['test_r2']:<8.3f} {res['train_time']:<10.2f}")
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print("="*60)
    print(f"MAE (original scale): €{best_model_result['mae_original']:,.0f}")
    print(f"RMSE (original scale): €{best_model_result['rmse_original']:,.0f}")
    print(f"R² Score: {best_model_result['test_r2']:.3f}")
    print(f"Training Time: {best_model_result['train_time']:.2f} seconds")

    return best_model_name, best_model_result


def visualize_results(results: dict, y_test: pd.Series, save_dir = PLOTS_DIR):
    """
    Create comprehensive visualizations for model evaluation.
    
    Args:
        results: Dictionary with model results
        y_test: Test target values (log-transformed)
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data for visualization
    model_names = list(results.keys())
    mae_values = [results[m]['mae_original'] for m in model_names]
    rmse_values = [results[m]['rmse_original'] for m in model_names]
    r2_values = [results[m]['test_r2'] for m in model_names]
    train_times = [results[m]['train_time'] for m in model_names]

    # Sort models by MAE for better visualization
    sorted_indices = np.argsort(mae_values)
    model_names_sorted = [model_names[i] for i in sorted_indices]
    mae_sorted = [mae_values[i] for i in sorted_indices]
    r2_sorted = [r2_values[i] for i in sorted_indices]

    # Create main comparison figure
    fig = plt.figure(figsize=(20, 15))

    # 1. Model MAE comparison (horizontal bar chart)
    ax1 = plt.subplot(3, 2, 1)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(model_names_sorted)))
    bars = ax1.barh(model_names_sorted, mae_sorted, color=colors)
    ax1.set_xlabel('Mean Absolute Error (€)', fontsize=12)
    ax1.set_title('Model Comparison by MAE (Lower is Better)', fontsize=14)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, mae_sorted)):
        ax1.text(value + max(mae_sorted)*0.01, i, f'€{value:,.0f}', 
                va='center', fontsize=10)
    
    # 2. Model R² comparison
    ax2 = plt.subplot(3, 2, 2)
    colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(model_names_sorted)))
    bars2 = ax2.barh(model_names_sorted, r2_sorted, color=colors2)
    ax2.set_xlabel('R² Score', fontsize=12)
    ax2.set_title('Model Comparison by R² (Higher is Better)', fontsize=14)
    ax2.set_xlim(0, 1)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars2, r2_sorted)):
        ax2.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=10)
    
    # 3. Best model predictions vs actual
    best_model = min(results, key=lambda x: results[x]['mae_original'])
    y_pred_best = results[best_model]['predictions']
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred_best)
    
    ax3 = plt.subplot(3, 2, 3)
    scatter = ax3.scatter(y_test_original, y_pred_original, alpha=0.5, s=30,
                         c=y_test_original, cmap='viridis')
    ax3.plot([y_test_original.min(), y_test_original.max()], 
            [y_test_original.min(), y_test_original.max()], 
            'r--', lw=2, label='Perfect prediction')
    ax3.set_xlabel('Actual Price (€)', fontsize=12)
    ax3.set_ylabel('Predicted Price (€)', fontsize=12)
    ax3.set_title(f'Best Model ({best_model}): Predictions vs Actual', fontsize=14)
    ax3.legend()
    plt.colorbar(scatter, ax=ax3, label='Actual Price (€)')
    
    # 4. Residual distribution
    residuals = results[best_model]['residuals']
    ax4 = plt.subplot(3, 2, 4)
    ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='lightblue')
    ax4.axvline(0, color='red', linestyle='--', lw=2, label='Zero residual')
    ax4.set_xlabel('Residual (€)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Residual Distribution', fontsize=14)
    ax4.legend()
    
    # Add residual statistics
    residual_stats = f'Mean: €{np.mean(residuals):,.0f}\nStd: €{np.std(residuals):,.0f}'
    ax4.text(0.7, 0.9, residual_stats, transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Training time vs performance
    ax5 = plt.subplot(3, 2, 5)
    scatter2 = ax5.scatter(train_times, mae_values, s=100, alpha=0.6,
                          c=r2_values, cmap='coolwarm')
    ax5.set_xlabel('Training Time (seconds)', fontsize=12)
    ax5.set_ylabel('MAE (€)', fontsize=12)
    ax5.set_title('Training Time vs Performance', fontsize=14)
    plt.colorbar(scatter2, ax=ax5, label='R² Score')
    
    # Add annotations for interesting models
    for name, time, mae, r2 in zip(model_names, train_times, mae_values, r2_values):
        if time < 1 or mae < np.percentile(mae_values, 20):
            ax5.annotate(name, (time, mae), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, alpha=0.7)
    
    # 6. Model performance summary table
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create summary table for top 10 models
    top_10_models = sorted(results.items(), key=lambda x: x[1]['mae_original'])[:10]
    table_data = []
    for name, res in top_10_models:
        table_data.append([
            name,
            f"€{res['mae_original']:,.0f}",
            f"{res['test_r2']:.3f}",
            f"{res['train_time']:.2f}s"
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Model', 'MAE', 'R²', 'Time'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax6.set_title('Top 10 Models Summary', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison_complete.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nModel comparison plots saved to {save_dir}/")
    
    # Create additional analysis plots
    create_residual_analysis(results, save_dir)


def create_residual_analysis(results, save_dir):
    """
    Create detailed residual analysis plots for top models.
    
    Args:
        results: Model results dictionary
        save_dir: Directory to save plots
    """

    os.makedirs(save_dir, exist_ok=True)

    # Get top 3 models
    top_3_models = sorted(results.items(), key=lambda x: x[1]['mae_original'])[:3]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (name, res) in enumerate(top_3_models):
        ax = axes[idx]
        
        # Residual vs predicted plot
        ax.scatter(res['y_pred_original'], res['residuals'], alpha=0.5, s=20)
        ax.axhline(y=0, color='red', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Price (€)')
        ax.set_ylabel('Residuals (€)')
        ax.set_title(f'{name}\nResidual Plot')
        
        # Add trend line
        z = np.polyfit(res['y_pred_original'], res['residuals'], 1)
        p = np.poly1d(z)
        ax.plot(sorted(res['y_pred_original']), p(sorted(res['y_pred_original'])), 
                "g--", alpha=0.8, lw=2)
        
        # Add statistics
        mae = res['mae_original']
        r2 = res['test_r2']
        ax.text(0.05, 0.95, f'MAE: €{mae:,.0f}\nR²: {r2:.3f}', 
                transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residual_analysis_top3.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Residual analysis plots saved to {save_dir}/")


def optimize_with_optuna(model_name, model, X_train, y_train, n_trials=50, cv_folds=3):
    """
    Optimize a model using Optuna hyperparameter tuning.
    
    Args:
        model_name: Name of the model
        model: Model instance
        X_train, y_train: Training data
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
    
    Returns:
        Optimized model
    """
    print(f"\nStarting Optuna optimization for {model_name}...")
    print(f"Number of trials: {n_trials}")
    
    # Define objective function for Optuna
    def objective(trial):
        # Suggest hyperparameters based on model type
        if 'Random Forest' in model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
            model.set_params(**params)
            
        elif 'Gradient Boosting' in model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            model.set_params(**params)
            
        elif 'XGBoost' in model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
            model.set_params(**params)
            
        elif 'LightGBM' in model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
            model.set_params(**params)
            
        elif 'CatBoost' in model_name:
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 1e-5, 10.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'border_count': trial.suggest_int('border_count', 32, 255)
            }
            model.set_params(**params)
            
        else:
            print(f"Optuna optimization not implemented for {model_name}")
            return model  # Skip optimization for unsupported models
        
        # Perform cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            scores.append(mae)
        
        # Return mean MAE across folds
        return np.mean(scores)
    
    # Create and run Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    # Print optimization results
    print(f"\nOptimization completed for {model_name}:")
    print(f"  Best trial value (MAE): {study.best_value:.4f}")
    print("  Best parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # Set best parameters to model
    model.set_params(**study.best_params)
    
    # Retrain on full training data with best parameters
    model.fit(X_train, y_train)
    
    return model


def optimize_best_model(best_model_name, best_model, X_train, y_train):
    """
    Perform hyperparameter optimization for the best model.
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    print(f"Optimizing {best_model_name}...")
    
    # Use Optuna for supported models
    optimized_model = optimize_with_optuna(
        best_model_name, 
        best_model,
        X_train,
        y_train,
        n_trials=50,
        cv_folds=3
    )
    return optimized_model


def generate_model_report(results, save_path=os.path.join(BASE_DIR, 'model_report.txt')):
    """
    Generate a comprehensive text report of model performance.
    
    Args:
        results: Dictionary of model results
        save_path: Path to save the report
    """

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("HOUSE PRICE PREDICTION - MODEL EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total models evaluated: {len(results)}\n")
        
        mae_values = [res['mae_original'] for res in results.values()]
        f.write(f"Best MAE: €{min(mae_values):,.0f}\n")
        f.write(f"Worst MAE: €{max(mae_values):,.0f}\n")
        f.write(f"Average MAE: €{np.mean(mae_values):,.0f}\n\n")
        
        # Model rankings
        f.write("MODEL RANKINGS BY MAE\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Rank':<6}{'Model':<30}{'MAE (€)':<15}{'R²':<10}{'Time (s)':<10}\n")
        f.write("-"*70 + "\n")
        
        sorted_models = sorted(results.items(), key=lambda x: x[1]['mae_original'])
        for i, (name, res) in enumerate(sorted_models, 1):
            f.write(f"{i:<6}{name:<30}€{res['mae_original']:<14,.0f}"
                   f"{res['test_r2']:<10.3f}{res['train_time']:<10.2f}\n")
        
        # Model category analysis
        f.write("\n\nMODEL CATEGORY ANALYSIS\n")
        f.write("-"*40 + "\n")
        
        categories = {
            'Linear': ['Linear', 'Ridge', 'Bayesian'],
            'Tree-based': ['Tree', 'Forest', 'Gradient', 'AdaBoost', 'Extra'],
            'Advanced': ['XGB', 'LightGBM', 'CatBoost'],
            'Other': ['KNN']
        }
        
        for category, keywords in categories.items():
            cat_models = [(name, res) for name, res in results.items() 
                         if any(kw in name for kw in keywords)]
            if cat_models:
                maes = [res['mae_original'] for _, res in cat_models]
                f.write(f"\n{category} Models:\n")
                f.write(f"  Count: {len(cat_models)}\n")
                f.write(f"  Best MAE: €{min(maes):,.0f}\n")
                f.write(f"  Average MAE: €{np.mean(maes):,.0f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\nModel report saved to '{save_path}'")


def save_results(results, best_model_name, save_dir=MODELS_DIR):
    """
    Save model results and the best model.
    
    Args:
        results: Dictionary of model results
        best_model_name: Name of the best model
        save_dir: Directory to save models
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save all results
    results_path = os.path.join(save_dir, 'all_model_results.pkl')
    joblib.dump(results, results_path)
    print(f"\nAll model results saved to '{results_path}'")
    
    # Save best model
    if best_model_name and best_model_name in results:
        best_model = results[best_model_name]['model']
        best_model_path = os.path.join(save_dir, 'best_model.pkl')
        joblib.dump({
            'model': best_model,
            'name': best_model_name,
            'metrics': {
                'mae': results[best_model_name]['mae_original'],
                'rmse': results[best_model_name]['rmse_original'],
                'r2': results[best_model_name]['test_r2']
            }
        }, best_model_path)
        print(f"Best model saved to '{best_model_path}'")


def main():
    """
    Main function to run model selection and evaluation.
    """
    print("="*60)
    print("MODEL SELECTION AND EVALUATION")
    print("="*60)
    
    print(f"\nOutput directories:")
    print(f"  Base: {BASE_DIR}")
    print(f"  Plots: {PLOTS_DIR}")
    print(f"  Models: {MODELS_DIR}")

    # Show current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\nScript directory: {script_dir}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Load prepared data
    print("\n1. Loading prepared data...")
    data_path = os.path.join(script_dir, 'prepared_data.pkl')
    
    try:
        prepared_data = joblib.load(data_path)
        print(f"   ✓ Data loaded successfully from {data_path}")
    except FileNotFoundError:
        print(f"   ✗ Error: '{data_path}' not found!")
        print("   Please run data_preprocessing.py first.")
        print("\n   Available .pkl files in script directory:")
        for file in os.listdir(script_dir):
            if file.endswith('.pkl'):
                print(f"     - {file}")
        return
    
    # Extract data
    X_train = prepared_data['X_train']
    X_test = prepared_data['X_test']
    y_train_log = prepared_data['y_train_log']
    y_test_log = prepared_data['y_test_log']
    
    print(f"\nData shapes:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")
    
    # Train models
    print("\n2. Training models...")
    results = train_models(
        X_train, X_test, y_train_log, y_test_log,
        cv_folds=5
    )
    
    if not results:
        print("   ✗ No models were successfully trained!")
        return
    
    # Find best model
    print("\n3. Finding best model...")
    best_model_name, best_model_results = find_best_model(results)
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    visualize_results(results, y_test_log)
    
    # Generate report
    print("\n5. Generating report...")
    generate_model_report(results)
    
    # Optimize best model (optional)
    if best_model_name and input("\nOptimize best model using Optuna? (y/n): ").lower() == 'y':
        optimized_model = optimize_best_model(
            best_model_name, 
            results[best_model_name]['model'],
            X_train, 
            y_train_log
        )
        if optimized_model:
            # Re-evaluate optimized model
            print("\nEvaluating optimized model...")
            optimized_results = train_single_model(
                f"{best_model_name} (Optimized)",
                optimized_model,
                X_train, y_train_log,
                X_test, y_test_log
            )
            if optimized_results:
                results[f"{best_model_name} (Optimized)"] = optimized_results
                # Check if optimized model is better
                if optimized_results['mae_original'] < best_model_results['mae_original']:
                    best_model_name = f"{best_model_name} (Optimized)"
                    best_model_results = optimized_results
                    print("\n✓ Optimized model performs better!")
    
    # Save results
    print("\n6. Saving results...")
    save_results(results, best_model_name)
    
    print("\n" + "="*60)
    print("MODEL SELECTION COMPLETED!")
    print("="*60)
    print(f"\nBest model: {best_model_name}")
    print(f"Best MAE: €{best_model_results['mae_original']:,.0f}")
    print(f"Best R²: {best_model_results['test_r2']:.3f}")
    
    return results


if __name__ == "__main__":
    results = main()