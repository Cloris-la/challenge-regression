import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import chi2_contingency
import warnings
# Global settings
warnings.filterwarnings('ignore')
# Use non-interactive mode to prevent blocking
plt.ioff()


def load_and_clean_data(file_path: str, save_path: str) -> pd.DataFrame:
    """
    Load data, remove duplicates, and save cleaned data.
    
    Args:
        file_path: Path to raw data file
        save_path: Path to save cleaned data
    
    Returns:
        Cleaned DataFrame
    """
    # Load data
    df = pd.read_csv(r'D:\Projects\Bouman9projects\challenge-regression\solutions\cloris_f_chen_regression\data\dataset_wout_surf_encoded.csv')
    
    # Check and remove duplicate rows
    num_duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {num_duplicates}")
    df_cleaned = df.drop_duplicates()
    print(f"New number of rows: {df_cleaned.shape[0]}")
    
    # Check missing values
    missing_values = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values[missing_values > 0])
    
    total_missing = df.isnull().sum().sum()
    print(f"\nTotal missing values: {total_missing}")
    
    # Check text columns
    text_columns = df.select_dtypes(include=["object"]).columns
    print(f"\nText columns: {list(text_columns)}")
    
    # Save cleaned data
    save_path= r'D:\Projects\Bouman9projects\challenge-regression\solutions\cloris_f_chen_regression\data\cleaned_dataset.csv'
    df_cleaned.to_csv(save_path, index=False)
    print(f"Cleaned data saved to: {save_path}")
    
    return df_cleaned


def analyze_numeric_correlation(
    df: pd.DataFrame, save_dir: str, threshold: float = 0.8
) -> list:
    """
    Analyze correlations between numeric features.
    
    Args:
        df: DataFrame containing numeric features
        save_dir: Directory to save plots
        threshold: Correlation threshold for strong relationships
    
    Returns:
        List of strongly correlated feature pairs (var1, var2, correlation)
    """
    print("\n" + "=" * 60)
    print("NUMERIC CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    print(f"Number of numeric columns: {len(numeric_df.columns)}")
    
    # Calculate Pearson correlation
    corr_matrix = numeric_df.corr(method="pearson")
    
    # Select upper triangle to avoid duplicates
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1
    ).astype(bool))

    # Analyze correlation distribution
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    upper_abs = upper.abs().stack()
    counts = pd.cut(upper_abs, bins=bins, labels=labels).value_counts().sort_index()
    print("\nCorrelation distribution:")
    print(counts)
    
    # Find strong correlations
    strong_correlations = []
    for col in upper.columns:
        for idx in upper.index:
            if (
                abs(upper.loc[idx, col]) > threshold
                and not pd.isna(upper.loc[idx, col])
            ):
                strong_correlations.append((idx, col, upper.loc[idx, col]))
    
    # Print strong correlations
    if strong_correlations:
        print("\nStrong correlations (|r| > 0.8):")
        for var1, var2, corr in strong_correlations:
            print(f"  {var1} <-> {var2}: {corr:.3f}")
    else:
        print("\nNo strong correlations found")
    
    # Create correlation heatmap
    print("\nCreating correlation heatmap...")
    plt.figure(figsize=(20, 15))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0
    )
    plt.title("Correlation Matrix of Numeric Features")
    plt.savefig(
        os.path.join(save_dir, "correlation_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("✓ Correlation heatmap saved")
    
    return strong_correlations

def batch_anova(
    df: pd.DataFrame,
    category_vars: list,
    numeric_vars: list,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Perform batch ANOVA tests between categorical and numeric variables.
    
    Args:
        df: Input DataFrame
        category_vars: List of categorical variables
        numeric_vars: List of numeric variables
        alpha: Significance level
    
    Returns:
        DataFrame of ANOVA results
    """
    results = []
    total_tests = len(category_vars) * len(numeric_vars)
    
    for i, cat_col in enumerate(category_vars):
        for num_col in numeric_vars:
            # Print progress every 10 tests
            if (i * len(numeric_vars) + numeric_vars.index(num_col)) % 10 == 0:
                current = i * len(numeric_vars) + numeric_vars.index(num_col) + 1
                print(f"  Processing ANOVA test {current}/{total_tests}...")
            
            try:
                # Filter valid data
                valid_data = df[[cat_col, num_col]].dropna()
                
                # Skip if insufficient samples
                if len(valid_data) < 10:
                    continue
                
                # Group data by category
                groups = [
                    group[num_col].values
                    for name, group in valid_data.groupby(cat_col)
                ]
                
                # Skip if insufficient groups
                if len(groups) < 2 or any(len(g) < 2 for g in groups):
                    continue
                
                # Perform ANOVA
                f_val, p_val = stats.f_oneway(*groups)
                significant = p_val < alpha
                
                results.append(
                    {
                        "Category": cat_col,
                        "Numeric": num_col,
                        "F": f_val,
                        "P": p_val,
                        "Significant": significant,
                    }
                )
            except Exception:
                continue
    
    return pd.DataFrame(results)


def analyze_anova(
    df: pd.DataFrame,
    category_vars: list,
    numeric_vars: list,
    save_dir: str,
) -> pd.DataFrame:
    """
    Analyze categorical-numeric relationships using ANOVA.
    
    Args:
        df: Input DataFrame
        category_vars: List of categorical variables
        numeric_vars: List of numeric variables
        save_dir: Directory to save plots
    
    Returns:
        DataFrame of ANOVA results
    """
    print("\n" + "=" * 60)
    print("ANOVA ANALYSIS (Category vs Numeric)")
    print("=" * 60)
    
    # Print unique value counts for categoricals
    print("\nUnique values in categorical variables:")
    for cat_var in category_vars:
        n_unique = df[cat_var].nunique()
        print(f"  {cat_var}: {n_unique} unique values")
    
    # Perform ANOVA tests
    print("\nPerforming ANOVA tests...")
    anova_results = batch_anova(df, category_vars, numeric_vars)
    print(f"\n✓ ANOVA completed: {len(anova_results)} valid tests")
    
    # Show significant results
    significant_anova = anova_results[anova_results["Significant"]]
    print(
        f"\nSignificant ANOVA results (p < 0.05): {len(significant_anova)}"
    )
    if len(significant_anova) > 0:
        print("\nTop 10 most significant ANOVA results:")
        print(
            significant_anova.nsmallest(10, "P")[
                ["Category", "Numeric", "F", "P"]
            ]
        )
    
    # Visualize p-value distribution
    if len(anova_results) > 0:
        print("\nCreating ANOVA p-value distribution plot...")
        plt.figure(figsize=(10, 8))
        plt.hist(
            anova_results["P"].dropna(), bins=30, edgecolor="black", alpha=0.7
        )
        plt.axvline(0.05, color="red", linestyle="--", label="alpha=0.05")
        plt.title("Distribution of p-values (ANOVA)")
        plt.xlabel("P-value")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(
            os.path.join(save_dir, "P_value_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ P-value distribution plot saved")
    
    return anova_results


def cramers_v(confusion_matrix: pd.DataFrame) -> float:
    """
    Calculate Cramér's V statistic for categorical-categorical association.
    
    Args:
        confusion_matrix: Contingency table of two categorical variables
    
    Returns:
        Cramér's V statistic (0-1)
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    # Apply bias corrections
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def analyze_categorical_correlation(
    df: pd.DataFrame, category_vars: list, save_dir: str, threshold: float = 0.8
) -> tuple:
    """
    Analyze correlations between categorical variables using Cramér's V.
    
    Args:
        df: Input DataFrame
        category_vars: List of categorical variables
        save_dir: Directory to save plots
        threshold: Correlation threshold for strong relationships
    
    Returns:
        Tuple: (Cramér's V matrix, list of high-correlation pairs)
    """
    print("\n" + "=" * 60)
    print("CATEGORICAL CORRELATION ANALYSIS (Cramér's V)")
    print("=" * 60)
    
    # Initialize results matrix
    cramers_results = pd.DataFrame(
        np.zeros((len(category_vars), len(category_vars))),
        columns=category_vars,
        index=category_vars,
    )
    
    # Calculate Cramér's V for each pair
    print("Calculating Cramér's V...")
    for i, var1 in enumerate(category_vars):
        for j, var2 in enumerate(category_vars):
            if var1 == var2:
                cramers_results.loc[var1, var2] = 1.0  # Diagonal = perfect correlation
            else:
                try:
                    # Create contingency table
                    confusion_matrix = pd.crosstab(df[var1], df[var2])
                    cramers_v_value = cramers_v(confusion_matrix)
                    cramers_results.loc[var1, var2] = cramers_v_value
                except Exception:
                    cramers_results.loc[var1, var2] = np.nan
        print(f"  Processed {i+1}/{len(category_vars)} variables")
    
    print("\nCramér's V Correlation Matrix:")
    print(cramers_results.round(3))
    
    # Visualize as heatmap
    print("\nCreating Cramér's V heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cramers_results.astype(float),
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    plt.title("Cramér's V Correlation Matrix")
    plt.savefig(
        os.path.join(save_dir, "cramers_v_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("✓ Cramér's V heatmap saved")
    
    # Find high correlations
    high_cramers = []
    for i in range(len(category_vars)):
        for j in range(i + 1, len(category_vars)):
            cv = cramers_results.iloc[i, j]
            if cv > threshold and not pd.isna(cv):
                high_cramers.append(
                    (category_vars[i], category_vars[j], cv)
                )
    
    return cramers_results, high_cramers


def generate_summary(
    strong_correlations: list,
    high_cramers: list,
    save_dir: str,
) -> None:
    """
    Generate summary report of high-correlation features.
    
    Args:
        strong_correlations: High-correlation numeric pairs
        high_cramers: High-correlation categorical pairs
        save_dir: Directory containing analysis outputs
    """
    print("\n" + "=" * 60)
    print("SUMMARY: FEATURES TO REMOVE DUE TO HIGH CORRELATION")
    print("=" * 60)
    
    # Numeric features summary
    if strong_correlations:
        print("\n1. Numeric features with strong correlation (|r| > 0.8):")
        print("   Recommendation: Keep one feature from each pair")
        for var1, var2, corr in strong_correlations:
            print(f"   - {var1} <-> {var2}: {corr:.3f}")
    else:
        print("\n1. No numeric features with correlation > 0.8")
    
    # Categorical features summary
    if high_cramers:
        print("\n2. Categorical features with high Cramér's V (> 0.8):")
        for var1, var2, cv in high_cramers:
            print(f"   - {var1} <-> {var2}: {cv:.3f}")
    else:
        print("\n2. No categorical features with Cramér's V > 0.8")
    
    # Final completion message
    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETED SUCCESSFULLY!")
    print(f"All plots saved in: {os.path.abspath(save_dir)}")
    print("=" * 60)


def main() -> None:
    """Main function to execute the full analysis pipeline."""
    # Configure paths
    DATA_DIR = r"D:\Projects\Bouman9projects\challenge-regression\solutions\cloris_f_chen_regression\data"
    PLOTS_DIR = r"D:\Projects\Bouman9projects\challenge-regression\solutions\cloris_f_chen_regression\plots"
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Step 1: Data loading and cleaning
    raw_data_path = os.path.join(DATA_DIR, "dataset_wout_surf_encoded.csv")
    clean_data_path = os.path.join(DATA_DIR, "cleaned_dataset.csv")
    df = load_and_clean_data(raw_data_path, clean_data_path)
    
    # Define analysis variables
    numeric_vars = [
        "bedroomCount",
        "bathroomCount",
        "habitableSurface",
        "toiletCount",
        "totalParkingCount",
        "epcScore_encoded",
        "hasAttic_encoded",
        "hasGarden_encoded",
        "hasAirConditioning_encoded",
        "hasArmoredDoor_encoded",
        "hasVisiophone_encoded",
        "hasTerrace_encoded",
        "hasOffice_encoded",
        "hasSwimmingPool_encoded",
        "hasFireplace_encoded",
        "hasBasement_encoded",
        "hasDressingRoom_encoded",
        "hasDiningRoom_encoded",
        "hasLift_encoded",
        "hasHeatPump_encoded",
        "hasPhotovoltaicPanels_encoded",
        "hasLivingRoom_encoded",
        "price",
    ]
    
    category_vars = [
        "postCode",
        "province_encoded",
        "type_encoded",
        "subtype_encoded",
        "locality_encoded",
    ]
    
    # Step 2: Numeric correlation analysis
    strong_numeric = analyze_numeric_correlation(df, PLOTS_DIR)
    
    # Step 3: Categorical-Numeric ANOVA analysis
    anova_results = analyze_anova(df, category_vars, numeric_vars, PLOTS_DIR)
    
    # Step 4: Categorical-Categorical correlation
    _, high_categorical = analyze_categorical_correlation(
        df, category_vars, PLOTS_DIR
    )
    
    # Step 5: Generate summary report
    generate_summary(strong_numeric, high_categorical, PLOTS_DIR)


if __name__ == "__main__":
    main()