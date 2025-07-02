import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["id"])
    return df


def transform_types(df):
    boolean_cols = df.columns[df.columns.str.startswith("has")]
    for col in boolean_cols:
        df[col] = df[col].astype("boolean")

    numerical_cols = df.select_dtypes(["float64", "int64"])
    numerical_cols = numerical_cols.drop(columns=["postCode"], errors="ignore")
    df[numerical_cols.columns] = df[numerical_cols.columns].astype("float64")

    categorical_cols = df.select_dtypes(["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    if "postCode" in df.columns:
        df["postCode"] = df["postCode"].astype("category")

    return df


def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]


def remove_surface_outliers(df):
    return df[df["habitableSurface"] <= 5000]


def preprocess_data(df):
    df = df.drop(columns=["locality", "floodZoneType"], errors="ignore")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_order_cols = ["epcScore"]
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    if "epcScore" in cat_cols:
        cat_cols.remove("epcScore")
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    ordinal_encoder = OrdinalEncoder(
        categories=[["A", "B", "C", "D", "E", "F", "G"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        [
            ("num", KNNImputer(n_neighbors=5), num_cols),
            (
                "ordered_cat",
                Pipeline(
                    [("encode", ordinal_encoder), ("impute", KNNImputer(n_neighbors=5))]
                ),
                cat_order_cols,
            ),
            (
                "nominal_cat",
                Pipeline(
                    [("encode", onehot_encoder), ("impute", KNNImputer(n_neighbors=5))]
                ),
                cat_cols,
            ),
        ],
        remainder="passthrough",
    )

    print("Starting transformation ....")
    df_preprocessed = preprocessor.fit_transform(df)

    encoded_cat_names = (
        preprocessor.named_transformers_["nominal_cat"]
        .named_steps["encode"]
        .get_feature_names_out(cat_cols)
    )
    final_columns = (
        num_cols + cat_order_cols + list(encoded_cat_names) + list(bool_cols)
    )

    df_preprocessed = pd.DataFrame(df_preprocessed, columns=final_columns)
    return df_preprocessed


def main():
    df = load_data("data/cleaned_real_estate_data.csv")
    df = transform_types(df)
    df = remove_outliers_iqr(df, "price")
    df = remove_surface_outliers(df)
    df_preprocessed = preprocess_data(df)
    df_preprocessed.to_csv(
        "data/dataset_knn_neighbors_new.csv", index=False, encoding="utf-8"
    )
    print(
        "Preprocessing complete. Output saved to 'data/dataset_knn_neighbors_new.csv'"
    )


if __name__ == "__main__":
    main()
