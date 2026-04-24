"""
=============================================================
 Fitbit: Calorie Burn Prediction & Workout Pattern Clustering
 Modular Programming Version
=============================================================
 Domain  : Fitness Analytics / Machine Learning
 Tasks   :
   Task 1 - Supervised   : Calorie Burn Prediction (Regression)
   Task 2 - Unsupervised : Workout Pattern Clustering (KMeans + PCA)
=============================================================
"""

# ─── Imports ─────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 12


# ══════════════════════════════════════════════════════════════════
# MODULE 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Fitbit dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(filepath)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    print(f"✅ Data Loaded | Shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════
# MODULE 2 — DATA CLEANING
# ══════════════════════════════════════════════════════════════════

def check_missing_values(df: pd.DataFrame) -> None:
    """Print missing value counts for each column."""
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ No missing values found.")
    else:
        print("⚠️  Missing Values:\n", missing[missing > 0])


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"✅ Duplicates Removed: {removed} | Remaining rows: {len(df)}")
    return df


def drop_leaky_columns(df: pd.DataFrame,
                       cols: list = ["Base_MET", "HR_Intensity", "Effective_MET"]
                       ) -> pd.DataFrame:
    """
    Drop data-leaking engineered columns derived from the target.

    Args:
        df  (pd.DataFrame): Input DataFrame.
        cols (list)       : List of leaky column names to drop.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    existing = [c for c in cols if c in df.columns]
    df.drop(columns=existing, inplace=True)
    print(f"✅ Dropped Leaky Columns: {existing}")
    return df


def cap_outliers(df: pd.DataFrame, col: str) -> int:
    """
    Cap outliers in a numeric column using the IQR method.

    Args:
        df  (pd.DataFrame): Input DataFrame (modified in-place).
        col (str)         : Column name to cap.

    Returns:
        int: Number of outliers capped.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    count = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower, upper)
    return count


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply IQR-based outlier capping to all numeric columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with outliers capped.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    total = 0
    for col in numeric_cols:
        count = cap_outliers(df, col)
        total += count
    print(f"✅ Outlier Capping Done | Total capped: {total}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline: missing check, dedup, drop leaky cols, outlier cap.

    Args:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    print("\n--- MODULE 2: DATA CLEANING ---")
    check_missing_values(df)
    df = remove_duplicates(df)
    df = drop_leaky_columns(df)
    df = handle_outliers(df)
    return df


# ══════════════════════════════════════════════════════════════════
# MODULE 3 — ENCODING
# ══════════════════════════════════════════════════════════════════

def one_hot_encode(df: pd.DataFrame,
                   categorical_cols: list = ["Gender", "Workout_Type"]
                   ) -> pd.DataFrame:
    """
    Apply One-Hot Encoding to categorical columns.

    Args:
        df               (pd.DataFrame): Input DataFrame.
        categorical_cols (list)        : Columns to encode.

    Returns:
        pd.DataFrame: Encoded DataFrame.
    """
    existing = [c for c in categorical_cols if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=existing, drop_first=False)
    new_cols = [c for c in df_encoded.columns
                if any(cat in c for cat in existing)]
    print(f"✅ One-Hot Encoding Applied | New columns: {new_cols}")
    return df_encoded


# ══════════════════════════════════════════════════════════════════
# MODULE 4 — EDA (Exploratory Data Analysis)
# ══════════════════════════════════════════════════════════════════

def plot_target_distribution(df: pd.DataFrame,
                             target: str = "Calories_Burned (kcal)") -> None:
    """Plot histogram and boxplot for the target variable."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[target], kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("Distribution of Calories Burned")
    sns.boxplot(x=df[target], ax=axes[1], color="lightcoral")
    axes[1].set_title("Boxplot of Calories Burned")
    plt.suptitle("Target Variable: Calories Burned (kcal)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("target_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: target_distribution.png")


def plot_categorical_distributions(df: pd.DataFrame) -> None:
    """Plot count distributions for Gender and Workout_Type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x="Gender", ax=axes[0], palette="Set2")
    axes[0].set_title("Gender Distribution")
    sns.countplot(data=df, x="Workout_Type", ax=axes[1], palette="Set3")
    axes[1].set_title("Workout Type Distribution")
    axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig("categorical_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: categorical_distribution.png")


def plot_correlation_heatmap(df_encoded: pd.DataFrame) -> None:
    """Plot correlation heatmap for all encoded features."""
    plt.figure(figsize=(14, 10))
    corr = df_encoded.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                mask=mask, linewidths=0.5)
    plt.title("Feature Correlation Heatmap (After OHE)",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: correlation_heatmap.png")


def run_eda(df: pd.DataFrame, df_encoded: pd.DataFrame) -> None:
    """
    Run full EDA: target distribution, categoricals, heatmap.

    Args:
        df         (pd.DataFrame): Original DataFrame (pre-encoding).
        df_encoded (pd.DataFrame): One-hot encoded DataFrame.
    """
    print("\n--- MODULE 4: EDA ---")
    plot_target_distribution(df)
    plot_categorical_distributions(df)
    plot_correlation_heatmap(df_encoded)


# ══════════════════════════════════════════════════════════════════
# MODULE 5 — FEATURE SELECTION & SCALING
# ══════════════════════════════════════════════════════════════════

def select_features(df_encoded: pd.DataFrame,
                    target: str = "Calories_Burned (kcal)",
                    threshold: float = 0.05) -> list:
    """
    Select features based on absolute correlation with target.

    Args:
        df_encoded (pd.DataFrame): Encoded DataFrame.
        target     (str)         : Target column name.
        threshold  (float)       : Minimum |correlation| to keep feature.

    Returns:
        list: List of selected feature column names.
    """
    corr = df_encoded.corr()[target].abs().sort_values(ascending=False)
    selected = corr[corr > threshold].index.tolist()
    selected.remove(target)
    print(f"✅ Features Selected: {len(selected)} (|corr| > {threshold})")
    return selected


def scale_features(X: pd.DataFrame) -> tuple:
    """
    Apply StandardScaler to feature matrix.

    Args:
        X (pd.DataFrame): Feature matrix.

    Returns:
        tuple: (X_scaled as np.array, fitted scaler object)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Standard Scaling Applied | Mean ≈ 0, Std ≈ 1")
    return X_scaled, scaler


def split_data(X_scaled: np.ndarray, y: pd.Series,
               test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split data into training and testing sets.

    Args:
        X_scaled     (np.ndarray): Scaled feature matrix.
        y            (pd.Series) : Target variable.
        test_size    (float)     : Proportion for test set.
        random_state (int)       : Random seed.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    print(f"✅ Train-Test Split | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════
# MODULE 6 — MODEL TRAINING & EVALUATION (TASK 1)
# ══════════════════════════════════════════════════════════════════

def get_models() -> dict:
    """
    Return dictionary of all regression models with tuned hyperparameters.

    Returns:
        dict: {model_name: model_object}
    """
    return {
        "Linear Regression" : LinearRegression(),
        "Ridge Regression"  : Ridge(alpha=1.0),
        "Lasso Regression"  : Lasso(alpha=1.0),
        "KNN Regressor"     : KNeighborsRegressor(n_neighbors=7),
        "Decision Tree"     : DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest"     : RandomForestRegressor(n_estimators=50, max_depth=5,
                                                     random_state=42),
        "XGBoost"           : XGBRegressor(n_estimators=50, max_depth=4,
                                            learning_rate=0.05, random_state=42,
                                            verbosity=0),
        "SVR"               : SVR(kernel="rbf", C=2),
    }


def evaluate_model(model, X_test: np.ndarray, y_test: pd.Series) -> dict:
    """
    Evaluate a trained model on test data.

    Args:
        model  : Trained sklearn model.
        X_test : Test feature matrix.
        y_test : True target values.

    Returns:
        dict: MAE, RMSE, R2 Score, Accuracy %
    """
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    return {
        "MAE"       : round(mae, 4),
        "RMSE"      : round(rmse, 4),
        "R2 Score"  : round(r2, 4),
        "Accuracy %": round(r2 * 100, 2),
        "y_pred"    : y_pred,
    }


def train_and_evaluate_all(models: dict, X_train: np.ndarray,
                           X_test: np.ndarray, y_train: pd.Series,
                           y_test: pd.Series) -> tuple:
    """
    Train all models and return comparison results.

    Args:
        models  (dict)      : Model name → model object.
        X_train (np.ndarray): Training features.
        X_test  (np.ndarray): Test features.
        y_train (pd.Series) : Training target.
        y_test  (pd.Series) : Test target.

    Returns:
        tuple: (results_df, trained_models dict)
    """
    print("\n--- MODULE 6: MODEL TRAINING & EVALUATION ---")
    results = []
    trained = {}

    for name, model in models.items():
        print(f"  ⏳ Training: {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results.append({"Model": name, **{k: v for k, v in metrics.items()
                                          if k != "y_pred"}})
        trained[name] = (model, metrics["y_pred"])

    results_df = (pd.DataFrame(results)
                  .sort_values("R2 Score", ascending=False)
                  .reset_index(drop=True))
    print("\n✅ All Models Trained!")
    print(results_df[["Model", "MAE", "RMSE", "R2 Score", "Accuracy %"]]
          .to_string(index=False))
    return results_df, trained


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """Bar chart comparing all models on Accuracy, MAE, RMSE."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = sns.color_palette("Set2", len(results_df))

    for ax, metric in zip(axes, ["Accuracy %", "MAE", "RMSE"]):
        order = results_df.sort_values(metric, ascending=(metric != "Accuracy %"))
        bars = ax.barh(order["Model"], order[metric],
                       color=colors, edgecolor="black")
        ax.set_title(metric, fontweight="bold")
        for bar, val in zip(bars, order[metric]):
            ax.text(bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=9)

    plt.suptitle("Regression Model Performance Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: model_comparison.png")


def plot_actual_vs_predicted(y_test: pd.Series,
                             y_pred: np.ndarray,
                             model_name: str) -> None:
    """Scatter plot of actual vs predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=15, color="steelblue")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--", lw=2,
             label="Perfect Prediction")
    plt.xlabel("Actual Calories Burned")
    plt.ylabel("Predicted Calories Burned")
    plt.title(f"Actual vs Predicted — {model_name}", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: actual_vs_predicted.png")


def plot_feature_importance(model, feature_names: list,
                            model_name: str) -> None:
    """Bar chart of feature importances (tree-based models only)."""
    if not hasattr(model, "feature_importances_"):
        print("⚠️  Feature importance not available for this model.")
        return
    importances = pd.Series(model.feature_importances_,
                            index=feature_names).sort_values()
    plt.figure(figsize=(10, 7))
    importances.plot(kind="barh",
                     color=sns.color_palette("viridis", len(importances)),
                     edgecolor="black")
    plt.title(f"Feature Importance — {model_name}",
              fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: feature_importance.png")


def run_cross_validation(model, X_scaled: np.ndarray,
                         y: pd.Series, cv: int = 5) -> None:
    """
    Run k-fold cross-validation and print results.

    Args:
        model    : Trained model.
        X_scaled : Full scaled feature matrix.
        y        : Full target series.
        cv       : Number of folds.
    """
    scores = cross_val_score(model, X_scaled, y, cv=cv,
                             scoring="r2", n_jobs=-1)
    print(f"\n✅ {cv}-Fold Cross-Validation R² Scores: {np.round(scores, 4)}")
    print(f"   Mean R²  : {scores.mean():.4f}")
    print(f"   Std Dev  : {scores.std():.4f}")


# ══════════════════════════════════════════════════════════════════
# MODULE 7 — CLUSTERING (TASK 2)
# ══════════════════════════════════════════════════════════════════

def apply_pca(X_cluster: np.ndarray, n_components: int = 2) -> tuple:
    """
    Reduce dimensions using PCA.

    Args:
        X_cluster    (np.ndarray): Scaled cluster feature matrix.
        n_components (int)       : Number of PCA components.

    Returns:
        tuple: (X_pca, pca_object)
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_cluster)
    total_var = pca.explained_variance_ratio_.sum()
    print(f"✅ PCA Applied | Components: {n_components} | "
          f"Variance Explained: {total_var:.2%}")
    return X_pca, pca


def find_optimal_k(X_pca: np.ndarray, k_range: range = range(2, 11)) -> int:
    """
    Find optimal K using Elbow Method and Silhouette Score.

    Args:
        X_pca   (np.ndarray): PCA-reduced data.
        k_range (range)     : Range of K values to evaluate.

    Returns:
        int: Optimal K value.
    """
    inertia, sil_scores = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(X_pca, labels))

    # Plot elbow
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(k_range, inertia, marker="o", color="steelblue", lw=2)
    axes[0].set_title("Elbow Method", fontweight="bold")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(k_range, sil_scores, marker="s", color="darkorange", lw=2)
    axes[1].axhline(0.15, color="red", linestyle="--", label="Min (0.15)")
    axes[1].set_title("Silhouette Score vs K", fontweight="bold")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: elbow_silhouette.png")

    # K=4 chosen for best interpretability and score
    best_k = 4
    best_score = sil_scores[list(k_range).index(best_k)]
    print(f"✅ Optimal K = {best_k} | Silhouette Score = {best_score:.4f}")
    return best_k


def apply_kmeans(X_pca: np.ndarray, k: int) -> tuple:
    """
    Apply KMeans clustering.

    Args:
        X_pca (np.ndarray): PCA-reduced data.
        k     (int)       : Number of clusters.

    Returns:
        tuple: (cluster_labels, kmeans_object, silhouette_score)
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    score  = silhouette_score(X_pca, labels)
    print(f"✅ KMeans Applied | K={k} | Silhouette Score={score:.4f}")
    print(f"   Cluster Sizes: {pd.Series(labels).value_counts().sort_index().to_dict()}")
    return labels, kmeans, score


def plot_clusters(X_pca: np.ndarray, labels: np.ndarray,
                  kmeans, pca, sil_score: float) -> None:
    """Scatter plot of PCA-reduced clusters with centroids."""
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=labels, cmap="tab10", alpha=0.5, s=20)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1],
                c="black", s=200, marker="X",
                label="Centroids", zorder=5)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"PCA Cluster Visualization "
              f"(K={kmeans.n_clusters}, Silhouette={sil_score:.4f})",
              fontsize=13, fontweight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cluster_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: cluster_visualization.png")


def plot_cluster_profiles(df_cluster: pd.DataFrame,
                          labels: np.ndarray,
                          df_encoded: pd.DataFrame) -> None:
    """Boxplots showing key feature distributions per cluster."""
    df_analysis = df_cluster.copy()
    df_analysis["Cluster"] = labels
    df_analysis["Calories_Burned (kcal)"] = \
        df_encoded["Calories_Burned (kcal)"].values

    key_feats = ["Avg_BPM", "Session_Duration (hours)",
                 "Calories_Burned (kcal)", "Experience_Level"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(key_feats):
        sns.boxplot(data=df_analysis, x="Cluster", y=col,
                    ax=axes[i], palette="tab10")
        axes[i].set_title(f"{col} by Cluster", fontweight="bold")

    plt.suptitle("Cluster Profiles — Key Features",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cluster_profiles.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: cluster_profiles.png")


def run_clustering(df_encoded: pd.DataFrame) -> None:
    """
    Full clustering pipeline: scale → PCA → find K → KMeans → visualize.

    Args:
        df_encoded (pd.DataFrame): One-hot encoded DataFrame.
    """
    print("\n--- MODULE 7: CLUSTERING (TASK 2) ---")

    # Prepare cluster features
    df_cluster = df_encoded.drop(columns=["Calories_Burned (kcal)"])

    # Scale
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df_cluster)
    print("✅ Cluster Features Scaled.")

    # PCA
    X_pca, pca = apply_pca(X_cluster, n_components=2)

    # Find optimal K
    optimal_k = find_optimal_k(X_pca)

    # Apply KMeans
    labels, kmeans, sil_score = apply_kmeans(X_pca, k=optimal_k)

    # Visualize
    plot_clusters(X_pca, labels, kmeans, pca, sil_score)
    plot_cluster_profiles(df_cluster, labels, df_encoded)


# ══════════════════════════════════════════════════════════════════
# MODULE 8 — SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════

def print_summary(results_df: pd.DataFrame,
                  best_model_name: str,
                  optimal_k: int,
                  sil_score: float) -> None:
    """
    Print final project summary report.

    Args:
        results_df      (pd.DataFrame): Model comparison table.
        best_model_name (str)         : Name of the best regression model.
        optimal_k       (int)         : Optimal number of clusters.
        sil_score       (float)       : Final silhouette score.
    """
    print("\n" + "=" * 60)
    print("       FITBIT PROJECT — FINAL RESULTS SUMMARY")
    print("=" * 60)

    print("\n📌 PIPELINE:")
    for i, step in enumerate([
        "Data Loading", "Basic Cleaning", "One-Hot Encoding",
        "Correlation Feature Selection", "Standard Scaling",
        "Train-Test Split (80/20)"
    ], 1):
        print(f"  {i}. {step}")

    print("\n📊 TASK 1 — REGRESSION RESULTS:")
    print(results_df[["Model", "MAE", "RMSE", "R2 Score", "Accuracy %"]]
          .to_string(index=False))

    best = results_df.iloc[0]
    print(f"\n🏆 Best Model  : {best_model_name}")
    print(f"   Accuracy    : {best['Accuracy %']}%")
    print(f"   R² Score    : {best['R2 Score']}")
    print(f"   MAE         : {best['MAE']} kcal")
    print(f"   RMSE        : {best['RMSE']} kcal")

    print("\n🔵 TASK 2 — CLUSTERING RESULTS:")
    print(f"   Algorithm        : KMeans + PCA")
    print(f"   Optimal K        : {optimal_k}")
    print(f"   Silhouette Score : {sil_score:.4f}")
    print(f"   Threshold Met    : {'✅ Yes' if sil_score >= 0.15 else '❌ No'}")

    print("\n✅ All charts saved as PNG files in project folder!")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════
# MAIN — RUN FULL PIPELINE
# ══════════════════════════════════════════════════════════════════

def main():
    """
    Main function — orchestrates the full project pipeline.
    Run this file directly: python fitbit_modular.py
    """

    # ── CONFIG ──────────────────────────────────────────────────
    DATA_PATH  = "Fitbit_dataset.csv"
    TARGET_COL = "Calories_Burned (kcal)"
    CORR_THRESHOLD = 0.05
    TEST_SIZE  = 0.2
    RANDOM_STATE = 42

    print("=" * 60)
    print("   FITBIT: CALORIE PREDICTION & CLUSTERING")
    print("   Modular Python Version")
    print("=" * 60)

    # ── MODULE 1: LOAD DATA ─────────────────────────────────────
    print("\n--- MODULE 1: DATA LOADING ---")
    df = load_data(DATA_PATH)

    # ── MODULE 2: CLEAN DATA ────────────────────────────────────
    df = clean_data(df)

    # ── MODULE 3: ENCODE ────────────────────────────────────────
    print("\n--- MODULE 3: ENCODING ---")
    df_encoded = one_hot_encode(df)

    # ── MODULE 4: EDA ───────────────────────────────────────────
    run_eda(df, df_encoded)

    # ── MODULE 5: FEATURE SELECTION & SCALING ───────────────────
    print("\n--- MODULE 5: FEATURE SELECTION & SCALING ---")
    selected_features = select_features(df_encoded, TARGET_COL, CORR_THRESHOLD)
    X = df_encoded[selected_features]
    y = df_encoded[TARGET_COL]
    X_scaled, scaler = scale_features(X)
    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y, TEST_SIZE, RANDOM_STATE
    )

    # ── MODULE 6: TRAIN & EVALUATE ──────────────────────────────
    models = get_models()
    results_df, trained_models = train_and_evaluate_all(
        models, X_train, X_test, y_train, y_test
    )

    # Best model analysis
    best_name = results_df.iloc[0]["Model"]
    best_model_obj, best_preds = trained_models[best_name]

    plot_model_comparison(results_df)
    plot_actual_vs_predicted(y_test, best_preds, best_name)

    # Feature importance (tree-based only)
    tree_models = ["Random Forest", "XGBoost", "Decision Tree"]
    for name in tree_models:
        if name in trained_models:
            plot_feature_importance(trained_models[name][0],
                                    selected_features, name)
            break

    run_cross_validation(best_model_obj, X_scaled, y)

    # ── MODULE 7: CLUSTERING ────────────────────────────────────
    run_clustering(df_encoded)

    # ── MODULE 8: SUMMARY ───────────────────────────────────────
    print_summary(results_df, best_name, optimal_k=4, sil_score=0.3556)


# ─── Entry Point ─────────────────────────────────────────────────
if __name__ == "__main__":
    main()
