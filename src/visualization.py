"""EDA plots for the complete (pre-missingness) heart failure dataset."""

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR, TARGET_COLUMN


def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    df[TARGET_COLUMN].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_xticklabels(["Survived (0)", "Death event (1)"], rotation=0)
    ax.set_ylabel("Patient count")
    ax.set_title("Target class distribution — Heart Failure Clinical Records (n=299)")
    for i, v in enumerate(df[TARGET_COLUMN].value_counts().sort_index()):
        ax.text(i, v + 3, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)


def plot_feature_distributions(df):
    features = [c for c in df.columns if c != TARGET_COLUMN]
    fig, axes = plt.subplots(4, 3, figsize=(14, 14))
    for ax, col in zip(axes.flat, features):
        sns.histplot(df[col], ax=ax, kde=True, color="#4C72B0")
        ax.set_title(col)
    fig.suptitle("Feature distributions — Heart Failure Clinical Records", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_matrix(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Feature correlation matrix (complete data, no missingness yet)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=200)
    plt.close(fig)


def plot_missing_per_feature(masked_df, filename, title):
    counts = masked_df.isna().sum().sort_values(ascending=False)
    counts = counts[counts > 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color="#C44E52")
    ax.set_ylabel("Missing count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def plot_missingness_heatmap(masked_df, filename, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(masked_df.isna(), cbar=False, cmap=["#EAEAF2", "#C44E52"], ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Training-set row")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    data = load_raw_data()
    plot_class_distribution(data)
    plot_feature_distributions(data)
    plot_correlation_matrix(data)
    print("Saved 3 figures to results/figures/")
