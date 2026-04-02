import matplotlib.pyplot as plt
import pandas as pd

from YZM304_Proje1.src.config import FIGURES_DIR


def basic_info(df: pd.DataFrame):
    print("\n--- FIRST 5 ROWS ---")
    print(df.head())

    print("\n--- SHAPE ---")
    print(df.shape)

    print("\n--- INFO ---")
    print(df.info())

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    print("\n--- DESCRIPTIVE STATS ---")
    print(df.describe())

    print("\n--- CLASS DISTRIBUTION ---")
    print(df["class"].value_counts())
    print("\n--- CLASS DISTRIBUTION (RATIO) ---")
    print(df["class"].value_counts(normalize=True))


def plot_class_distribution(df: pd.DataFrame):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    counts = df["class"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png")
    plt.show()


def plot_feature_histograms(df: pd.DataFrame):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df.hist(figsize=(10, 8))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_histograms.png")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame):
    import seaborn as sns

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png")
    #plt.show()