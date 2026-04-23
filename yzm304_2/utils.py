"""
utils.py — Yardimci Fonksiyonlar
Loss grafigi, confusion matrix, metrik hesaplama ve sonuclari kaydetme.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI olmadan calisabilmek icin
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)


def ensure_dir(path):
    """Klasor yoksa olusturur."""
    os.makedirs(path, exist_ok=True)


def plot_loss_curve(train_losses, model_name, save_dir="figures"):
    """
    Epoch basina train loss grafigi cizer ve kaydeder.

    Args:
        train_losses (list): Her epoch icin ortalama loss degerleri
        model_name (str): Model adi (dosya adi icin)
        save_dir (str): Kayit dizini
    """
    ensure_dir(save_dir)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o',
             linewidth=2, color='#2196F3')
    plt.title(f"{model_name} — Egitim Loss Grafigi", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(save_dir, f"{model_name}_loss.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"[Kayit] Loss grafigi: {filepath}")


def plot_confusion_matrix(y_true, y_pred, model_name, class_names=None,
                          save_dir="figures"):
    """
    Confusion matrix heatmap cizer ve kaydeder.

    Args:
        y_true (array): Gercek etiketler
        y_pred (array): Tahmin edilen etiketler
        model_name (str): Model adi
        class_names (list): Sinif isimleri (0-9)
        save_dir (str): Kayit dizini
    """
    ensure_dir(save_dir)
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} — Karisiklik Matrisi", fontsize=14)
    plt.xlabel("Tahmin", fontsize=12)
    plt.ylabel("Gercek", fontsize=12)
    plt.tight_layout()
    filepath = os.path.join(save_dir, f"{model_name}_confusion.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"[Kayit] Confusion matrix: {filepath}")


def compute_metrics(y_true, y_pred, model_name):
    """
    Accuracy, Precision, Recall, F1 hesaplar ve yazdirir.

    Args:
        y_true (array): Gercek etiketler
        y_pred (array): Tahmin edilen etiketler
        model_name (str): Model adi

    Returns:
        dict: Metrikler sozlugu
    """
    metrics = {
        "model": model_name,
        "accuracy":  accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, average='macro') * 100,
        "recall":    recall_score(y_true, y_pred, average='macro') * 100,
        "f1":        f1_score(y_true, y_pred, average='macro') * 100,
    }

    print(f"\n{'='*50}")
    print(f" {model_name} — Test Sonuclari")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.2f}%")
    print(f"  Precision : {metrics['precision']:.2f}%")
    print(f"  Recall    : {metrics['recall']:.2f}%")
    print(f"  F1 Score  : {metrics['f1']:.2f}%")
    print(f"{'='*50}\n")

    return metrics


def print_classification_report(y_true, y_pred, model_name):
    """Detayli sinif bazli rapor yazdirir."""
    print(f"\n{model_name} — Sinif Bazli Rapor:")
    print(classification_report(y_true, y_pred,
                                target_names=[str(i) for i in range(10)]))
