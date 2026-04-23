"""
main.py — Ana Giris Noktasi
YZM304 Derin Ogrenme Proje 2

Kullanim (CMD):
    python main.py --step 1
    python main.py --step 2
    python main.py --step 3
    python main.py --step 4
    python main.py --step all

Adim 1: Model 1 (LeNet-5)
Adim 2: Model 2 (LeNet-5 + BatchNorm)
Adim 3: Model 3 (Pretrained ResNet18)
Adim 4: Model 4 (Hibrit: LeNet-5 + SVM)
"""

import argparse
import os
import numpy as np
import torch

from dataset import get_mnist_loaders, get_mnist_loaders_3ch
from models import LeNet5, LeNet5BN, get_resnet18
from train import train_model, evaluate_model, extract_features_lenet
from utils import (
    plot_loss_curve, plot_confusion_matrix,
    compute_metrics, print_classification_report
)


# ============================================================================
# Konfigurasyonlar
# ============================================================================
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_model1():
    """Adim 1: Model 1 — Temel LeNet-5 egitim ve degerlendirme."""
    print("\n" + "=" * 60)
    print(" ADIM 1: Model 1 — LeNet-5 (Temel Model)")
    print("=" * 60)
    print(f" Cihaz: {DEVICE}")
    print(f" Epoch: {EPOCHS}, LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}")
    print("=" * 60)

    # 1) Veri yukleme
    print("\n[1/4] Veri seti yukleniyor...")
    train_loader, test_loader, _, _ = get_mnist_loaders(batch_size=BATCH_SIZE)

    # 2) Model olusturma
    print("\n[2/4] Model olusturuluyor...")
    model = LeNet5()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  LeNet-5 toplam parametre: {total_params:,}")

    # 3) Egitim
    print(f"\n[3/4] Egitim basliyor ({EPOCHS} epoch)...")
    train_losses = train_model(
        model, train_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE
    )

    # 4) Test ve Degerlendirme
    print("\n[4/4] Test seti uzerinde degerlendirme...")
    y_true, y_pred = evaluate_model(model, test_loader, device=DEVICE)

    # 5) Metrikler ve Gorseller
    metrics = compute_metrics(y_true, y_pred, "Model1_LeNet5")
    print_classification_report(y_true, y_pred, "Model1_LeNet5")
    plot_loss_curve(train_losses, "Model1_LeNet5")
    plot_confusion_matrix(y_true, y_pred, "Model1_LeNet5")

    print("\n[Tamamlandi] Model 1 sonuclari 'figures/' klasorune kaydedildi.\n")
    return model, metrics


def run_model2():
    """Adim 2: Model 2 — LeNet-5 + BatchNorm egitim ve degerlendirme."""
    print("\n" + "=" * 60)
    print(" ADIM 2: Model 2 — LeNet-5 + BatchNorm")
    print("=" * 60)
    print(f" Cihaz: {DEVICE}")
    print(f" Epoch: {EPOCHS}, LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}")
    print("=" * 60)

    # 1) Veri yukleme (ayni veri seti)
    print("\n[1/4] Veri seti yukleniyor...")
    train_loader, test_loader, _, _ = get_mnist_loaders(batch_size=BATCH_SIZE)

    # 2) Model olusturma
    print("\n[2/4] Model olusturuluyor...")
    model = LeNet5BN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  LeNet-5+BN toplam parametre: {total_params:,}")

    # 3) Egitim (ayni hiperparametreler)
    print(f"\n[3/4] Egitim basliyor ({EPOCHS} epoch)...")
    train_losses = train_model(
        model, train_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE
    )

    # 4) Test ve Degerlendirme
    print("\n[4/4] Test seti uzerinde degerlendirme...")
    y_true, y_pred = evaluate_model(model, test_loader, device=DEVICE)

    # 5) Metrikler ve Gorseller
    metrics = compute_metrics(y_true, y_pred, "Model2_LeNet5BN")
    print_classification_report(y_true, y_pred, "Model2_LeNet5BN")
    plot_loss_curve(train_losses, "Model2_LeNet5BN")
    plot_confusion_matrix(y_true, y_pred, "Model2_LeNet5BN")

    print("\n[Tamamlandi] Model 2 sonuclari 'figures/' klasorune kaydedildi.\n")
    return model, metrics


def run_model3():
    """Adim 3: Model 3 — Pretrained ResNet18 fine-tuning."""
    resnet_lr = 0.001  # Transfer learning icin daha dusuk LR

    print("\n" + "=" * 60)
    print(" ADIM 3: Model 3 — Pretrained ResNet18")
    print("=" * 60)
    print(f" Cihaz: {DEVICE}")
    print(f" Epoch: {EPOCHS}, LR: {resnet_lr}, Batch: {BATCH_SIZE}")
    print(f" Pretrained: True, Freeze Features: True")
    print("=" * 60)

    # 1) Veri yukleme (3 kanalli)
    print("\n[1/4] Veri seti yukleniyor (3-kanal)...")
    train_loader, test_loader, _, _ = get_mnist_loaders_3ch(batch_size=BATCH_SIZE)

    # 2) Model olusturma
    print("\n[2/4] Pretrained ResNet18 yukleniyor...")
    model = get_resnet18(num_classes=10, pretrained=True, freeze_features=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Toplam parametre       : {total_params:,}")
    print(f"  Egitilebilir parametre : {trainable:,}")

    # 3) Egitim (sadece FC katmani egitilir)
    print(f"\n[3/4] Fine-tuning basliyor ({EPOCHS} epoch)...")
    train_losses = train_model(
        model, train_loader,
        epochs=EPOCHS, lr=resnet_lr, device=DEVICE
    )

    # 4) Test ve Degerlendirme
    print("\n[4/4] Test seti uzerinde degerlendirme...")
    y_true, y_pred = evaluate_model(model, test_loader, device=DEVICE)

    # 5) Metrikler ve Gorseller
    metrics = compute_metrics(y_true, y_pred, "Model3_ResNet18")
    print_classification_report(y_true, y_pred, "Model3_ResNet18")
    plot_loss_curve(train_losses, "Model3_ResNet18")
    plot_confusion_matrix(y_true, y_pred, "Model3_ResNet18")

    print("\n[Tamamlandi] Model 3 sonuclari 'figures/' klasorune kaydedildi.\n")
    return model, metrics


def run_model4():
    """
    Adim 4: Model 4 — Hibrit (LeNet-5 Feature Extraction + SVM)
    1) LeNet-5 egitilir
    2) F4 katmanindan 84 boyutlu ozellikler cikarilir
    3) .npy dosyalarina kaydedilir
    4) SVM ile siniflandirilir
    5) Model 1 (ayni LeNet-5) ile karsilastirilir (Model 5 muafiyeti)
    """
    from sklearn.svm import SVC

    print("\n" + "=" * 60)
    print(" ADIM 4: Model 4 — Hibrit (LeNet-5 + SVM)")
    print("=" * 60)
    print(f" Cihaz: {DEVICE}")
    print(f" CNN Epoch: {EPOCHS}, LR: {LEARNING_RATE}")
    print(f" Klasik ML: SVM (RBF kernel)")
    print("=" * 60)

    # 1) Veri yukleme
    print("\n[1/6] Veri seti yukleniyor...")
    train_loader, test_loader, _, _ = get_mnist_loaders(batch_size=BATCH_SIZE)

    # 2) LeNet-5 egitimi
    print("\n[2/6] LeNet-5 egitiliyor (ozellik cikarimi icin)...")
    model = LeNet5()
    train_losses = train_model(
        model, train_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE
    )

    # 3) Ozellik cikarimi
    print("\n[3/6] Ozellik cikarimi (F4 katmani, 84 boyut)...")
    print("  Train seti:")
    features_train, labels_train = extract_features_lenet(
        model, train_loader, device=DEVICE
    )
    print("  Test seti:")
    features_test, labels_test = extract_features_lenet(
        model, test_loader, device=DEVICE
    )

    # 4) .npy dosyalarina kaydet
    print("\n[4/6] Ozellikler .npy dosyalarina kaydediliyor...")
    features_dir = "features"
    os.makedirs(features_dir, exist_ok=True)

    np.save(os.path.join(features_dir, "features_train.npy"), features_train)
    np.save(os.path.join(features_dir, "labels_train.npy"), labels_train)
    np.save(os.path.join(features_dir, "features_test.npy"), features_test)
    np.save(os.path.join(features_dir, "labels_test.npy"), labels_test)

    print(f"  features_train.npy : {features_train.shape}")
    print(f"  labels_train.npy   : {labels_train.shape}")
    print(f"  features_test.npy  : {features_test.shape}")
    print(f"  labels_test.npy    : {labels_test.shape}")

    # 5) SVM egitimi ve degerlendirmesi
    print("\n[5/6] SVM egitiliyor (RBF kernel)...")
    svm = SVC(kernel='rbf', gamma='scale', C=1.0)
    svm.fit(features_train, labels_train)
    svm_preds = svm.predict(features_test)

    metrics_svm = compute_metrics(labels_test, svm_preds, "Model4_Hibrit_SVM")
    print_classification_report(labels_test, svm_preds, "Model4_Hibrit_SVM")
    plot_confusion_matrix(labels_test, svm_preds, "Model4_Hibrit_SVM")

    # 6) Model 1 (LeNet-5 end-to-end) ile karsilastirma (Model 5 muafiyeti)
    print("\n[6/6] Model 1 vs Model 4 karsilastirma (Model 5 muafiyeti)...")
    y_true_cnn, y_pred_cnn = evaluate_model(model, test_loader, device=DEVICE)
    metrics_cnn = compute_metrics(y_true_cnn, y_pred_cnn,
                                  "Model1_LeNet5 (karsilastirma)")

    print("\n" + "=" * 60)
    print(" KARSILASTIRMA: LeNet-5 (end-to-end) vs LeNet-5+SVM (hibrit)")
    print("=" * 60)
    print(f"  LeNet-5 End-to-End : {metrics_cnn['accuracy']:.2f}%")
    print(f"  LeNet-5 + SVM      : {metrics_svm['accuracy']:.2f}%")
    print("=" * 60)

    print("\n[Tamamlandi] Model 4 sonuclari 'figures/' ve 'features/' klasorlerine kaydedildi.\n")
    return metrics_svm, metrics_cnn


def main():
    parser = argparse.ArgumentParser(
        description="YZM304 Derin Ogrenme — Proje 2"
    )
    parser.add_argument(
        "--step", type=str, default="1",
        help="Calistirilacak adim: 1, 2, 3, 4, all (varsayilan: 1)"
    )
    args = parser.parse_args()

    if args.step in ("1", "all"):
        run_model1()

    if args.step in ("2", "all"):
        run_model2()

    if args.step in ("3", "all"):
        run_model3()

    if args.step in ("4", "all"):
        run_model4()


if __name__ == "__main__":
    main()
