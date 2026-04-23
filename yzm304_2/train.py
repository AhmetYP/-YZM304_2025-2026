"""
train.py — Egitim ve Degerlendirme Fonksiyonlari
Herhangi bir modeli egitmek ve test etmek icin genel amacli fonksiyonlar.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def train_model(model, train_loader, epochs=10, lr=0.01, device="cpu"):
    """
    Verilen modeli egitir ve epoch basina loss degerlerini doner.

    Args:
        model (nn.Module): Egitilecek model
        train_loader (DataLoader): Egitim verisi
        epochs (int): Epoch sayisi
        lr (float): Ogrenme orani
        device (str): "cpu" veya "cuda"

    Returns:
        list: Epoch basina ortalama loss degerleri
    """
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        avg_loss = running_loss / batch_count
        epoch_losses.append(avg_loss)
        print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    return epoch_losses


def evaluate_model(model, test_loader, device="cpu"):
    """
    Modeli test seti uzerinde degerlendirir.

    Args:
        model (nn.Module): Degerlendrilecek model
        test_loader (DataLoader): Test verisi
        device (str): "cpu" veya "cuda"

    Returns:
        tuple: (y_true, y_pred) — gercek ve tahmin edilen etiketler
    """
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    correct = (y_true == y_pred).sum()
    total = len(y_true)
    print(f"  Test Dogrulugu: {correct}/{total} ({100*correct/total:.2f}%)")

    return y_true, y_pred


def extract_features_lenet(model, data_loader, device="cpu"):
    """
    LeNet-5 modelinin F4 katmanindan (84 boyutlu) ozellik vektorlerini cikarir.
    Son siniflandirici (F5) oncesindeki ciktilar feature olarak kullanilir.

    Args:
        model (nn.Module): Egitilmis LeNet5 modeli
        data_loader (DataLoader): Veri yukleyici
        device (str): "cpu" veya "cuda"

    Returns:
        tuple: (features, labels) — numpy dizileri
    """
    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    # F4 ciktisini yakalamak icin hook
    features_buffer = []

    def hook_fn(module, input, output):
        features_buffer.append(output.detach().cpu())

    # F4 katmanina hook ekle
    hook = model.f4.register_forward_hook(hook_fn)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            _ = model(images)  # Forward pass (hook ciktiyi yakalar)
            all_labels.extend(labels.numpy())

    hook.remove()

    features = torch.cat(features_buffer, dim=0).numpy()
    labels = np.array(all_labels)

    print(f"  Ozellik vektoru boyutu: {features.shape}")
    print(f"  Etiket boyutu         : {labels.shape}")

    return features, labels

