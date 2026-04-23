"""
dataset.py — Veri Yükleme ve Ön İşleme
MNIST veri setini yükler, gerekli dönüşümleri uygular ve DataLoader nesnelerini döner.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loaders(batch_size=64, data_dir="./data"):
    """
    MNIST veri setini yükler ve DataLoader nesnelerini döner.
    Görüntüler 28x28'den 32x32'ye pad edilir (LeNet-5 için).
    Normalize: mean=0.5, std=0.5

    Args:
        batch_size (int): Batch boyutu
        data_dir (str): Verinin indirileceği dizin

    Returns:
        train_loader, test_loader, train_data, test_data
    """
    transform = transforms.Compose([
        transforms.Pad(2),                        # 28x28 -> 32x32
        transforms.ToTensor(),                    # PIL -> Tensor [0, 1]
        transforms.Normalize((0.5,), (0.5,)),     # [-1, 1] aralığına normalize
    ])

    train_data = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"[Dataset] Egitim seti boyutu : {len(train_data)}")
    print(f"[Dataset] Test seti boyutu   : {len(test_data)}")
    print(f"[Dataset] Goruntu boyutu     : {train_data[0][0].shape}")
    print(f"[Dataset] Sinif sayisi       : {len(train_data.classes)}")
    print(f"[Dataset] Batch boyutu       : {batch_size}")

    return train_loader, test_loader, train_data, test_data


def get_mnist_loaders_3ch(batch_size=64, data_dir="./data"):
    """
    MNIST veri setini 3 kanalli olarak yukler (Pretrained CNN modelleri icin).
    1-kanal gri tonlama -> 3-kanal (ayni kanal 3 kez tekrarlanir).
    32x32 boyut korunur (ResNet18 AdaptiveAvgPool sayesinde calisir).

    Args:
        batch_size (int): Batch boyutu
        data_dir (str): Verinin indirileceği dizin

    Returns:
        train_loader, test_loader, train_data, test_data
    """
    transform = transforms.Compose([
        transforms.Pad(2),                        # 28x28 -> 32x32
        transforms.ToTensor(),                    # PIL -> Tensor [0, 1]
        transforms.Normalize((0.5,), (0.5,)),     # [-1, 1] arasi normalize
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1ch -> 3ch
    ])

    train_data = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"[Dataset-3ch] Egitim seti boyutu : {len(train_data)}")
    print(f"[Dataset-3ch] Test seti boyutu   : {len(test_data)}")
    print(f"[Dataset-3ch] Goruntu boyutu     : {train_data[0][0].shape}")
    print(f"[Dataset-3ch] Sinif sayisi       : {len(train_data.classes)}")
    print(f"[Dataset-3ch] Batch boyutu       : {batch_size}")

    return train_loader, test_loader, train_data, test_data


if __name__ == "__main__":
    print("=== 1-Kanal (LeNet-5 icin) ===")
    train_loader, test_loader, _, _ = get_mnist_loaders()
    images, labels = next(iter(train_loader))
    print(f"Batch shape : {images.shape}")

    print("\n=== 3-Kanal (ResNet18 icin) ===")
    train_loader3, test_loader3, _, _ = get_mnist_loaders_3ch()
    images3, labels3 = next(iter(train_loader3))
    print(f"Batch shape : {images3.shape}")


