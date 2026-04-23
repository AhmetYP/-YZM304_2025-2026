"""
models.py — Model Mimarileri
Model 1: LeNet-5 (Temel CNN)
Diger modeller ilerleyen adimlarda eklenecektir.
"""

import torch.nn as nn
from collections import OrderedDict


# ============================================================================
# MODEL 1 — LeNet-5 (Temel Model)
# ============================================================================

class C1(nn.Module):
    """Birinci Evrisimli Blok: Conv2d(1,6,5) -> ReLU -> MaxPool2d(2,2)"""
    def __init__(self):
        super(C1, self).__init__()
        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        return self.c1(img)


class C2(nn.Module):
    """Ikinci Evrisimli Blok: Conv2d(6,16,5) -> ReLU -> MaxPool2d(2,2)"""
    def __init__(self):
        super(C2, self).__init__()
        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        return self.c2(img)


class C3(nn.Module):
    """Ucuncu Evrisimli Blok: Conv2d(16,120,5) -> ReLU"""
    def __init__(self):
        super(C3, self).__init__()
        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        return self.c3(img)


class F4(nn.Module):
    """Birinci Tam Bagli Katman: Linear(120,84) -> ReLU"""
    def __init__(self):
        super(F4, self).__init__()
        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        return self.f4(img)


class F5(nn.Module):
    """Cikis Katmani: Linear(84,10) -> LogSoftmax"""
    def __init__(self):
        super(F5, self).__init__()
        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        return self.f5(img)


class LeNet5(nn.Module):
    """
    LeNet-5 Mimarisi (Residual C2 bloklari ile)
    Input  : 1 x 32 x 32
    Output : 10 sinif

    Akis: C1 -> C2_1 + C2_2 (residual) -> C3 -> Flatten -> F4 -> F5
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1   = C1()
        self.c2_1 = C2()
        self.c2_2 = C2()
        self.c3   = C3()
        self.f4   = F4()
        self.f5   = F5()

    def forward(self, img):
        output = self.c1(img)

        x = self.c2_1(output)
        output = self.c2_2(output)
        output += x  # Residual baglanti

        output = self.c3(output)
        output = output.view(img.size(0), -1)  # Flatten
        output = self.f4(output)
        output = self.f5(output)
        return output


# ============================================================================
# MODEL 2 — LeNet-5 + BatchNorm
# ============================================================================

class LeNet5BN(nn.Module):
    """
    LeNet-5 + Batch Normalization
    Ayni hiperparametreler (kernel=5x5, filtre=6->16->120, FC=120->84->10)
    korunarak her Conv2d'den sonra BatchNorm2d, ilk FC'den sonra BatchNorm1d
    eklenmistir.

    Input  : 1 x 32 x 32
    Output : 10 sinif
    """
    def __init__(self):
        super(LeNet5BN, self).__init__()

        # C1: Conv2d(1,6,5) -> BN -> ReLU -> MaxPool
        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('bn1', nn.BatchNorm2d(6)),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

        # C2_1: Conv2d(6,16,5) -> BN -> ReLU -> MaxPool (residual dal 1)
        self.c2_1 = nn.Sequential(OrderedDict([
            ('c2_1', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('bn2_1', nn.BatchNorm2d(16)),
            ('relu2_1', nn.ReLU()),
            ('s2_1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

        # C2_2: Conv2d(6,16,5) -> BN -> ReLU -> MaxPool (residual dal 2)
        self.c2_2 = nn.Sequential(OrderedDict([
            ('c2_2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('bn2_2', nn.BatchNorm2d(16)),
            ('relu2_2', nn.ReLU()),
            ('s2_2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

        # C3: Conv2d(16,120,5) -> BN -> ReLU
        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('bn3', nn.BatchNorm2d(120)),
            ('relu3', nn.ReLU())
        ]))

        # F4: Linear(120,84) -> BN -> ReLU
        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('bn4', nn.BatchNorm1d(84)),
            ('relu4', nn.ReLU())
        ]))

        # F5: Linear(84,10) -> LogSoftmax
        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.c1(img)

        x = self.c2_1(output)
        output = self.c2_2(output)
        output += x  # Residual baglanti

        output = self.c3(output)
        output = output.view(img.size(0), -1)  # Flatten
        output = self.f4(output)
        output = self.f5(output)
        return output


# ============================================================================
# MODEL 3 — Pretrained ResNet18
# ============================================================================

def get_resnet18(num_classes=10, pretrained=True, freeze_features=True):
    """
    Pretrained ResNet18 modelini MNIST icin uyarlar.
    Son FC katmani num_classes cikisli olarak degistirilir.
    freeze_features=True ise sadece son FC egitilir (transfer learning).

    Args:
        num_classes (int): Cikis sinif sayisi
        pretrained (bool): ImageNet onceden egitilmis agirliklar
        freeze_features (bool): Ozellik cikarma katmanlarini dondur

    Returns:
        model (nn.Module): Uyarlanmis ResNet18
    """
    import torchvision.models as models

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Ozellik katmanlarini dondur (sadece FC egitilsin)
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    # Son FC katmanini degistir (512 -> num_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == "__main__":
    import torch

    # Model 1 testi
    model1 = LeNet5()
    dummy = torch.randn(2, 1, 32, 32)
    out1 = model1(dummy)
    print(f"LeNet5    cikti: {out1.shape}, param: {sum(p.numel() for p in model1.parameters()):,}")

    # Model 2 testi
    model2 = LeNet5BN()
    out2 = model2(dummy)
    print(f"LeNet5BN  cikti: {out2.shape}, param: {sum(p.numel() for p in model2.parameters()):,}")

    # Model 3 testi
    model3 = get_resnet18(num_classes=10, pretrained=False)
    dummy3 = torch.randn(2, 3, 32, 32)
    out3 = model3(dummy3)
    trainable = sum(p.numel() for p in model3.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model3.parameters())
    print(f"ResNet18  cikti: {out3.shape}, param: {total:,} (trainable: {trainable:,})")

