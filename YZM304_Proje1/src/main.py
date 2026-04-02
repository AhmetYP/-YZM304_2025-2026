from pathlib import Path
from YZM304_Proje1.src.model_sklearn import train_sklearn_mlp, evaluate_sklearn_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from YZM304_Proje1.src.model_pytorch import train_pytorch_model, evaluate_pytorch_model
from YZM304_Proje1.src.data_utils import load_data, split_data, standardize_data
from YZM304_Proje1.src.eda import (
    basic_info,
    plot_class_distribution,
    plot_feature_histograms,
    plot_correlation_heatmap,
)
from YZM304_Proje1.src.model_numpy import train_model, predict, train_model_2hidden, predict_2hidden

def evaluate_model(parameters, X_train_np, y_train_np, X_val_np, y_val_np, X_test_np, y_test_np):
    y_train_pred = predict(parameters, X_train_np).flatten()
    y_val_pred = predict(parameters, X_val_np).flatten()
    y_test_pred = predict(parameters, X_test_np).flatten()

    y_train_true = y_train_np.flatten()
    y_val_true = y_val_np.flatten()
    y_test_true = y_test_np.flatten()

    results = {
        "train_acc": accuracy_score(y_train_true, y_train_pred),
        "val_acc": accuracy_score(y_val_true, y_val_pred),
        "test_acc": accuracy_score(y_test_true, y_test_pred),
        "test_precision": precision_score(y_test_true, y_test_pred),
        "test_recall": recall_score(y_test_true, y_test_pred),
        "test_f1": f1_score(y_test_true, y_test_pred),
    }

    return results



def main():
    print("Program başladı")

    df = load_data()

    print("AŞAMA 1: VERİ ANALİZİ")
    basic_info(df)
    plot_class_distribution(df)
    plot_feature_histograms(df)
    plot_correlation_heatmap(df)

    print("\nAŞAMA 1.2: TRAIN / VAL / TEST SPLIT")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print("X_train shape:", X_train.shape)
    print("X_val shape  :", X_val.shape)
    print("X_test shape :", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape  :", y_val.shape)
    print("y_test shape :", y_test.shape)

    print("\nAŞAMA 1.3: STANDARDIZATION")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_data(
        X_train, X_val, X_test
    )

    print("Scaled train shape:", X_train_scaled.shape)
    print("Scaled val shape  :", X_val_scaled.shape)
    print("Scaled test shape :", X_test_scaled.shape)

    print("\nAŞAMA 1 TAMAMLANDI.")

    print("\nAŞAMA 2: BASELINE MLP MODEL")

    X_train_np = X_train_scaled.T
    X_val_np = X_val_scaled.T
    X_test_np = X_test_scaled.T

    y_train_np = y_train.values.reshape(1, -1)
    y_val_np = y_val.values.reshape(1, -1)
    y_test_np = y_test.values.reshape(1, -1)

    parameters, history = train_model(
        X_train_np,
        y_train_np,
        X_val=X_val_np,
        Y_val=y_val_np,
        n_h=6,
        n_steps=1000,
        lr=0.1
    )

    y_train_pred = predict(parameters, X_train_np).flatten()
    y_train_true = y_train_np.flatten()

    y_val_pred = predict(parameters, X_val_np).flatten()
    y_val_true = y_val_np.flatten()

    y_test_pred = predict(parameters, X_test_np).flatten()
    y_test_true = y_test_np.flatten()

    print("\n--- TRAIN SONUÇLARI ---")
    print("Accuracy :", accuracy_score(y_train_true, y_train_pred))
    print("Precision:", precision_score(y_train_true, y_train_pred))
    print("Recall   :", recall_score(y_train_true, y_train_pred))
    print("F1       :", f1_score(y_train_true, y_train_pred))

    print("\n--- VALIDATION SONUÇLARI ---")
    print("Accuracy :", accuracy_score(y_val_true, y_val_pred))
    print("Precision:", precision_score(y_val_true, y_val_pred))
    print("Recall   :", recall_score(y_val_true, y_val_pred))
    print("F1       :", f1_score(y_val_true, y_val_pred))

    print("\n--- TEST SONUÇLARI ---")
    print("Accuracy :", accuracy_score(y_test_true, y_test_pred))
    print("Precision:", precision_score(y_test_true, y_test_pred))
    print("Recall   :", recall_score(y_test_true, y_test_pred))
    print("F1       :", f1_score(y_test_true, y_test_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_true, y_test_pred))

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "train_val_loss.png")
    # plt.show()

    print("\nLoss grafiği kaydedildi:", figures_dir / "train_val_loss.png")

    print("\nAŞAMA 4.1: FARKLI HIDDEN NÖRON SAYILARI KARŞILAŞTIRMASI")

    hidden_sizes = [6, 16, 32]
    comparison_results = []

    for hidden_size in hidden_sizes:
        print(f"\nModel eğitiliyor: hidden_size = {hidden_size}")

        parameters_h, history_h = train_model(
            X_train_np,
            y_train_np,
            X_val=X_val_np,
            Y_val=y_val_np,
            n_h=hidden_size,
            n_steps=1000,
            lr=0.1
        )

        results = evaluate_model(
            parameters_h,
            X_train_np, y_train_np,
            X_val_np, y_val_np,
            X_test_np, y_test_np
        )

        results["hidden_size"] = hidden_size
        comparison_results.append(results)

        print("Train Accuracy:", results["train_acc"])
        print("Val Accuracy  :", results["val_acc"])
        print("Test Accuracy :", results["test_acc"])
        print("Test Precision:", results["test_precision"])
        print("Test Recall   :", results["test_recall"])
        print("Test F1       :", results["test_f1"])

    print("\n--- HIDDEN SIZE KARŞILAŞTIRMA TABLOSU ---")
    for row in comparison_results:
        print(
            f"hidden={row['hidden_size']} | "
            f"train_acc={row['train_acc']:.4f} | "
            f"val_acc={row['val_acc']:.4f} | "
            f"test_acc={row['test_acc']:.4f} | "
            f"precision={row['test_precision']:.4f} | "
            f"recall={row['test_recall']:.4f} | "
            f"f1={row['test_f1']:.4f}"
        )
    print("\nAŞAMA 4.2: 2 HIDDEN LAYER MODEL")

    parameters_2h, history_2h = train_model_2hidden(
        X_train_np,
        y_train_np,
        X_val=X_val_np,
        Y_val=y_val_np,
        n_h1=16,
        n_h2=8,
        n_steps=2000,
        lr=0.01
    )

    y_train_pred_2h = predict_2hidden(parameters_2h, X_train_np).flatten()
    y_val_pred_2h = predict_2hidden(parameters_2h, X_val_np).flatten()
    y_test_pred_2h = predict_2hidden(parameters_2h, X_test_np).flatten()

    print("\n--- 2 HIDDEN LAYER TRAIN SONUÇLARI ---")
    print("Accuracy :", accuracy_score(y_train_true, y_train_pred_2h))
    print("Precision:", precision_score(y_train_true, y_train_pred_2h))
    print("Recall   :", recall_score(y_train_true, y_train_pred_2h))
    print("F1       :", f1_score(y_train_true, y_train_pred_2h))

    print("\n--- 2 HIDDEN LAYER VALIDATION SONUÇLARI ---")
    print("Accuracy :", accuracy_score(y_val_true, y_val_pred_2h))
    print("Precision:", precision_score(y_val_true, y_val_pred_2h))
    print("Recall   :", recall_score(y_val_true, y_val_pred_2h))
    print("F1       :", f1_score(y_val_true, y_val_pred_2h))

    print("\n--- 2 HIDDEN LAYER TEST SONUÇLARI ---")
    print("Accuracy :", accuracy_score(y_test_true, y_test_pred_2h))
    print("Precision:", precision_score(y_test_true, y_test_pred_2h))
    print("Recall   :", recall_score(y_test_true, y_test_pred_2h))
    print("F1       :", f1_score(y_test_true, y_test_pred_2h))

    print("\n2 Hidden Layer Confusion Matrix:")
    print(confusion_matrix(y_test_true, y_test_pred_2h))

    print("\nAŞAMA 5: SCIKIT-LEARN MLPClassifier")

    sklearn_model = train_sklearn_mlp(
        X_train_scaled,
        y_train,
        hidden_layer_sizes=(16, 8),
        max_iter=1000,
        random_state=42
    )

    sklearn_train_results = evaluate_sklearn_model(sklearn_model, X_train_scaled, y_train)
    sklearn_val_results = evaluate_sklearn_model(sklearn_model, X_val_scaled, y_val)
    sklearn_test_results = evaluate_sklearn_model(sklearn_model, X_test_scaled, y_test)

    print("\n--- SKLEARN TRAIN SONUÇLARI ---")
    print("Accuracy :", sklearn_train_results["accuracy"])
    print("Precision:", sklearn_train_results["precision"])
    print("Recall   :", sklearn_train_results["recall"])
    print("F1       :", sklearn_train_results["f1"])

    print("\n--- SKLEARN VALIDATION SONUÇLARI ---")
    print("Accuracy :", sklearn_val_results["accuracy"])
    print("Precision:", sklearn_val_results["precision"])
    print("Recall   :", sklearn_val_results["recall"])
    print("F1       :", sklearn_val_results["f1"])

    print("\n--- SKLEARN TEST SONUÇLARI ---")
    print("Accuracy :", sklearn_test_results["accuracy"])
    print("Precision:", sklearn_test_results["precision"])
    print("Recall   :", sklearn_test_results["recall"])
    print("F1       :", sklearn_test_results["f1"])

    print("\nSKLEARN Confusion Matrix:")
    print(sklearn_test_results["confusion_matrix"])

    print("\nAŞAMA 6: PYTORCH MLP")

    pytorch_model, pytorch_history = train_pytorch_model(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        epochs=500,
        lr=0.001
    )

    pytorch_train_results = evaluate_pytorch_model(pytorch_model, X_train_scaled, y_train)
    pytorch_val_results = evaluate_pytorch_model(pytorch_model, X_val_scaled, y_val)
    pytorch_test_results = evaluate_pytorch_model(pytorch_model, X_test_scaled, y_test)

    print("\n--- PYTORCH TRAIN SONUÇLARI ---")
    print("Accuracy :", pytorch_train_results["accuracy"])
    print("Precision:", pytorch_train_results["precision"])
    print("Recall   :", pytorch_train_results["recall"])
    print("F1       :", pytorch_train_results["f1"])

    print("\n--- PYTORCH VALIDATION SONUÇLARI ---")
    print("Accuracy :", pytorch_val_results["accuracy"])
    print("Precision:", pytorch_val_results["precision"])
    print("Recall   :", pytorch_val_results["recall"])
    print("F1       :", pytorch_val_results["f1"])

    print("\n--- PYTORCH TEST SONUÇLARI ---")
    print("Accuracy :", pytorch_test_results["accuracy"])
    print("Precision:", pytorch_test_results["precision"])
    print("Recall   :", pytorch_test_results["recall"])
    print("F1       :", pytorch_test_results["f1"])

    print("\nPYTORCH Confusion Matrix:")
    print(pytorch_test_results["confusion_matrix"])
if __name__ == "__main__":
    main()