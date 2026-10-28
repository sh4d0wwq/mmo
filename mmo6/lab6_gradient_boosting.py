"""Лабораторная работа 6. Градиентный бустинг (вариант 17, нечётный)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 17
DATA_PATH = Path(__file__).resolve().parent / "Bankloan.csv"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Размер сырых данных:", df.shape)
    print("Пропуски по столбцам:\n", df.isna().sum())
    labeled = df.dropna().copy()
    print("Размер после удаления пропусков:", labeled.shape)
    print("Доли классов default:\n", labeled["default"].value_counts(normalize=True))
    return labeled


def prepare_xy(labeled):
    X = labeled.drop(columns=["default"])
    y = labeled["default"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train, X_test, X_train_s, X_test_s, y_train, y_test, list(X.columns)


def metrics_dict(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_score),
    }


def print_metrics(title, m):
    print(f"\n=== {title} ===")
    for k, v in m.items():
        print(f"{k}: {v:.4f}")


def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Нет дефолта", "Дефолт"],
        yticklabels=["Нет дефолта", "Дефолт"],
    )
    plt.xlabel("Прогноз")
    plt.ylabel("Истина")
    plt.title(title)
    plt.tight_layout()
    path = FIG_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print("Сохранена матрица ошибок:", path)
    return cm


def plot_learning_study(X_train, y_train, X_test, y_test):
    rates = [0.01, 0.05, 0.1, 0.2]
    n_trees = [50, 100, 150, 200, 300]
    rows = []
    for lr in rates:
        for n in n_trees:
            model = GradientBoostingClassifier(
                n_estimators=n,
                learning_rate=lr,
                max_depth=3,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rows.append(
                {
                    "learning_rate": lr,
                    "n_estimators": n,
                    "Accuracy": accuracy_score(y_test, pred),
                    "F1": f1_score(y_test, pred),
                }
            )
    study = pd.DataFrame(rows)
    print("\n=== Влияние learning_rate и n_estimators ===")
    print(study.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for lr in rates:
        part = study[study["learning_rate"] == lr]
        axes[0].plot(part["n_estimators"], part["Accuracy"], marker="o", label=f"lr={lr}")
        axes[1].plot(part["n_estimators"], part["F1"], marker="o", label=f"lr={lr}")
    axes[0].set_xlabel("n_estimators")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs число деревьев")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("n_estimators")
    axes[1].set_ylabel("F1")
    axes[1].set_title("F1 vs число деревьев")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIG_DIR / "lr_n_estimators.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("Сохранён график влияния параметров:", path)
    return study


def main():
    labeled = load_data()
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, feature_names = prepare_xy(
        labeled
    )
    print(
        f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, "
        f"классы train: {dict(y_train.value_counts())}"
    )

    # 1. Базовая модель (параметры по умолчанию)
    base = GradientBoostingClassifier(random_state=RANDOM_STATE)
    base.fit(X_train, y_train)
    base_pred = base.predict(X_test)
    base_score = base.predict_proba(X_test)[:, 1]
    base_m = metrics_dict(y_test, base_pred, base_score)
    print_metrics("Базовый GradientBoostingClassifier", base_m)
    print(classification_report(y_test, base_pred, digits=4))
    base_cm = plot_confusion(
        y_test,
        base_pred,
        "Матрица ошибок: базовый GBM",
        "cm_base.png",
    )
    print("Confusion matrix (base):\n", base_cm)

    # 2. GridSearchCV
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.05, 0.1, 0.2],
        "min_samples_leaf": [1, 3],
    }
    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    print("\nЛучшие параметры GridSearchCV:", grid.best_params_)
    print("Лучший F1 (CV):", round(grid.best_score_, 4))

    best = grid.best_estimator_
    best_pred = best.predict(X_test)
    best_score = best.predict_proba(X_test)[:, 1]
    best_m = metrics_dict(y_test, best_pred, best_score)
    print_metrics("GBM после GridSearchCV", best_m)
    print(classification_report(y_test, best_pred, digits=4))
    best_cm = plot_confusion(
        y_test,
        best_pred,
        "Матрица ошибок: GBM после GridSearchCV",
        "cm_best.png",
    )
    print("Confusion matrix (best):\n", best_cm)

    # 3. Влияние learning_rate и числа деревьев
    study = plot_learning_study(X_train, y_train, X_test, y_test)

    # Сравнение на масштабированных данных (не обязательно, но полезно)
    base_scaled = GradientBoostingClassifier(random_state=RANDOM_STATE)
    base_scaled.fit(X_train_s, y_train)
    scaled_pred = base_scaled.predict(X_test_s)
    scaled_m = metrics_dict(
        y_test, scaled_pred, base_scaled.predict_proba(X_test_s)[:, 1]
    )
    print_metrics("Базовый GBM на StandardScaler", scaled_m)

    # Сохранить сводку
    summary = pd.DataFrame(
        [
            {"Модель": "Базовый GBM", **base_m},
            {"Модель": "GBM + GridSearchCV", **best_m},
            {"Модель": "Базовый GBM (scaled)", **scaled_m},
        ]
    )
    summary_path = Path(__file__).resolve().parent / "results_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    study_path = Path(__file__).resolve().parent / "lr_study.csv"
    study.to_csv(study_path, index=False, float_format="%.6f")
    print("\nСводка сохранена:", summary_path)
    print(summary.round(4).to_string(index=False))

    # Важность признаков лучшей модели (для справки)
    imp = pd.Series(best.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )
    print("\nВажность признаков (лучшая модель):\n", imp.round(4))


if __name__ == "__main__":
    main()
