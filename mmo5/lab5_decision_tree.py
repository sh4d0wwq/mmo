"""Лабораторная работа 5. Самописное дерево решений (классификация)."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 17
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 3
MIN_SAMPLES_SPLIT = 6


class Node:
    def __init__(self, index, t, true_branch, false_branch):
        self.index = index
        self.t = t
        self.true_branch = true_branch
        self.false_branch = false_branch


class Leaf:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.prediction = self.predict()

    def predict(self):
        classes = {}
        for label in self.labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
        return max(classes, key=classes.get)


def gini(labels):
    """Критерий Джини: 1 - sum(p_k^2)."""
    labels = np.asarray(labels)
    n = labels.shape[0]
    if n == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / n
    return 1.0 - float(np.sum(probabilities ** 2))


def gain(left_labels, right_labels, root_gini):
    """Прирост информации (уменьшение неопределённости Джини)."""
    n = len(left_labels) + len(right_labels)
    if n == 0:
        return 0.0
    p_left = len(left_labels) / n
    p_right = len(right_labels) / n
    return root_gini - p_left * gini(left_labels) - p_right * gini(right_labels)


def split(data, labels, column_index, t):
    left = np.where(data[:, column_index] <= t)
    right = np.where(data[:, column_index] > t)
    true_data = data[left]
    false_data = data[right]
    true_labels = labels[left]
    false_labels = labels[right]
    return true_data, false_data, true_labels, false_labels


def find_best_split(data, labels, min_samples_leaf=MIN_SAMPLES_LEAF):
    root_gini = gini(labels)
    best_gain = 0.0
    best_t = None
    best_index = None
    n_features = data.shape[1]

    for index in range(n_features):
        # Пороги как середины между соседними уникальными значениями (как в sklearn)
        values = np.unique(data[:, index])
        if values.shape[0] < 2:
            continue
        t_values = (values[:-1] + values[1:]) / 2.0

        for t in t_values:
            true_data, false_data, true_labels, false_labels = split(
                data, labels, index, t
            )
            # Критерий останова: лист не меньше min_samples_leaf
            if (
                true_data.shape[0] < min_samples_leaf
                or false_data.shape[0] < min_samples_leaf
            ):
                continue

            current_gain = gain(true_labels, false_labels, root_gini)
            # При равном приросте берём последний кандидат (как в sklearn)
            if current_gain >= best_gain and current_gain > 0:
                best_gain, best_t, best_index = current_gain, t, index

    return best_gain, best_t, best_index


def build_tree(
    data,
    labels,
    depth=0,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
):
    # Критерии останова:
    # 1) достигнута максимальная глубина
    # 2) в узле слишком мало объектов для разбиения
    # 3) узел чистый / нет прироста качества
    if depth >= max_depth or data.shape[0] < min_samples_split:
        return Leaf(data, labels)

    split_gain, t, index = find_best_split(
        data, labels, min_samples_leaf=min_samples_leaf
    )

    if split_gain == 0 or t is None or index is None:
        return Leaf(data, labels)

    true_data, false_data, true_labels, false_labels = split(data, labels, index, t)
    true_branch = build_tree(
        true_data,
        true_labels,
        depth=depth + 1,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    false_branch = build_tree(
        false_data,
        false_labels,
        depth=depth + 1,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    return Node(index, t, true_branch, false_branch)


def classify_object(obj, node):
    if isinstance(node, Leaf):
        return node.prediction
    if obj[node.index] <= node.t:
        return classify_object(obj, node.true_branch)
    return classify_object(obj, node.false_branch)


def predict(data, tree):
    classes = []
    for obj in data:
        classes.append(classify_object(obj, tree))
    return classes


def accuracy_metric(actual, predicted):
    """Доля правильных ответов."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return float(np.sum(actual == predicted) / actual.shape[0])


def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Прогноз:", node.prediction)
        return
    print(spacing + "Индекс", str(node.index), "<=", str(node.t))
    print(spacing + "--> True:")
    print_tree(node.true_branch, spacing + "  ")
    print(spacing + "--> False:")
    print_tree(node.false_branch, spacing + "  ")


def tree_depth(node):
    if isinstance(node, Leaf):
        return 0
    return 1 + max(tree_depth(node.true_branch), tree_depth(node.false_branch))


def count_leaves(node):
    if isinstance(node, Leaf):
        return 1
    return count_leaves(node.true_branch) + count_leaves(node.false_branch)


def main():
    X, y = make_classification(
        n_samples=400,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=RANDOM_STATE,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Train class balance:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Test class balance:", dict(zip(*np.unique(y_test, return_counts=True))))

    my_tree = build_tree(
        X_train,
        y_train,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
    )

    y_train_pred = predict(X_train, my_tree)
    y_test_pred = predict(X_test, my_tree)
    acc_train = accuracy_metric(y_train, y_train_pred)
    acc_test = accuracy_metric(y_test, y_test_pred)

    print("\n--- Самописное дерево ---")
    print(f"Глубина: {tree_depth(my_tree)}, листьев: {count_leaves(my_tree)}")
    print(f"Accuracy train: {acc_train:.6f}")
    print(f"Accuracy test:  {acc_test:.6f}")
    print("Структура дерева:")
    print_tree(my_tree)

    sk_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
    )
    sk_tree.fit(X_train, y_train)
    sk_train_pred = sk_tree.predict(X_train)
    sk_test_pred = sk_tree.predict(X_test)
    sk_acc_train = accuracy_score(y_train, sk_train_pred)
    sk_acc_test = accuracy_score(y_test, sk_test_pred)

    print("\n--- sklearn DecisionTreeClassifier ---")
    print(f"Глубина: {sk_tree.get_depth()}, листьев: {sk_tree.get_n_leaves()}")
    print(f"Accuracy train: {sk_acc_train:.6f}")
    print(f"Accuracy test:  {sk_acc_test:.6f}")

    train_match = np.array_equal(y_train_pred, sk_train_pred)
    test_match = np.array_equal(y_test_pred, sk_test_pred)
    print("\n--- Сравнение прогнозов ---")
    print(f"Совпадение train: {train_match}")
    print(f"Совпадение test:  {test_match}")
    print(f"|Δ accuracy train|: {abs(acc_train - sk_acc_train):.6e}")
    print(f"|Δ accuracy test|:  {abs(acc_test - sk_acc_test):.6e}")

    if not (train_match and test_match):
        # Покажем долю совпадений, если полное равенство не достигнуто
        train_eq = float(np.mean(np.asarray(y_train_pred) == sk_train_pred))
        test_eq = float(np.mean(np.asarray(y_test_pred) == sk_test_pred))
        print(f"Доля одинаковых ответов train: {train_eq:.6f}")
        print(f"Доля одинаковых ответов test:  {test_eq:.6f}")


if __name__ == "__main__":
    main()
