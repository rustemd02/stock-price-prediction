import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from joblib import dump

from service_aggregator.dictionary_words import get_ngrams


# Ваш импорт get_ngrams(...) и т.д.


########################################
# 1. Ручная функция assign_label
########################################

def assign_label(score: float, epsilon: float) -> int:
    """Преобразует вещественное значение score в один из классов {-1, 0, 1},
       в зависимости от порога epsilon."""
    if score > epsilon:
        return 1
    elif score < -epsilon:
        return -1
    else:
        return 0


########################################
# 2. Функция для поиска лучшего epsilon
########################################

def search_best_epsilon(X, labels, epsilon_candidates, n_splits=5):
    """
    Подбираем лучший epsilon, оценивая, как хорошо одна ИЗ моделей (в примере NaiveBayes)
    предсказывает на кросс-валидации (KFold).

    Аргументы:
    ---------
    X : array-like или разреженная матрица
        Матрица признаков (уже после get_ngrams).
    labels : array-like
        Вещественные тональности (df['label']), еще не дискретизированные.
    epsilon_candidates : iterable
        Список/массив значений epsilon, которые перебираем.
    n_splits : int
        Количество фолдов в KFold.

    Возвращает:
    -----------
    (best_epsilon, best_score) : кортеж
        Лучшая найденная epsilon и средняя accuracy на фолдах.
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Для перебора epsilon используем простую модель; пусть это будет NB.
    # Можно подставить любой один базовый классификатор, на котором будем "мерить"
    # качество разбиения (LogisticRegression, SVC и т.д.).
    base_clf = MultinomialNB()

    best_epsilon = None
    best_score = 0.0

    for eps in epsilon_candidates:
        fold_accuracies = []

        for train_idx, test_idx in kf.split(X):
            # Формируем тренировочные и тестовые выборки
            X_train, X_test = X[train_idx], X[test_idx]

            # "assign_label" для train и test
            y_train = [assign_label(val, eps) for val in labels[train_idx]]
            y_test = [assign_label(val, eps) for val in labels[test_idx]]

            # Обучаем выбранную модель
            base_clf.fit(X_train, y_train)

            # Предсказываем
            y_pred = base_clf.predict(X_test)

            # Оцениваем точность на фолде
            fold_acc = accuracy_score(y_test, y_pred)
            fold_accuracies.append(fold_acc)

        # Средняя accuracy по всем фолдам
        mean_acc = np.mean(fold_accuracies)

        if mean_acc > best_score:
            best_score = mean_acc
            best_epsilon = eps

    return best_epsilon, best_score


########################################
# 3. Основной код
########################################

if __name__ == "__main__":
    # 1) Загружаем данные
    df = pd.read_csv("../static/newsv6.csv", header=None)
    df.columns = ['uuid', 'title', 'post_dttm', 'url',
                  'processed_dttm', 'type', 'label']

    # Преобразуем метки в float (если были строки и т.п.)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')

    # 2) Генерируем n-граммы (X, vectorizer, ...)
    #    Предполагается, что get_ngrams возвращает (X, feature_names, vectorizer)
    X, feature_names, vectorizer = get_ngrams(df, ngram_range=(1, 2))

    # Наша "исходная" вещественная тональность
    labels = df['label'].values

    # 3) Перебираем значения epsilon
    epsilon_values = np.linspace(0.0, 0.5, 1000)

    best_epsilon, best_cv_score = search_best_epsilon(
        X, labels, epsilon_values, n_splits=5
    )
    print(f"Лучший epsilon = {best_epsilon}, средняя accuracy на 5 фолдах = {best_cv_score:.3f}")

    # 4) Теперь, когда epsilon найден,
    #    формируем "итоговые" метки (y) для обучения финальной модели
    df['score'] = df['label'].apply(lambda x: assign_label(x, best_epsilon))
    y = df['score'].values

    print("Распределение классов после assign_label:")
    print(df['score'].value_counts())

    # 5) Делим на train/test (если данные не очень большие и нет специфики временного ряда)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6) Тренируем несколько классификаторов
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Support Vector Machine": SVC(probability=True)
    }

    best_clf_name = None
    best_clf_score = 0.0
    best_clf = None

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        if acc > best_clf_score:
            best_clf_score = acc
            best_clf_name = name
            best_clf = clf

    print(f"\nЛучший классификатор: {best_clf_name} c accuracy = {best_clf_score:.3f}")

    # 7) Сохраняем лучший классификатор и vectorizer
    dump(best_clf, 'model/best_classifier.joblib')
    dump(vectorizer, 'model/vectorizer.joblib')

    print("Готово. Модель и vectorizer сохранены.")
