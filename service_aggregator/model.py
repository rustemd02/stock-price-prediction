import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from joblib import dump

from service_aggregator.dictionary_words import get_ngrams


def assign_label(score: float, epsilon: float) -> int:
    if score > epsilon:
        return 1
    elif score < -epsilon:
        return -1
    else:
        return 0


def search_best_epsilon(X, labels, epsilon_candidates, n_splits=5):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    base_clf = MultinomialNB()

    best_epsilon = None
    best_score = 0.0

    for eps in epsilon_candidates:
        fold_accuracies = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]

            y_train = [assign_label(val, eps) for val in labels[train_idx]]
            y_test = [assign_label(val, eps) for val in labels[test_idx]]

            base_clf.fit(X_train, y_train)

            y_pred = base_clf.predict(X_test)

            fold_acc = accuracy_score(y_test, y_pred)
            fold_accuracies.append(fold_acc)

        mean_acc = np.mean(fold_accuracies)

        if mean_acc > best_score:
            best_score = mean_acc
            best_epsilon = eps

    return best_epsilon, best_score



if __name__ == "__main__":
    df = pd.read_csv("../static/newsv6.csv", header=None)
    df.columns = ['uuid', 'title', 'post_dttm', 'url',
                  'processed_dttm', 'type', 'label']

    df['label'] = pd.to_numeric(df['label'], errors='coerce')

    X, feature_names, vectorizer = get_ngrams(df, ngram_range=(1, 2))

    labels = df['label'].values

    epsilon_values = np.linspace(0.0, 0.5, 1000)

    best_epsilon, best_cv_score = search_best_epsilon(
        X, labels, epsilon_values, n_splits=5
    )
    print(f"Средняя accuracy на 5 фолдах = {best_cv_score:.3f}")

    df['score'] = df['label'].apply(lambda x: assign_label(x, best_epsilon))
    y = df['score'].values

    print("Распределение классов после assign_label:")
    print(df['score'].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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

    dump(best_clf, 'model/best_classifier.joblib')
    dump(vectorizer, 'model/vectorizer.joblib')
