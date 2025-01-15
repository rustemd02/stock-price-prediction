import  pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from joblib import dump
from service_aggregator.dictionary_words import get_ngrams

# Загрузка данных
df = pd.read_csv("../last-work/sentiment-analysis/static/newsv6.csv", header=None)
df.columns = ['uuid', 'title', 'post_dttm', 'url', 'processed_dttm', '', 'label', "type"]

def assign_label(score, epsilon):
    if score > epsilon:
        return 1
    elif score < -epsilon:
        return -1
    else:
        return 0

X, feature_names, vectorizer = get_ngrams(df, ngram_range=(1, 2))
weights = df['label']

epsilon_values = np.linspace(0.0, 0.5, 1000)

# Создадим KFold для кросс-валидации
kf = KFold(n_splits=5, shuffle=True, random_state=0)

best_epsilon = None
best_accuracy = 0

for epsilon in epsilon_values:
    accuracies = []

    df['score'] = df['label'].apply(lambda x: assign_label(x, epsilon))
    y = df['score']

    for train_index, test_index in kf.split(X):
        X_fold_train, X_fold_test = X[train_index], X[test_index]
        y_fold_train, y_fold_test = y.iloc[train_index], y.iloc[test_index]

        # Присваиваем метки тональности на основе текущего значения epsilon
        y_pred = np.where(weights[test_index] > epsilon, 1, np.where(weights[test_index] < -epsilon, -1, 0))

        # Оцениваем точность на текущем фолде
        accuracy = accuracy_score(y_fold_test, y_pred)
        accuracies.append(accuracy)

    # Средняя точность для текущего значения epsilon
    mean_accuracy = np.mean(accuracies)

    # Обновляем лучшее значение epsilon, если текущая точность выше
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_epsilon = epsilon

print(f'Лучшее значение epsilon: {best_epsilon}')
print(f'Наилучшая точность: {best_accuracy}')

# Пересчитываем метки на основе лучшего значения epsilon
df['score'] = df['label'].apply(lambda x: assign_label(x, 0.2))
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Тренируем модели на основе пересчитанных меток
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine": SVC(probability=True)
}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Results for {name}:")
    print(classification_report(y_test, y_pred, zero_division=0))

# Выбор оптимального классификатора на основе метрик
best_clf_name = None
best_clf_score = 0
for name, clf in classifiers.items():
    score = clf.score(X_test, y_test)
    if score > best_clf_score:
        best_clf_score = score
        best_clf_name = name

print(f"Best classifier: {best_clf_name} with score {best_clf_score}")

best_clf = classifiers[best_clf_name]
dump(best_clf, '../service_aggregator/model/best_classifier.joblib')
dump(vectorizer, '../service_aggregator/model/vectorizer.joblib')
