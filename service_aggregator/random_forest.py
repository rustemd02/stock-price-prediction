from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sqlalchemy.orm import Session, sessionmaker
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from pymystem3 import Mystem

from service_aggregator.dictionary_words import CORPUS_COLS
from service_repository.crud import get_corpus_by_date_range
from service_repository.models import NewsModel
from service_repository.database import engine
from joblib import load, dump

news = pd.read_csv('../last-work/service_aggregator/news_with_probabilities.csv')

# Загрузка данных индекса с правильным разделителем и именами столбцов
index_data = pd.read_csv(
    '../last-work/sentiment-analysis/IMOEXv2.csv',
    sep=';',
    names=['TICKER', 'PER', 'DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL'],
    skiprows=1  # Пропуск первой строки, если она содержит заголовки
)
index_data['DATE'] = pd.to_datetime(index_data['DATE'], format='%Y%m%d')
index_data['returns'] = index_data['CLOSE'].pct_change()
index_data['target'] = (index_data['returns'] > 0).astype(int)

# Подготовка данных новостей
news['post_dttm'] = pd.to_datetime(news['post_dttm'])
news['date'] = news['post_dttm'].dt.date

# Приведение поля 'date' к типу datetime для согласования с полем 'DATE' из index_data
news['date'] = pd.to_datetime(news['date'])


# Функция для агрегации данных и расчета средних значений тональностей
def aggregate_news(news, news_type):
    news_type_data = news[news['type'] == news_type]
    aggregated_data = news_type_data.groupby('date').agg({
        '-1': 'sum',
        '0': 'sum',
        '1': 'sum'
    }).reset_index()

    # Суммы тональностей
    total_negative = aggregated_data['-1'].sum()
    total_neutral = aggregated_data['0'].sum()
    total_positive = aggregated_data['1'].sum()

    # Средние значения тональностей
    aggregated_data[f'neg_mean_{news_type}'] = aggregated_data['-1'] / total_negative
    aggregated_data[f'neutral_mean_{news_type}'] = aggregated_data['0'] / total_neutral
    aggregated_data[f'pos_mean_{news_type}'] = aggregated_data['1'] / total_positive

    # Удаление ненужных столбцов
    aggregated_data = aggregated_data.drop(columns=['-1', '0', '1'])

    return aggregated_data


# Агрегация данных для экономических и политических новостей
economic_news_aggregated = aggregate_news(news, 'economic')
political_news_aggregated = aggregate_news(news, 'political')

# Объединение данных экономических и политических новостей по дате
news_combined = pd.merge(
    economic_news_aggregated,
    political_news_aggregated,
    on='date',
    how='outer'
)

# Объединение данных новостей и индекса
merged_data = news_combined.merge(index_data[['DATE', 'target']], left_on='date', right_on='DATE', how='inner')
merged_data = merged_data.drop(columns=['DATE'])

# Подготовка признаков и целевой переменной
X = merged_data[['neg_mean_economic', 'neutral_mean_economic', 'pos_mean_economic',
                 'neg_mean_political', 'neutral_mean_political', 'pos_mean_political']]
y = merged_data['target']

print(merged_data)
# Сохранение итогового DataFrame в CSV
merged_data.to_csv('../last-work/merged_data.csv', index=False)

merged_data = pd.read_csv('../last-work/merged_data.csv')
X = merged_data[['neg_mean_economic', 'neutral_mean_economic', 'pos_mean_economic',
                 'neg_mean_political', 'neutral_mean_political', 'pos_mean_political']]
y = merged_data['target']
print(merged_data)
# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели случайного леса
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_y_pred = rf_clf.predict(X_test)

dump(rf_clf, 'model/random_forest.joblib')
# Оценка качества модели случайного леса
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_report = classification_report(y_test, rf_y_pred)

print(f'Random Forest Accuracy: {rf_accuracy:.2f}')
print('Random Forest Classification Report:')
print(rf_report)


def preprocess_text(text):
    tokenizer = WordPunctTokenizer()
    mystem = Mystem()
    stop_words = set(stopwords.words('russian'))
    tokens = tokenizer.tokenize(text.lower())
    cleaned_text = ' '.join([mystem.lemmatize(word)[0] for word in tokens if word.isalpha() and word not in stop_words])
    return cleaned_text
def prepare_corpus(
        corpus: List[NewsModel], cols: List[str] = None) -> pd.DataFrame:
    """Преобразование корпуса текстов из orm моделей в pandas DF"""
    cols = cols or ["title", "type", "post_dttm"]
    corpus_sel_col = [{"title": news.title, "type": news.type, "post_dttm": news.post_dttm} for news in corpus]
    return pd.DataFrame(corpus_sel_col, columns=cols)

def get_probabilities(text):
    cleaned_text = preprocess_text(text)
    vectorizer = load('../service_aggregator/model/vectorizer.joblib')
    X_new = vectorizer.transform([cleaned_text])
    clf = load('../service_aggregator/model/best_classifier.joblib')
    y_prob_new = clf.predict_proba(X_new)

    # Создание словаря с вероятностями и метками классов
    class_labels = clf.classes_
    probabilities = {str(label): prob for label, prob in zip(class_labels, y_prob_new[0])}
    return probabilities

# def run():
#     SessionLocal = sessionmaker(bind=engine)
#     db: Session = SessionLocal()
#
#     corpus = get_corpus_by_date_range(db, '2024-05-05 00:00:00.000000', '2024-06-05 00:00:00.000000')
#     news = prepare_corpus(corpus)
#     print(news)
#     probabilities_list = []
#     for title in news['title']:
#         probabilities = get_probabilities(title)
#         probabilities_list.append(probabilities)
#
#     # Преобразование вероятностей в DataFrame
#     probabilities_df = pd.DataFrame(probabilities_list)
#     news = pd.concat([news, probabilities_df], axis=1)
#     news.to_csv('news_with_probabilities.csv', index=False)
#
# if __name__ == "__main__":
#     run()
