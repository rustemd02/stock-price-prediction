import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
from datetime import datetime
import csv
from nltk.corpus import stopwords
import re
from joblib import load
from pymystem3 import Mystem

nltk.download('wordnet')
# Загрузка данных из файла
data = pd.read_csv('../service_aggregator/news_with_probabilities.csv')

data['date'] = pd.to_datetime(data['post_dttm']).dt.date


# Инициализация лемматизатора NLTK
mystem = Mystem()
tokenizer = WordPunctTokenizer()
stop_words = set(stopwords.words('russian'))

def preprocess_text(text):
    tokens = tokenizer.tokenize(text.lower())
    cleaned_text = ' '.join([mystem.lemmatize(word)[0] for word in tokens if word.isalpha() and word not in stop_words])
    return cleaned_text


def get_probabilities(df):
    df['cleaned_text'] = df['title'].apply(preprocess_text)
    vectorizer = load('model/vectorizer.joblib')
    X_new = vectorizer.transform(df['cleaned_text'])
    clf = load('model/best_classifier.joblib')
    # Предсказание вероятностей
    y_prob_new = clf.predict_proba(X_new)
    class_labels = clf.classes_

    # Получение класса с самой высокой вероятностью
    best_labels = [class_labels[probs.argmax()] for probs in y_prob_new]
    df['lable'] = best_labels

    return df

def save_sentiment_scores_to_csv(news_data):
    # Создаем DataFrame из новостей
    df = pd.DataFrame(news_data)

    # Подготавливаем данные для сохранения
    df_to_save = df[['post_dttm', 'lable']].rename(columns={'post_dttm': 'TRADEDATE', 'lable': 'SentScore'})

    # Определяем путь к файлу относительно текущего скрипта
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, '..', 'static', 'news_sentiment.csv')

    # Проверяем, существует ли файл
    file_exists = os.path.isfile(file_path)

    # Сохраняем данные в файл, добавляя новые строки
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['TRADEDATE', 'SentScore']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Если файл не существует, записываем заголовки
        if not file_exists:
            writer.writeheader()

        for index, row in df_to_save.iterrows():
            writer.writerow(row.to_dict())

    print(f"Sentiment scores saved to {file_path}")

def calculate_and_save_sentiment_scores(news_data):
    # Очистка и форматирование данных
    news_data = format_news_data(news_data)

    # Убедитесь, что news_data является DataFrame
    if not isinstance(news_data, pd.DataFrame):
        news_data = pd.DataFrame(news_data)

    # Предсказание lable
    news_data = get_probabilities(news_data)

    # Сохранение результатов в CSV файл
    save_sentiment_scores_to_csv(news_data)

    # Преобразование DataFrame в список словарей
    return news_data.to_dict(orient='records')

def format_news_data(news_data):
    for row in news_data:
        row['title'] = row['title'].strip()
        if isinstance(row['post_dttm'], str):
            row['post_dttm'] = datetime.strptime(row['post_dttm'].strip(), "%Y-%m-%d %H:%M:%S%z")
        row['post_dttm'] = row['post_dttm'].strftime("%Y-%m-%d")
    return news_data


