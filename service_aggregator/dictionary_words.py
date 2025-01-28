import csv
from typing import List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict
from pymystem3 import Mystem
from sqlalchemy.orm import Session, sessionmaker
import math
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from service_repository.crud import get_corpus
from service_repository.database import engine
from nltk.tokenize import WordPunctTokenizer
from service_repository.models import NewsModel

mystem = Mystem()
tokenizer = WordPunctTokenizer()
stop_words = set(stopwords.words('russian'))

CORPUS_COLS = ["title", "type"]


def prepare_corpus(
        corpus: List[Tuple[NewsModel]], cols: List[str] = None) -> pd.DataFrame:
    """Преобразование корпуса текстов из orm моделей в pandas DF"""
    cols = cols or CORPUS_COLS
    corpus_sel_col = [{"title": news.title, "type": news.type} for news in corpus]
    return pd.DataFrame(corpus_sel_col, columns=cols)


def preprocess_text(text):
    tokens = tokenizer.tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    lemmatized = mystem.lemmatize(' '.join(tokens))
    return [token for token in lemmatized if token.strip()]


def update_word_counts(existing_counts, new_counts):
    for word, count in new_counts.items():
        if word in existing_counts:
            existing_counts[word] += count
        else:
            existing_counts[word] = count
    return existing_counts


def get_most_common_words(corpus: pd.DataFrame):
    all_words = []
    for text in corpus['title']:
        words = preprocess_text(text)
        all_words.extend(words)

    # Считаем частотность слов
    word_counts = Counter(all_words)

    # Чтение существующего CSV файла, если он существует
    try:
        with open('../sentiment-analysis/dictionary.csv', 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_counts = {row['word']: int(row['priority']) for row in reader}
    except FileNotFoundError:
        existing_counts = {}

    # Обновление существующих данных новыми данными
    updated_counts = update_word_counts(existing_counts, word_counts)

    # Сортировка слов по частотности
    sorted_word_counts = sorted(updated_counts.items(), key=lambda x: x[1], reverse=True)

    # Запись обновленного списка в CSV файл
    with open('../service_aggregator/dictionary.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['word', 'priority', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for word, priority in sorted_word_counts:
            if priority > 30:
                writer.writerow({'word': word, 'priority': priority, 'label': ''})


def get_most_common_words_from_json(data):
    all_words = []
    for item in data:
        text = item['text']
        words = preprocess_text(text)
        all_words.extend(words)

    # Считаем частотность слов
    word_counts = Counter(all_words)

    # Чтение существующего CSV файла, если он существует
    try:
        with open('dictionary-json.csv', 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_counts = {row['word']: int(row['priority']) for row in reader}
    except FileNotFoundError:
        existing_counts = {}

    # Обновление существующих данных новыми данными
    updated_counts = update_word_counts(existing_counts, word_counts)

    # Сортировка слов по частотности
    sorted_word_counts = sorted(updated_counts.items(), key=lambda x: x[1], reverse=True)

    # Запись обновленного списка в CSV файл
    with open('dictionary-json.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['word', 'priority', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for word, priority in sorted_word_counts:
            writer.writerow({'word': word, 'priority': priority, 'label': ''})


# Вычисление релевантности терминов (RF)
def compute_rf(tf, categories):
    rf = {category: {} for category in categories}
    for category in categories:
        for word in tf[category]:
            a = tf[category][word]
            b = sum(tf[cat][word] for cat in categories if cat != category)
            rf[category][word] = math.log2(2 + a / max(1, b))
    return rf


def get_ngrams(corpus: pd.DataFrame, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(tokenizer=preprocess_text, ngram_range=ngram_range)
    X = vectorizer.fit_transform(corpus["title"])
    feature_names = vectorizer.get_feature_names_out()
    return X, feature_names, vectorizer


def select_thresholds(rf):
    thresholds = {}
    for category, words in rf.items():
        rf_values = list(words.values())
        plt.hist(rf_values, bins=50, edgecolor='k', alpha=0.7)
        plt.title(f'Distribution of RF Values for {category}')
        plt.xlabel('RF Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

        # Выбор порогового значения визуально (например, через интерпретацию гистограммы)
        threshold = float(input(f"Enter threshold for {category}: "))
        thresholds[category] = threshold

    return thresholds


def save_words_to_csv(rf, thresholds, output_dir='../service_aggregator/'):
    for category, words in rf.items():
        filename = f"{output_dir}dictionary_{category}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['word', 'priority', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for word, rf_value in words.items():
                if rf_value >= thresholds[category]:
                    writer.writerow({'word': word, 'priority': rf_value, 'label': ''})


def run():
    SessionLocal = sessionmaker(bind=engine)
    db: Session = SessionLocal()

    corpus = get_corpus(db)
    corpus_df = prepare_corpus(corpus)

    X, feature_names, vectorizer = get_ngrams(corpus_df, ngram_range=(1, 1))

    # Вычисление частоты терминов в каждой категории
    tf = defaultdict(lambda: defaultdict(int))
    for idx, category in enumerate(corpus_df["type"]):
        tokens = vectorizer.inverse_transform(X[idx])[0]
        for token in tokens:
            tf[category][token] += 1
    print(tf)

    rf = compute_rf(tf, corpus_df["type"])
    thresholds = select_thresholds(rf)
    save_words_to_csv(rf, thresholds)


if __name__ == "__main__":
    run()
