from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sqlalchemy.orm import Session, sessionmaker
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from pymystem3 import Mystem

from service_aggregator.dictionary_words import CORPUS_COLS
from service_repository.crud import get_corpus_by_date_range
from service_repository.models import NewsModel
from service_repository.database import engine
from joblib import load, dump




# Функция для агрегации данных и расчета средних значений тональностей
# Цель: агрегировать новости по дате и типу новостей (например, economic или political),
# чтобы получить средние тональности на конкретную дату.
# Пример функции, которая суммирует вероятности по (date, type), а потом нормирует их внутри дня
def aggregate_news(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    На входе:
      news_df должен содержать столбцы:
        - date (формат datetime.date или datetime64[ns])
        - type (например, 'economic'/'political')
        - -1, 0, 1 (вероятности для каждого класса)
    На выходе:
      DataFrame со строками = (date) и столбцами вида:
        neg_mean_economic, neutral_mean_economic, pos_mean_economic,
        neg_mean_political, ...
      и т.д. для всех типов, которые присутствуют в данных.
    """
    # Суммируем вероятности по группам (date, type)
    grouped = news_df.groupby(['date', 'type'])[['-1','0','1']].sum().reset_index()

    # Считаем сумму вероятностей за день (для конкретного типа)
    # total_prob = sum_neg + sum_neu + sum_pos
    grouped['total_prob'] = grouped['-1'] + grouped['0'] + grouped['1']

    # Считаем долю (например, neg_mean = sum_neg / total_prob)
    grouped['neg_mean'] = grouped['-1'] / grouped['total_prob']
    grouped['neu_mean'] = grouped['0'] / grouped['total_prob']
    grouped['pos_mean'] = grouped['1'] / grouped['total_prob']

    # Удаляем сырые колонки
    grouped.drop(columns=['-1','0','1','total_prob'], inplace=True)

    # Теперь нам нужно распивотить по type, чтобы были разные колонки
    # для экономических и политических новостей
    # Индекс = date, столбцы = (neg_mean, neu_mean, pos_mean) для каждого типа
    pivoted = grouped.pivot(index='date', columns='type', values=['neg_mean','neu_mean','pos_mean'])

    # У pivoted будет MultiIndex колонок вида ('neg_mean', 'economic') и т.д.
    # Преобразуем в плоские названия столбцов:
    #  neg_mean_economic, neu_mean_economic, pos_mean_economic, ...
    pivoted.columns = [f"{col[0]}_{col[1]}" for col in pivoted.columns]

    # Сбросим индекс date в обычный столбец
    pivoted = pivoted.reset_index()

    return pivoted

def time_based_train_test_split(df: pd.DataFrame, split_date: str):
    """
    Разделяем данные по дате: всё, что строго < split_date пойдёт в train,
    остальное — в test.
    Аргумент split_date должен быть строкой, которую можно
    интерпретировать в pd.to_datetime().
    """
    cutoff = pd.to_datetime(split_date)
    train = df[df['date'] < cutoff].copy()
    test  = df[df['date'] >= cutoff].copy()
    return train, test



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

def get_probabilities(title: str):
    """
    Возвращает словарь с вероятностями по классам -1, 0, 1
    Например: {'-1': 0.1, '0': 0.5, '1': 0.4}
    """
    # подгружаем обученный vectorizer и модель
    vectorizer = load('model/vectorizer.joblib')
    clf = load('model/best_classifier.joblib')

    text_clean = preprocess_text(title)
    X_new = vectorizer.transform([text_clean])
    y_prob = clf.predict_proba(X_new)[0]  # np.array из 3 вероятностей

    # Предположим, clf.classes_ = [-1, 0, 1]
    # тогда:
    class_labels = clf.classes_  # например, array([-1,  0,  1])
    prob_dict = {str(lbl): p for lbl, p in zip(class_labels, y_prob)}

    return prob_dict

def train():
    # 1) Читаем подготовленный файл с вероятностями
    news = pd.read_csv('news_with_probabilities.csv')
    news['date'] = pd.to_datetime(news['post_dttm'])

    # 2) Агрегируем данные за каждый день и для каждого типа
    news_aggregated = aggregate_news(news)
    # Теперь у нас колонки вида:
    #   date, neg_mean_economic, neu_mean_economic, pos_mean_economic,
    #        neg_mean_political, neu_mean_political, pos_mean_political, ...

    # 3) Читаем котировки индекса:
    index_data = pd.read_csv(
        '../static/IMOEX.csv',
        sep=';',  # или ваш разделитель
        names=['TICKER','PER','DATE','TIME','OPEN','HIGH','LOW','CLOSE','VOL'],
        skiprows=1
    )
    index_data['DATE'] = pd.to_datetime(index_data['DATE'], format='%Y%m%d')
    # Добавим доходность и бинарный target
    index_data['returns'] = index_data['CLOSE'].pct_change()
    index_data['target'] = (index_data['returns'] > 0).astype(int)

    # 4) Объединяем с агрегированными новостями
    merged_data = pd.merge(
        news_aggregated,            # содержит date
        index_data[['DATE','target']],  # содержит DATE, target
        left_on='date',
        right_on='DATE',
        how='inner'
    )
    merged_data.drop(columns=['DATE'], inplace=True)

    # Проверим, сколько строк осталось
    print("merged_data.shape =", merged_data.shape)
    print(merged_data.head())

    # Предположим, в merged_data появились следующие признаки:
    # neg_mean_economic, neu_mean_economic, pos_mean_economic,
    # neg_mean_political, neu_mean_political, pos_mean_political, ...
    # А также 'target'.

    # 5) Делим по времени
    train_data, test_data = time_based_train_test_split(merged_data, split_date='2025-01-20')

    # 6) Формируем X, y
    feature_cols = [col for col in merged_data.columns if col.startswith(('neg_mean','neu_mean','pos_mean'))]
    X_train = train_data[feature_cols]
    y_train = train_data['target']
    X_test  = test_data[feature_cols]
    y_test  = test_data['target']

    print("Train set size =", X_train.shape, ", Test set size =", X_test.shape)

    # 7) Обучаем RandomForest
    rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf_clf.fit(X_train, y_train)

    # 8) Предсказываем на тесте
    y_pred = rf_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("RandomForest Accuracy =", acc)
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 9) Сохраняем модель
    dump(rf_clf, 'model/random_forest.joblib')



def run():
    SessionLocal = sessionmaker(bind=engine)
    db: Session = SessionLocal()

    corpus = get_corpus_by_date_range(db, '2025-01-11 00:00:00.000000', '2025-01-28 00:00:00.000000')
    news = prepare_corpus(corpus)
    print(news)
    probabilities_list = []
    count = 0
    for title in news['title']:
        print(count)
        probabilities = get_probabilities(title)
        probabilities_list.append(probabilities)
        count +=1

    # Преобразование вероятностей в DataFrame
    probabilities_df = pd.DataFrame(probabilities_list)
    news = pd.concat([news, probabilities_df], axis=1)
    news.to_csv('news_with_probabilities.csv', index=False)

if __name__ == "__main__":
    # run()
     train()
