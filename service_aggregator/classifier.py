import csv
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
import joblib
import re

# Загрузка данных
df = pd.read_csv("../last-work/sentiment-analysis/news_sentimentv5.csv")

# Лемматизация и токенизация текста
lemmatizer = WordNetLemmatizer()
tokenizer = WordPunctTokenizer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    words = tokenizer.tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words]
    return [word for word in words if word.isalpha() and len(word) > 1]

def filter_tokens(words, token_counts, total_tokens):
    filtered_words = []
    for word in words:
        if word.isalpha() and len(word) > 1:
            if 3 <= token_counts[word] <= total_tokens * 0.95:
                filtered_words.append(word)
    return filtered_words


df['tokens'] = df['title'].apply(preprocess_text)


# Преобразование токенов в индексы и паддинг
all_tokens = [token for tokens in df['tokens'] for token in tokens]
token_counts = Counter(all_tokens)
total_tokens = len(token_counts)

df['filtered_tokens'] = df['tokens'].apply(lambda words: filter_tokens(words, token_counts, total_tokens))

vocab = {token: idx for idx, (token, _) in enumerate(token_counts.items())}
df['indices'] = df['filtered_tokens'].apply(lambda x: [vocab[token] for token in x])
max_len = df['indices'].str.len().max()
df['padded_indices'] = df['indices'].apply(lambda x: x + [0] * (max_len - len(x)) if len(x) < max_len else x[:max_len])

# Преобразование в массив NumPy
X = np.array(df['padded_indices'].tolist())
y = np.array(df['label'].tolist())

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели логистической регрессии
model = LogisticRegression(max_iter=2000, multi_class='ovr').fit(X_train, y_train)

# Оценка модели
for name, X, y, model in [
    ('train', X_train, y_train, model),
    ('test ', X_test, y_test, model)
]:
    proba = model.predict(X)
    print(y)
    print(proba)
    accuracy = accuracy_score(y, proba)
    print(f'{name} AUC: {accuracy:.4f}')

joblib.dump(model, '../application/logistic_regression_model.pkl')

# Сохранение словаря токенов
with open('../application/vocab.pkl', 'wb') as f:
    joblib.dump(vocab, f)
