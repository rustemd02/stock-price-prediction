import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from pymystem3 import Mystem
import re
import pickle
import os

# Скачивание необходимых NLTK-ресурсов (если их еще нет)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

class TextPreprocessor:
    """
    Класс для предобработки текстовых данных.

    ПОРЯДОК ИСПОЛЬЗОВАНИЯ:
    1. Создание экземпляра класса: preprocessor = TextPreprocessor()
    2. Обучение токенизатора: preprocessor.fit(тексты)
    3. Преобразование текстов: preprocessor.transform(тексты)
    4. Опционально: сохранение/загрузка preprocessor.save()/TextPreprocessor.load()
    """

    def __init__(self, max_words=10000, max_sequence_length=100):
        """
        Инициализация предобработчика текста.

        Аргументы:
            max_words: Максимальное количество слов в словаре токенизатора.
            max_sequence_length: Максимальная длина последовательности после паддинга.
        """
        # Инициализация инструментов обработки текста
        self.mystem = Mystem()  # Лемматизатор для русского языка
        self.tokenizer_nltk = WordPunctTokenizer()  # Токенизатор разбивающий текст на слова

        # Загрузка русских стоп-слов
        try:
            self.stop_words = set(stopwords.words('russian'))
        except LookupError:
            print("NLTK stopwords for Russian not found. Downloading...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('russian'))

        # Инициализация токенизатора Keras
        # oov_token='' определяет токен для слов вне словаря (out-of-vocabulary)
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='')
        self.max_sequence_length = max_sequence_length

    def clean_text(self, text):
        """
        Очистка и нормализация текста:
        1. Приведение к нижнему регистру
        2. Удаление всех символов кроме букв, цифр и пробелов
        3. Нормализация пробелов
        """
        text = str(text).lower()  # Преобразуем в строку и приводим к нижнему регистру
        text = re.sub(r"[^a-zа-я0-9\s]", " ", text)  # Оставляем только буквы, цифры и пробелы
        text = re.sub(r"\s+", " ", text).strip()  # Заменяем множественные пробелы одиночными
        return text

    def lemmatize(self, text):
        """
        Лемматизация слов в тексте:
        1. Токенизация текста
        2. Фильтрация стоп-слов и коротких токенов
        3. Лемматизация с помощью Mystem
        4. Объединение токенов обратно в текст
        """
        # Токенизация и фильтрация
        tokens = self.tokenizer_nltk.tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2 and token.isalpha()]

        try:
            # Лемматизация с mystem и фильтрация результата
            lemmatized = self.mystem.lemmatize(' '.join(tokens))
            return ' '.join([token for token in lemmatized if
                             token.strip() and token not in self.stop_words and token != ' ' and token != '\n'])
        except Exception as e:
            # В случае ошибки лемматизации возвращаем просто токены
            print(f"Error during lemmatization: {e}. Original text part: {' '.join(tokens)}")
            return ' '.join(tokens)

    def preprocess(self, texts):
        """
        Предобработка списка текстов:
        1. Очистка каждого текста
        2. Лемматизация

        Применяется как при обучении токенизатора, так и при трансформации новых данных.
        """
        processed_texts = []
        for text in texts:
            try:
                cleaned = self.clean_text(text)
                lemmatized = self.lemmatize(cleaned)
                processed_texts.append(lemmatized)
            except Exception as e:
                print(f"Error processing text: {text}. Error: {e}")
                processed_texts.append("")  # Добавляем пустую строку в случае ошибки
        return processed_texts

    def fit(self, texts):
        """
        Обучение токенизатора на текстовых данных:
        1. Предобработка текстов
        2. Создание словаря часто встречающихся слов

        Этот метод должен вызываться ОДИН РАЗ перед transform()
        и только на обучающей выборке данных.
        """
        processed_texts = self.preprocess(texts)
        self.tokenizer.fit_on_texts(processed_texts)

    def transform(self, texts):
        """
        Преобразование текстов в последовательности чисел и применение паддинга:
        1. Предобработка текстов
        2. Преобразование в последовательности индексов (каждый индекс соответствует слову в словаре)
        3. Дополнение последовательностей нулями до одинаковой длины (padding)

        Вызывается ПОСЛЕ fit() как для обучающей, так и для тестовой/валидационной выборки.
        """
        processed_texts = self.preprocess(texts)
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        return padded_sequences

    def save(self, path='model/lstm/text_preprocessor.pkl'):
        """
        Сохранение препроцессора в файл:
        1. Создание директории, если она не существует
        2. Сериализация словаря с параметрами в файл

        Вызывается после обучения токенизатора (fit).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'max_sequence_length': self.max_sequence_length,
                'max_words': self.tokenizer.num_words
            }, f)

    @classmethod
    def load(cls, path='model/lstm/text_preprocessor.pkl'):
        """
        Загрузка препроцессора из файла.

        Вызывается для восстановления уже обученного препроцессора
        при использовании модели для предсказаний.

        Пример использования:
        preprocessor = TextPreprocessor.load('path/to/file.pkl')
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Используем сохраненное значение max_words, если оно есть, иначе берем из токенизатора
        max_words_saved = data.get('max_words', data['tokenizer'].num_words)

        # Создаем новый экземпляр класса
        processor = cls(max_words=max_words_saved,
                        max_sequence_length=data['max_sequence_length'])
        processor.tokenizer = data['tokenizer']  # Заменяем токенизатор на загруженный

        # Проверка корректности num_words в токенизаторе
        if processor.tokenizer.num_words is None or processor.tokenizer.num_words != max_words_saved:
            print(
                f"Warning: Tokenizer num_words ({processor.tokenizer.num_words}) differs from saved max_words ({max_words_saved}). Adjusting.")
            processor.tokenizer.num_words = max_words_saved

        return processor


# ================================================================================================
# ФУНКЦИЯ ДИСКРЕТИЗАЦИИ ТОНАЛЬНОСТИ
# ================================================================================================
def assign_label(score, epsilon=0.05):
    """
    Дискретизация оценки тональности в три класса.

    ПОРЯДОК ВЫЗОВА:
    Эта функция вызывается в prepare_data() для преобразования непрерывных
    оценок тональности в дискретные классы.

    Аргументы:
        score (float): Исходное числовое значение тональности.
        epsilon (float): Пороговое значение для разделения классов.

    Возвращает:
        int: Класс тональности:
            0 - негативная (score < -epsilon)
            1 - нейтральная (-epsilon <= score <= epsilon)
            2 - позитивная (score > epsilon)
    """
    if pd.isna(score):  # Обработка NaN значений
        return 1  # Нейтральная тональность по умолчанию

    try:
        score = float(score)
    except (ValueError, TypeError):
        return 1  # Если не удалось преобразовать в число - считаем нейтральным

    # Дискретизация на основе порога epsilon
    if score > epsilon:
        return 2  # Позитивная тональность (индекс 2)
    elif score < -epsilon:
        return 0  # Негативная тональность (индекс 0)
    else:
        return 1  # Нейтральная тональность (индекс 1)


# ================================================================================================
# ОПРЕДЕЛЕНИЕ СЛОЕВ НЕЙРОННОЙ СЕТИ: МНОГОГОЛОВОЕ ВНИМАНИЕ
# ================================================================================================
class MultiHeadSelfAttention(layers.Layer):
    """
    Многоголовый механизм внимания, как в трансформерах.

    Этот слой является ключевым компонентом Transformer-блока,
    позволяющим модели учиться уделять внимание различным частям входной последовательности.

    ПОРЯДОК ВЫЗОВА:
    1. Создается в составе TransformerBlock
    2. Используется в методе call() блока TransformerBlock
    """

    def __init__(self, embed_dim, num_heads=8, **kwargs):
        """
        Инициализация слоя многоголового внимания.

        Аргументы:
            embed_dim (int): Размерность эмбеддингов
            num_heads (int): Количество голов внимания
            **kwargs: Дополнительные аргументы, передаваемые родительскому классу Layer
        """
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Проверка делимости размерности на число голов
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )

        # Размерность каждой "головы" внимания
        self.projection_dim = embed_dim // num_heads

        # Слои для проекции query, key, value
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)

        # Слой для объединения результатов от всех голов
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        """
        Реализация механизма внимания.

        Аргументы:
            query: тензор запросов [batch_size, num_heads, seq_len, projection_dim]
            key: тензор ключей [batch_size, num_heads, seq_len, projection_dim]
            value: тензор значений [batch_size, num_heads, seq_len, projection_dim]

        Возвращает:
            output: взвешенные значения [batch_size, num_heads, seq_len, projection_dim]
            weights: веса внимания [batch_size, num_heads, seq_len, seq_len]

        Расчет внимания:
        1. Умножение запросов (query) на транспонированные ключи (key)
        2. Масштабирование результата
        3. Применение softmax для получения весов
        4. Умножение весов на значения (value)
        """
        # Матричное произведение запросов и ключей
        score = tf.matmul(query, key, transpose_b=True)

        # Масштабирование для стабильности градиентов
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # Применение softmax для получения весов
        weights = tf.nn.softmax(scaled_score, axis=-1)

        # Применение весов к значениям
        output = tf.matmul(weights, value)

        return output, weights

    def separate_heads(self, x, batch_size):
        """
        Разделение входного тензора на несколько голов внимания.

        Аргументы:
            x: входной тензор [batch_size, seq_len, embed_dim]
            batch_size: размер батча

        Возвращает:
            разделенный тензор [batch_size, num_heads, seq_len, projection_dim]
        """
        # Изменение формы тензора для разделения на головы
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))

        # Перестановка осей для получения формата [batch_size, num_heads, seq_len, projection_dim]
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        """
        Прямой проход через слой многоголового внимания.

        Аргументы:
            inputs: входной тензор [batch_size, seq_len, embed_dim]

        Возвращает:
            output: выходной тензор [batch_size, seq_len, embed_dim]

        Последовательность операций:
        1. Проекция входных данных в пространства query, key, value
        2. Разделение на головы внимания
        3. Вычисление внимания
        4. Объединение результатов от всех голов
        """
        batch_size = tf.shape(inputs)[0]

        # Проекция входных данных в пространства query, key, value
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Разделение на головы внимания
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        # Вычисление внимания
        attention, weights = self.attention(query, key, value)

        # Перестановка осей для объединения результатов
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        # Изменение формы тензора для объединения всех голов
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))

        # Применение финальной проекции
        output = self.combine_heads(concat_attention)

        return output

    def get_config(self):
        """
        Получение конфигурации слоя для сериализации.

        Необходим для сохранения и загрузки модели.

        Возвращает:
            dictionary: словарь с параметрами слоя
        """
        config = super(MultiHeadSelfAttention, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Создание экземпляра слоя из конфигурации.

        Используется при загрузке модели.
        """
        return cls(**config)


# ================================================================================================
# ОПРЕДЕЛЕНИЕ СЛОЕВ НЕЙРОННОЙ СЕТИ: TRANSFORMER БЛОК
# ================================================================================================
class TransformerBlock(layers.Layer):
    """
    Трансформерный блок, включающий механизм внимания и feed-forward сеть.

    Структура блока:
    1. Многоголовое внимание
    2. Резидуальное соединение и нормализация слоя
    3. Полносвязная feed-forward сеть
    4. Еще одно резидуальное соединение и нормализация слоя

    ПОРЯДОК ВЫЗОВА:
    1. Создается в функции build_lstm_transformer_model
    2. Используется в последовательности слоев модели
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        """
        Инициализация трансформерного блока.

        Аргументы:
            embed_dim (int): Размерность эмбеддингов
            num_heads (int): Количество голов в механизме внимания
            ff_dim (int): Размерность скрытого слоя feed-forward сети
            rate (float): Коэффициент dropout для регуляризации
            **kwargs: Дополнительные аргументы для родительского класса Layer
        """
        super(TransformerBlock, self).__init__(**kwargs)
        # Сохраняем параметры для возможности сериализации
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        # Слой многоголового внимания
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)

        # Feed-forward сеть (два полносвязных слоя)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ], name="ffn")

        # Слои нормализации для стабилизации обучения
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Dropout для предотвращения переобучения
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        """
        Прямой проход через трансформерный блок.

        Аргументы:
            inputs: входной тензор [batch_size, seq_len, embed_dim]
            training: флаг режима обучения (для dropout)

        Возвращает:
            output: выходной тензор [batch_size, seq_len, embed_dim]

        Последовательность операций:
        1. Применение механизма внимания
        2. Dropout и резидуальное соединение с нормализацией
        3. Применение feed-forward сети
        4. Еще один dropout и резидуальное соединение с нормализацией
        """
        # Многоголовое внимание
        attn_output = self.attention(inputs)

        # Dropout и первое резидуальное соединение с нормализацией
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward сеть
        ffn_output = self.ffn(out1)

        # Dropout и второе резидуальное соединение с нормализацией
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        """
        Получение конфигурации блока для сериализации.
        """
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Создание экземпляра блока из конфигурации.
        """
        return cls(**config)


# ================================================================================================
# ПОСТРОЕНИЕ МОДЕЛИ LSTM С ТРАНСФОРМЕР-БЛОКАМИ
# ================================================================================================
def build_lstm_transformer_model(max_words, sequence_length, embedding_dim=128, lstm_units=64,
                                 num_heads=2, ff_dim=32, num_transformer_blocks=1, num_classes=3,
                                 dropout_rate=0.1):
    """
    Построение гибридной модели LSTM с трансформерными блоками.

    ПОРЯДОК ВЫЗОВА:
    Функция вызывается из train_lstm_transformer() после подготовки данных
    и перед вызовом model.fit()

    Архитектура модели:
    1. Input слой - входная последовательность индексов слов
    2. Embedding слой - преобразование индексов в векторы фиксированной размерности
    3. Один или несколько TransformerBlock - для контекстного понимания
    4. Bidirectional LSTM - для обработки последовательности в обоих направлениях
    5. GlobalMaxPooling1D - извлечение наиболее важных признаков
    6. Dense - полносвязный слой с активацией ReLU
    7. Dropout - для предотвращения переобучения
    8. Выходной Dense слой с Softmax - для классификации

    Аргументы:
        max_words (int): Размер словаря (количество уникальных слов + 1)
        sequence_length (int): Длина входной последовательности
        embedding_dim (int): Размерность эмбеддингов
        lstm_units (int): Количество LSTM ячеек
        num_heads (int): Количество голов в механизме внимания
        ff_dim (int): Размерность скрытого слоя feed-forward сети в TransformerBlock
        num_transformer_blocks (int): Количество трансформерных блоков
        num_classes (int): Количество классов (3 для негативного, нейтрального, позитивного)
        dropout_rate (float): Коэффициент dropout для регуляризации

    Возвращает:
        model: Скомпилированная Keras-модель
    """
    # Создание входного слоя
    inputs = layers.Input(shape=(sequence_length,), name="input_layer")

    # Проверка корректности max_words
    if max_words <= 0:
        raise ValueError(f"max_words must be positive, but got {max_words}")

    # Слой эмбеддингов: преобразует индексы слов в векторы
    x = layers.Embedding(input_dim=max_words, output_dim=embedding_dim)(inputs)

    # Применение трансформерных блоков
    for i in range(num_transformer_blocks):
        x = TransformerBlock(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            rate=dropout_rate,
            name=f'transformer_block_{i}'
        )(x)

    # Двунаправленный LSTM слой
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)

    # Извлечение наиболее важных признаков
    x = layers.GlobalMaxPooling1D()(x)

    # Полносвязный слой для обработки признаков
    x = layers.Dense(ff_dim, activation="relu")(x)

    # Dropout для предотвращения переобучения
    x = layers.Dropout(dropout_rate)(x)

    # Выходной слой с softmax-активацией для классификации
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Создание модели
    model = models.Model(inputs, outputs)

    # Компиляция модели
    model.compile(
        optimizer=optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ================================================================================================
# ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ
# ================================================================================================
def load_data(filepath):
    """
    Загрузка данных из CSV файла с корректной обработкой заголовка.

    ПОРЯДОК ВЫЗОВА:
    Вызывается в начале функции train_lstm_transformer как первый шаг.

    Аргументы:
        filepath (str): Путь к CSV файлу с данными

    Возвращает:
        df (pandas.DataFrame): DataFrame с загруженными данными

    Формат загружаемого CSV:
    title,type,post_dttm,-1,0,1
    "Текст заголовка",тип_новости,дата,вероятность_негативного,вероятность_нейтрального,вероятность_позитивного
    """
    try:
        # Читаем CSV, пропуская первую строку (заголовок) и явно задаем имена столбцов
        df = pd.read_csv(filepath, skiprows=1, names=['title', 'type', 'post_dttm', '-1', '0', '1'])
        print(f"Загружен CSV с размером: {df.shape}")
        print("Первые несколько строк:")
        print(df.head(3))

        # Преобразуем столбцы с вероятностями в числовой формат
        for col in ['-1', '0', '1']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Создаем метку класса на основе максимальной вероятности
        # Для каждой строки находим столбец с максимальным значением и преобразуем его в int
        df['label'] = df[['-1', '0', '1']].idxmax(axis=1).astype(int)
        print("\nСоздан столбец 'label' на основе максимальной вероятности")
        print("Распределение меток класса:", df['label'].value_counts())

        return df
    except Exception as e:
        print(f"Ошибка при загрузке данных из {filepath}: {e}")
        raise


# ================================================================================================
# ФУНКЦИЯ ПОДГОТОВКИ ДАННЫХ
# ================================================================================================
def prepare_data(df, epsilon=0.05, test_size=0.2, random_state=42):
    """
    Подготовка данных для обучения модели:
    1. Анализ распределения меток
    2. Преобразование меток в формат, подходящий для обучения
    3. Разделение на обучающую и тестовую выборки

    ПОРЯДОК ВЫЗОВА:
    Вызывается после load_data в функции train_lstm_transformer.

    Аргументы:
        df (pandas.DataFrame): DataFrame с данными
        epsilon (float): Порог для функции assign_label
        test_size (float): Доля данных для тестовой выборки
        random_state (int): Начальное значение для генератора случайных чисел

    Возвращает:
        X_train, X_test, y_train, y_test: Наборы данных для обучения и тестирования
    """
    # Анализ исходных значений в колонке 'label'
    print("\nИсходная статистика по колонке 'label':")
    print(df['label'].describe())
    print(f"Количество NaN значений: {df['label'].isna().sum()}")

    # Преобразуем метки -1, 0, 1 в индексы классов 0, 1, 2 для categorical_crossentropy
    label_mapping = {-1: 0, 0: 1, 1: 2}
    df['sentiment_mapped'] = df['label'].map(label_mapping)

    # Анализ распределения классов
    print("\nРаспределение классов после маппинга:")
    class_counts = df['sentiment_mapped'].value_counts().sort_index()
    print(class_counts)

    print(f"\nПропорции классов:")
    print(class_counts / len(df))

    # Удаление строк с отсутствующими метками
    df = df.dropna(subset=['sentiment_mapped'])

    # Проверка на наличие достаточного количества данных для стратификации
    min_class_count = df['sentiment_mapped'].value_counts().min()
    n_splits_required = int(1 / test_size) if test_size > 0 else 1

    # Выбор способа стратификации в зависимости от количества примеров в наименьшем классе
    if min_class_count < n_splits_required and test_size > 0:
        print(
            f"Предупреждение: Наименее представленный класс имеет только {min_class_count} примеров, "
            f"что недостаточно для стратифицированного разделения с test_size={test_size}.")
        print("Стратификация будет отключена.")
        stratify_param = None
    else:
        stratify_param = df['sentiment_mapped']

    # Разделение на обучающую и тестовую выборки
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            df['title'],
            df['sentiment_mapped'],
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )

        # Проверка распределения классов в обучающей и тестовой выборках
        print("\nРаспределение классов в обучающей выборке:")
        print(pd.Series(y_train).value_counts().sort_index())

        print("\nРаспределение классов в тестовой выборке:")
        print(pd.Series(y_test).value_counts().sort_index())

        print(f"Данные разделены на обучающую ({len(X_train)}) и тестовую ({len(X_test)}) выборки.")
    else:
        # Если test_size=0, используем все данные для обучения
        X_train, y_train = df['title'], df['sentiment_mapped']
        X_test, y_test = pd.Series(dtype='object'), pd.Series(dtype='int')
        print("Используются все данные для обучения, тестовая выборка отсутствует.")

    return X_train, X_test, y_train, y_test


# ================================================================================================
# ФУНКЦИЯ ОБУЧЕНИЯ МОДЕЛИ
# ================================================================================================
def train_lstm_transformer(data_filepath, model_save_path='model/lstm/lstm_transformer_model.keras',
                           preprocessor_save_path='model/lstm/text_preprocessor.pkl',
                           max_words=10000, max_sequence_length=100, embedding_dim=128,
                           lstm_units=64, num_heads=2, ff_dim=32, num_transformer_blocks=1,
                           num_classes=3, dropout_rate=0.1, epsilon=0.05, epochs=15, batch_size=32):
    """
    Обучение модели LSTM с трансформером.

    ПОРЯДОК ВЫПОЛНЕНИЯ:
    1. Загрузка данных (load_data)
    2. Подготовка данных (prepare_data)
    3. Создание и обучение препроцессора (TextPreprocessor)
    4. Преобразование текстов в последовательности
    5. Создание модели (build_lstm_transformer_model)
    6. Обучение модели (model.fit)
    7. Оценка модели на тестовой выборке
    8. Сохранение модели и препроцессора

    Аргументы:
        data_filepath (str): Путь к файлу с данными
        model_save_path (str): Путь для сохранения обученной модели
        preprocessor_save_path (str): Путь для сохранения препроцессора
        max_words (int): Максимальное число слов в словаре
        max_sequence_length (int): Максимальная длина последовательности
        embedding_dim (int): Размерность эмбеддингов
        lstm_units (int): Количество LSTM ячеек
        num_heads (int): Количество голов в механизме внимания
        ff_dim (int): Размерность скрытого слоя feed-forward сети
        num_transformer_blocks (int): Количество трансформерных блоков
        num_classes (int): Количество классов
        dropout_rate (float): Коэффициент dropout
        epsilon (float): Порог для дискретизации тональности
        epochs (int): Количество эпох обучения
        batch_size (int): Размер батча

    Возвращает:
        model: Обученная модель
        preprocessor: Обученный препроцессор
        history: История обучения
    """
    # 1. Загрузка данных
    df = load_data(data_filepath)
    if df.empty:
        print("Загруженный DataFrame пуст. Останавливаем обучение.")
        return None, None, None

    # 2. Подготовка данных
    X_train, X_test, y_train, y_test = prepare_data(df, epsilon=epsilon, test_size=0.2 if len(df) > 10 else 0)

    # 3. Создание и обучение препроцессора
    preprocessor = TextPreprocessor(max_words=max_words, max_sequence_length=max_sequence_length)
    preprocessor.fit(X_train)

    # Определение размера словаря (количества уникальных слов)
    actual_max_words = len(preprocessor.tokenizer.word_index) + 1  # +1 для OOV токена
    print(f"Actual vocabulary size (including OOV): {actual_max_words}")

    # Выбор эффективного размера словаря (не больше, чем количество уникальных слов)
    effective_max_words = min(max_words, actual_max_words) if max_words > 0 else actual_max_words

    # 4. Преобразование текстов в последовательности
    X_train_seq = preprocessor.transform(X_train)
    if not X_test.empty:
        X_test_seq = preprocessor.transform(X_test)
    else:
        X_test_seq = np.array([])

    # Преобразование меток в формат one-hot (для категориальной кросс-энтропии)
    y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    if not y_test.empty:
        y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
    else:
        y_test_categorical = np.array([])

    # 5. Создание модели
    model = build_lstm_transformer_model(
        max_words=effective_max_words,
        sequence_length=max_sequence_length,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )

    # Вывод структуры модели
    model.summary()

    # Расчет весов классов для балансировки (учёт несбалансированности классов)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print("\nВеса классов для обучения:")
    print(class_weight_dict)

    # Определяем validation_data только если есть тестовые данные
    validation_data_param = (X_test_seq, y_test_categorical) if not y_test.empty else None

    # 6. Обучение модели с весами классов
    history = model.fit(
        X_train_seq, y_train_categorical,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data_param,
        class_weight=class_weight_dict  # Балансировка классов
    )

    # 7. Оценка модели и отчет только если есть тестовые данные
    if not y_test.empty:
        loss, accuracy = model.evaluate(X_test_seq, y_test_categorical)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

        # Получение предсказаний модели
        y_pred_prob = model.predict(X_test_seq)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Вывод отчета о классификации
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'],
                                    labels=[0, 1, 2], zero_division=0))
    else:
        print("No test data to evaluate.")

    # 8. Сохранение модели и препроцессора
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    preprocessor.save(preprocessor_save_path)

    print(f"Модель сохранена в {model_save_path}")
    print(f"Препроцессор сохранен в {preprocessor_save_path}")

    return model, preprocessor, history


# ================================================================================================
# ФУНКЦИЯ ПРЕДСКАЗАНИЯ СЕНТИМЕНТОВ
# ================================================================================================
def predict_sentiment(texts, model_path='model/lstm/lstm_transformer_model.keras',
                      preprocessor_path='model/lstm/text_preprocessor.pkl'):
    """
    Предсказание тональности текстов с помощью обученной модели.

    ПОРЯДОК ВЫПОЛНЕНИЯ:
    1. Загрузка препроцессора и модели
    2. Преобразование текстов в последовательности с помощью препроцессора
    3. Получение предсказаний от модели
    4. Преобразование индексов классов обратно в метки тональности
    5. Формирование результатов

    Аргументы:
        texts (list): Список текстов для анализа
        model_path (str): Путь к сохраненной модели
        preprocessor_path (str): Путь к сохраненному препроцессору

    Возвращает:
        pandas.DataFrame: Таблица с результатами предсказаний
    """
    # Проверка существования файлов
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    if not os.path.exists(preprocessor_path):
        print(f"Error: Preprocessor file not found at {preprocessor_path}")
        return None

    try:
        # 1. Загрузка препроцессора
        preprocessor = TextPreprocessor.load(preprocessor_path)

        # Загрузка модели с указанием пользовательских слоев
        custom_objects = {
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
            'TransformerBlock': TransformerBlock
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    except Exception as e:
        print(f"Error loading model or preprocessor: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 2. Преобразование текстов в последовательности
    sequences = preprocessor.transform(texts)

    # 3. Получение предсказаний
    predictions_prob = model.predict(sequences)

    # 4. Преобразование вероятностей в классы (0, 1, 2)
    predictions = np.argmax(predictions_prob, axis=1)

    # Обратное преобразование классов в исходные метки (-1, 0, 1)
    label_mapping_reverse = {0: -1, 1: 0, 2: 1}
    predictions_original = [label_mapping_reverse.get(pred, 0) for pred in predictions]

    # 5. Формирование результатов
    results = pd.DataFrame({
        'text': texts,
        'sentiment': predictions_original,
        'negative_prob': predictions_prob[:, 0],
        'neutral_prob': predictions_prob[:, 1],
        'positive_prob': predictions_prob[:, 2]
    })

    return results


# ================================================================================================
# ОСНОВНАЯ ТОЧКА ВХОДА
# ================================================================================================
if __name__ == "__main__":
    """
    Основная точка входа при запуске файла как скрипта.

    ПОСЛЕДОВАТЕЛЬНОСТЬ ВЫПОЛНЕНИЯ:
    1. Проверка наличия файла с данными
    2. Вызов функции train_lstm_transformer для обучения модели
    3. Проверка обучения модели
    4. Тестирование модели на примерах текстов
    """
    # Путь к файлу с данными
    data_filepath = "news_with_probabilities.csv"

    # Проверка существования файла
    if os.path.exists(data_filepath):
        print(f"Начинаем обучение с данными из: {data_filepath}")

        # Обучение модели
        model, preprocessor, history = train_lstm_transformer(
            data_filepath=data_filepath,
            epochs=15,  # Количество циклов обучения
            batch_size=32,  # Размер батча
            max_words=10000,  # Максимальный размер словаря
            max_sequence_length=100,  # Максимальная длина последовательности
            epsilon=0.05,  # Порог для разделения классов
            dropout_rate=0.2  # Коэффициент дропаута для регуляризации
        )

        # Проверка успешности обучения и тестирование на примерах
        if model and preprocessor:
            # Тестовые тексты для проверки модели
            sample_texts = [
                "Мишустин заявил, что Россия смогла адаптироваться к санкциям",
                "Reuters узнал об остановке более 60 нефтяных танкеров после санкций США",
                "Акционеры одобрили рекордные дивиденды",
                "Рынок акций упал на фоне негативных новостей"
            ]

            print("\nPredicting sentiment for sample texts...")
            # Получение предсказаний для тестовых текстов
            results = predict_sentiment(sample_texts)
            if results is not None:
                print("\nПример предсказаний:")
                print(results)

