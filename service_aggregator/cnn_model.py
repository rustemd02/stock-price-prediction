import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# ---------------------------
# 1. Загрузка и подготовка данных
# ---------------------------
def load_data(filepath):
    """
    Функция load_data загружает данные из CSV-файла.

    Ожидается, что файл (например, newsv6.csv) не содержит заголовков,
    а данные расположены в следующем порядке:
        1. uuid
        2. title (заголовок новости)
        3. post_dttm (дата и время публикации)
        4. url (ссылка на источник)
        5. processed_dttm (дата обработки новости)
        6. type (тип новости, например, economic или political)
        7. label (исходная непрерывная оценка тональности)

    После загрузки данные приводятся к нужному типу, а столбец label конвертируется
    в числовой формат.

    Аргументы:
      filepath (str): путь к CSV файлу.

    Возвращает:
      df (pandas.DataFrame): DataFrame с заданными столбцами.
    """
    # Читаем CSV без заголовков
    df = pd.read_csv(filepath, header=None)
    # Задаём имена столбцов согласно ожидаемому порядку
    df.columns = ['uuid', 'title', 'post_dttm', 'url', 'processed_dttm', 'type', 'label']

    # Преобразуем значения столбца 'label' к числовому типу (если встречаются ошибки, заменяет на NaN)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')

    return df


# ---------------------------
# 2. Дискретизация тональности
# ---------------------------
def assign_label(score, epsilon=0.0):
    """
    Функция assign_label принимает непрерывное числовое значение тональности (score)
    и преобразует его в дискретный класс:
        1     → если score больше порога epsilon (позитивная тональность)
       -1     → если score меньше отрицательного порога (-epsilon) (негативная тональность)
        0     → если score находится между -epsilon и epsilon (нейтральная тональность)

    Аргументы:
      score (float): исходное числовое значение тональности.
      epsilon (float): пороговое значение для разделения классов.

    Возвращает:
      int: дискретная метка тональности (-1, 0 или 1).
    """
    if score > epsilon:
        return 1
    elif score < -epsilon:
        return -1
    else:
        return 0


# ---------------------------
# 3. Предобработка текста
# ---------------------------
def preprocess_text(text):
    """
    Функция preprocess_text выполняет базовую очистку текста:
      - Приводит текст к нижнему регистру.
      - Удаляет все символы, не являющиеся буквами (латинскими или русскими), цифрами или пробелами.
      - Заменяет множественные пробелы на один и убирает лишние пробелы в начале и конце строки.

    Аргументы:
      text (str): исходный текст новости.

    Возвращает:
      str: очищенный текст.
    """
    # Приводим текст к нижнему регистру
    text = text.lower()
    # Заменяем все символы, не входящие в разрешённый набор (латинские и русские буквы, цифры, пробелы), на пробел
    text = re.sub(r"[^a-zа-я0-9\s]", " ", text)
    # Убираем лишние пробелы
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------
# 4. Подготовка данных для CNN
# ---------------------------
def prepare_cnn_data(df, epsilon=0.0, max_num_words=10000, max_sequence_length=100):
    """
    Функция prepare_cnn_data выполняет комплексную подготовку данных для обучения сверточной нейронной сети.
    Этапы обработки:
      1. Применение дискретизации тональности с помощью assign_label. Здесь исходный
         столбец label преобразуется в три класса: -1, 0 или 1 (с учетом epsilon).
      2. Очистка текста: к заголовкам применяется функция preprocess_text.
      3. Токенизация: используется Keras Tokenizer для преобразования текста в последовательности чисел.
      4. Padding: последовательности дополняются до фиксированной длины (max_sequence_length).
      5. Преобразование меток в one-hot представление с помощью to_categorical.

    Аргументы:
      df (pandas.DataFrame): исходный DataFrame с данными.
      epsilon (float): порог для дискретизации тональности.
      max_num_words (int): максимальное число слов для токенизатора.
      max_sequence_length (int): максимальная длина текстовой последовательности после padding.

    Возвращает:
      data (numpy.array): матрица входных последовательностей (после padding).
      labels_categorical (numpy.array): матрица меток в формате one-hot.
      tokenizer (Tokenizer): объект токенизатора (для последующего использования при инференсе).
      num_classes (int): количество уникальных классов тональности.
    """
    # Применяем дискретизацию тональности к исходному значению label
    df['sentiment'] = df['label'].apply(lambda x: assign_label(x, epsilon))

    # Выполняем предобработку текста для столбца 'title'
    df['cleaned_text'] = df['title'].apply(preprocess_text)

    # Получаем список очищенных текстов и соответствующих дискретных меток
    texts = df['cleaned_text'].tolist()
    raw_labels = df['sentiment'].tolist()

    # Задаём маппинг меток: например, чтобы преобразовать -1 -> 0, 0 -> 1, 1 -> 2
    label_mapping = {-1: 0, 0: 1, 1: 2}
    labels = [label_mapping[l] for l in raw_labels]

    # Определяем количество классов (ожидается 3 класса)
    num_classes = len(set(labels))
    # Преобразуем метки в формат one-hot (например, [1,0,0] для первого класса)
    labels_categorical = to_categorical(labels, num_classes=num_classes)

    # ---------------------------
    # Токенизация текста
    # ---------------------------
    # Создаем токенайзер, который будет учитывать только max_num_words наиболее часто встречающихся слов
    tokenizer = Tokenizer(num_words=max_num_words)
    # Обучаем токенайзер на списке текстов
    tokenizer.fit_on_texts(texts)
    # Преобразуем тексты в последовательности числовых индексов (каждое число — индекс слова)
    sequences = tokenizer.texts_to_sequences(texts)

    # Применяем padding к последовательностям: все последовательности будут иметь длину max_sequence_length.
    # Если последовательность короче, она дополняется нулями; если длиннее, усечется.
    data = pad_sequences(sequences, maxlen=max_sequence_length)

    return data, labels_categorical, tokenizer, num_classes


# ---------------------------
# 5. Построение модели CNN
# ---------------------------
def build_cnn_model(max_num_words, max_sequence_length, embedding_dim, num_classes):
    """
    Функция build_cnn_model строит сверточную нейронную сеть (CNN) для текстовой классификации.

    Структура модели:
      1. Слой Embedding – преобразует последовательность индексов слов в эмбеддинги фиксированной размерности.
      2. Сверточный слой (Conv1D) – извлекает локальные признаки (n-граммы) из эмбеддингов.
      3. GlobalMaxPooling1D – выбирает наиболее значимые признаки по всей последовательности.
      4. Dropout – снижает переобучение, случайным образом обнуляя часть входных сигналов.
      5. Полносвязный Dense слой – для дальнейшей нелинейной обработки признаков.
      6. Выходной Dense слой с softmax – для многоклассовой классификации.

    Аргументы:
      max_num_words (int): размер словаря (для Embedding слоя).
      max_sequence_length (int): длина входной последовательности.
      embedding_dim (int): размерность эмбеддингов.
      num_classes (int): количество классов (например, 3 для негативного, нейтрального и позитивного).

    Возвращает:
      model (tf.keras.Model): собранная и скомпилированная модель CNN.
    """
    model = Sequential()
    # Слой Embedding: преобразует входные индексы слов (целые числа) в плотные векторы фиксированной размерности
    model.add(Embedding(input_dim=max_num_words,
                        output_dim=embedding_dim,
                        input_length=max_sequence_length))
    # Сверточный слой Conv1D: применяет 128 фильтров размером kernel_size=5, функция активации relu
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    # GlobalMaxPooling1D: объединяет выходы по временной оси, оставляя только наиболее важные признаки
    model.add(GlobalMaxPooling1D())
    # Dropout: отключает случайно 50% нейронов для уменьшения переобучения
    model.add(Dropout(0.5))
    # Полносвязный (Dense) слой с 64 нейронами и активацией relu
    model.add(Dense(64, activation='relu'))
    # Выходной слой: количество нейронов соответствует числу классов, активация softmax для многоклассовой классификации
    model.add(Dense(num_classes, activation='softmax'))

    # Компиляция модели: оптимизатор adam, loss – categorical_crossentropy (подходит для one-hot меток), метрика – accuracy
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# ---------------------------
# 6. Обучение и оценка модели CNN
# ---------------------------
def train_and_evaluate_cnn(data, labels_categorical, tokenizer, num_classes, epochs=10, batch_size=32, test_size=0.2):
    """
    Функция train_and_evaluate_cnn делит данные на обучающую и тестовую выборки,
    строит, обучает и оценивает модель CNN.

    Этапы:
      1. Разбивка данных на train и test, с сохранением пропорции классов.
      2. Построение модели CNN с заданными гиперпараметрами.
      3. Обучение модели с выводом истории обучения.
      4. Оценка модели на тестовой выборке: вычисление accuracy и вывод classification_report.
      5. Сохранение обученной модели (в формате .h5) и объекта tokenizer (в pickle).

    Аргументы:
      data (numpy.array): входные последовательности (после токенизации и padding).
      labels_categorical (numpy.array): метки в one-hot представлении.
      tokenizer (Tokenizer): объект токенизатора.
      num_classes (int): количество классов.
      epochs (int): число эпох обучения.
      batch_size (int): размер батча.
      test_size (float): доля данных для тестовой выборки.

    Возвращает:
      model: обученная модель CNN.
      history: история обучения модели.
    """
    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels_categorical, test_size=test_size, random_state=42, stratify=labels_categorical
    )

    # Определяем параметры для модели. Здесь max_num_words и max_sequence_length
    # уже были определены при подготовке данных.
    max_num_words = 10000
    max_sequence_length = data.shape[1]
    embedding_dim = 100  # Размерность эмбеддингов – можно корректировать по необходимости

    # Строим модель CNN с помощью функции build_cnn_model
    model = build_cnn_model(max_num_words, max_sequence_length, embedding_dim, num_classes)
    # Выводим краткое описание модели
    model.summary()

    # Обучаем модель. history хранит историю обучения (например, значения loss, accuracy)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size
    )

    # Оцениваем модель на тестовой выборке и выводим accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Test Accuracy:", accuracy)

    # Получаем предсказания модели в виде вероятностей
    y_pred_prob = model.predict(X_test)
    # Преобразуем вероятности в предсказанные классы (индекс максимального значения)
    y_pred = np.argmax(y_pred_prob, axis=1)
    # Аналогично, преобразуем one-hot метки обратно в индексы
    y_true = np.argmax(y_test, axis=1)

    # Выводим подробный отчет по классификации с метриками для каждого класса
    print(classification_report(y_true, y_pred, target_names=['negative', 'neutral', 'positive']))

    # Сохраняем обученную модель в формате .h5, чтобы потом можно было её загрузить для инференса
    model.save("model/cnn_model.h5")
    print("CNN модель сохранена в 'model/cnn_model.h5'.")

    # Сохраняем объект токенизатора в pickle-файл для последующего использования при обработке новых текстов
    with open('model/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Tokenizer сохранён в 'model/tokenizer.pickle'.")

    return model, history


# ---------------------------
# 7. Основная функция для запуска всего кода
# ---------------------------
def main():
    """
    Основная функция для запуска обучения CNN.

    Шаги:
      1. Загрузка данных из CSV (например, newsv6.csv).
      2. Подготовка данных для CNN: дискретизация тональности, очистка текста,
         токенизация, паддинг и преобразование меток в one-hot представление.
      3. Обучение и оценка модели CNN с выводом метрик.
    """
    # Указываем путь к файлу с новостями. Путь можно изменить, если файл лежит в другом каталоге.
    data_filepath = "../static/newsv6.csv"

    # Загружаем данные из CSV-файла с использованием функции load_data
    df = load_data(data_filepath)

    # Подготавливаем данные для CNN:
    #   - Применяем дискретизацию тональности (assign_label) с заданным epsilon (по умолчанию 0.0)
    #   - Очищаем тексты, токенизируем, выполняем padding, и преобразуем метки в one-hot представление
    data, labels_categorical, tokenizer, num_classes = prepare_cnn_data(
        df, epsilon=0.0, max_num_words=10000, max_sequence_length=100
    )

    # Обучаем и оцениваем модель CNN, передавая подготовленные данные и гиперпараметры обучения
    model, history = train_and_evaluate_cnn(
        data, labels_categorical, tokenizer, num_classes,
        epochs=10, batch_size=32, test_size=0.2
    )


# Если данный скрипт запускается напрямую, вызывается функция main
if __name__ == "__main__":
    main()
