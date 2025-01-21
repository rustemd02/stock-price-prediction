import pandas as pd
from datetime import datetime

# Загружаем данные из CSV-файлов
df1 = pd.read_csv('../last-work/sentiment-analysis/news_sentimentv5.csv')
df2 = pd.read_csv('../static/news_sentiment.csv')

# Преобразуем строки с датами в формат datetime в df1 и форматируем их как строки "YYYY-MM-DD"
df1['TRADEDATE'] = pd.to_datetime(df1['date'].str.strip(), format="%Y-%m-%d %H:%M:%S").dt.strftime("%Y-%m-%d")

# Инициализируем список для новых данных
new_data = []

# Проходим по каждой строке в df1 и переносим данные в df2
for index, row in df1.iterrows():
    new_data.append([row['TRADEDATE'], row['label']])

# Создаем новый DataFrame из новых данных
new_df = pd.DataFrame(new_data, columns=['TRADEDATE', 'SentScore'])

# Сортируем данные по дате от старых к новым
new_df = new_df.sort_values(by='TRADEDATE')

# Сохраняем данные в CSV-файл
new_df.to_csv('../last-work/news_sentiment.csv', index=False)

print("Данные успешно обновлены и сохранены.")


