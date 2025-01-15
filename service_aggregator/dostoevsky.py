from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


df = pd.read_csv("../last-work/sentiment-analysis/static/news_sentiment.csv")
df.columns = ['title', 'date', 'label']

def get_sentiment(label: int) -> str:
    if label == 0:
        return "neutral"
    elif label == 1:
        return "positive"
    elif label == -1:
        return "negative"
    else:
        return "neutral"

df['label'] = df['label'].apply(get_sentiment)

X = np.array(df['title']).tolist()
y = np.array(df['label'].tolist())

tokenizer = RegexTokenizer()
model = FastTextSocialNetworkModel(tokenizer=tokenizer)
result = model.predict(X, k=3)
print(result)
predicted_labels = [max(item, key=item.get) for item in result]

accuracy = accuracy_score(y, predicted_labels)
print(f' AUC: {accuracy:.4f}')







