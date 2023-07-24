import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
def analyse_sentiment(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]


    nltk.download('stopwords')
    stemmer = nltk.SnowballStemmer('english')
    from nltk.corpus import stopwords
    import string
    stopword = set(stopwords.words('english'))

    def clean(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = [word for word in text.split(' ') if word not in stopword]
        text = " ".join(text)
        text = [stemmer.stem(word) for word in text.split(' ')]
        text = " ".join(text)
        return text

    df['message'] = df['message'].astype(str)
    df['message'] = df['message'].apply(clean)

    nltk.download('vader_lexicon')
    sentiments = SentimentIntensityAnalyzer()
    df["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]
    df["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]
    df["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]
    df = df[["message", "Positive", "Negative", "Neutral"]]

    X = round(sum(df['Positive']),2)
    Y = round(sum(df['Negative']),2)
    Z = round(sum(df['Neutral']),2)

    return X,Y,Z

def sentiment_score(a, b, c):
    if (a > b) and (a > c):
        an = "Positive 😊 "
    elif (b > a) and (b > c):
        an = "Negative 😠 "
    else:
        an = "Neutral 🙂 "
        return an