import pandas as pd

df = pd.read_csv('IMDB Dataset.csv')

print(df.head())
print(df.shape)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
print(f"✅ Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

def predict_sentiment(review):
    review_tfidf = vectorizer.transform([review])
    result = model.predict(review_tfidf)
    return "😊 Positive" if result[0] == "positive" else "😞 Negative"

print(predict_sentiment("This movie was absolutely amazing!!"))
print(predict_sentiment("Worst film I have ever seen in my life!"))
print(predict_sentiment("Border 2 was outstanding, loved every second!!"))
