import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

# Loading Dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

data = pd.read_csv(url, sep='\t', names=['label', 'text'])
data.columns = ['label', 'text']

# Cleaning the Text
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

data['cleaned_text'] = data['text'].apply(clean_text)

# Training the Model
X = data['cleaned_text']
y = data['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Testing the Accuracy
predictions = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")

# Trying Customized Input
def predict_spam(input_text):
    cleaned = clean_text(input_text)
    vec = vectorizer.transform([cleaned])
    return "Spam!" if model.predict(vec)[0] == 1 else "Not Spam"

print(predict_spam("WINNER!! Claim your prize now!"))  # Should print "Spam!"