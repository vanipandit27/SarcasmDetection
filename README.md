# SarcasmDetection
Developed a sarcasm detection system using Random Forest and SVM algorithms on  a dataset of 1 million entries, achieving an accuracy of 85%. • Implemented advanced data pre-processing techniques, including text cleaning,  tokenization, and feature engineering, resulting in a 20% improvement in accuracy  for sarcasm detection models

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/saarcasm.csv')
df.head()

df.columns

missing_values = df.isnull().sum()
print(missing_values)

df.fillna(df.mean(), inplace=True)

import nltk
import re
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean(comment):
    comment = str(comment).lower()
    comment = re.sub('\[.*?\]', '', comment)
    comment = re.sub('https?://\S+|www\.\S+', '', comment)
    comment = re.sub('<.*?>+', '', comment)
    comment = re.sub('[%s]' % re.escape(string.punctuation), '', comment)
    comment = re.sub('\n', '', comment)
    comment = re.sub('\w*\d\w*', '', comment)
    comment = [word for word in comment.split(' ') if word not in stopword]
    comment=" ".join(comment)
    comment = [stemmer.stem(word) for word in comment.split(' ')]
    comment=" ".join(comment)
    return comment
df["comment"] = df["comment"].apply(clean)

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
comment = " ".join(i for i in df.comment)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color="white").generate(comment)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

df["label"] = df["label"].map({0: "Not Sarcastic", 1: "Sarcastic"})
df = df[["comment", "label"]]
print(df.head())

X = df['comment']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

rfc = RandomForestClassifier(n_estimators=100, random_state=62)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
print("Random Forest Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_rfc))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rfc))
print("Classification Report:")
print(classification_report(y_test, y_pred_rfc))

svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))

import matplotlib.pyplot as plt

rfc_score = accuracy_score(y_test, y_pred_rfc)
svm_score = accuracy_score(y_test, y_pred_svm)

fig, ax = plt.subplots()
ax.bar(['Random Forest', 'SVM'], [rfc_score, svm_score])
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of Random Forest and SVM on Sarcasm Dataset')
plt.show()

new_comment = "Hey it's okay they were rained on by debris and body bits coz Trump screwed him on the deal...right guys?"
new_comment_vectorized = vectorizer.transform([new_comment])
rfc_predicted_label = rfc.predict(new_comment_vectorized)
svm_predicted_label = svm.predict(new_comment_vectorized)
print("Random Forest predicted label:", rfc_predicted_label)
print("SVM predicted label:", svm_predicted_label)

new_comment = "Loving this West Ham implosion"
new_comment_vectorized = vectorizer.transform([new_comment])
rfc_predicted_label = rfc.predict(new_comment_vectorized)
svm_predicted_label = svm.predict(new_comment_vectorized)
print("Random Forest predicted label:", rfc_predicted_label)
print("SVM predicted label:", svm_predicted_label)
