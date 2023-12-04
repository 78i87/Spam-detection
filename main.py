import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

#load dataset
pd.set_option('display.max_columns', None)
df = pd.read_csv('spam.csv', encoding = 'latin1')
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df = df.rename(columns={"v1":"label", "v2":"text"})
df['spam'] = df['label'].map( {'spam': 1, 'ham': 0} ).astype(int)
vectorizer = TfidfVectorizer()

# Fit and transform the text column
X = vectorizer.fit_transform(df['text'])
y = df['spam'].values

# Train and test sets split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Detailed classification report
print(classification_report(y_test, y_pred))
