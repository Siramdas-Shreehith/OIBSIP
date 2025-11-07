import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['Label', 'Message']
data['Label'] = data['Label'].map({'ham': 0, 'spam': 1})

plt.figure(figsize=(6,4))
sns.countplot(x='Label', data=data, palette='coolwarm')
plt.title('Spam vs Ham Distribution')
plt.xlabel('Email Type (0 = Ham, 1 = Spam)')
plt.ylabel('Count')
plt.show()

X = data['Message']
y = data['Label']
cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(" Spam Detection Results ")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

sample = ["Congratulations! You've won a free iPhone. Click here to claim now!"]
sample_vector = cv.transform(sample)
prediction = model.predict(sample_vector)
print("\nSample Test Email:", sample[0])
print("Prediction:", "Spam" if prediction[0] == 1 else "Ham")
