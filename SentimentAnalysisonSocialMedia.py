import pandas as pd

texts = ['I love this!', 'This is terrible.', 'Absolutely amazing!', 'Not good.', 'Pretty decent.', 'Worst experience ever.']
sentiments = [1, 0, 1, 0, 1, 0]

data = pd.DataFrame({'Text': texts, 'Sentiment': sentiments})
print("Synthetic social media sentiment data created.")



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Text'])
y = data['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Sentiment analysis accuracy: {accuracy:.2f}')

import matplotlib.pyplot as plt

# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
from sklearn.metrics import roc_curve, RocCurveDisplay
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
roc_disp.plot()
plt.title('ROC Curve')
plt.show()

# Word Cloud
from wordcloud import WordCloud
positive_texts = ' '.join(data[data['Sentiment'] == 1]['Text'])
negative_texts = ' '.join(data[data['Sentiment'] == 0]['Text'])

positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_texts)
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_texts)

# Display word clouds
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Positive Sentiments')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Negative Sentiments')
plt.axis('off')
plt.show()

# pip install numpy==1.23.5
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(data['Text'])
sequences = tokenizer.texts_to_sequences(data['Text'])
padded_sequences = pad_sequences(sequences, maxlen=10, padding='post')

X = padded_sequences
y = data['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=16, input_length=10))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Sentiment analysis accuracy: {accuracy:.2f}')

