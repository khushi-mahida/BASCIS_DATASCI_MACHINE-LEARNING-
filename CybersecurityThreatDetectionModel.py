import pandas as pd
import numpy as np

np.random.seed(56)
num_samples = 1000
normal_traffic = np.random.normal(loc=100, scale=20, size=(num_samples, 3))
anomalous_traffic = np.random.normal(loc=200, scale=50, size=(int(num_samples * 0.1), 3))

data = np.vstack((normal_traffic, anomalous_traffic))
labels = np.hstack((np.zeros(num_samples), np.ones(int(num_samples * 0.1))))

df = pd.DataFrame(data, columns=['PacketSize', 'ConnectionDuration', 'BytesTransferred'])
df['Anomaly'] = labels
print("Synthetic cybersecurity threat data created.")


# The algorithm assumes that anomalies are rare and different. Thus, anomalies require fewer partitions to be isolated,
# resulting in shorter paths in the tree.

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X = df[['PacketSize', 'ConnectionDuration', 'BytesTransferred']]
y = df['Anomaly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

# Predict and convert predictions to binary (0 for inliers, 1 for outliers)
predictions = model.predict(X_test)
predictions = np.where(predictions == 1, 0, 1)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f'Cybersecurity threat detection accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')


import numpy as np
import matplotlib.pyplot as plt



# Calculate feature importance
n_features = X.shape[1]
feature_importance = np.zeros(n_features)

for i in range(n_features):
    feature_importance[i] = np.mean(np.abs(model.estimators_[0].tree_.feature == i))

# Normalize feature importance
feature_importance = feature_importance / np.sum(feature_importance)

# Get feature names
features = X.columns


plt.figure(figsize=(10, 6))
plt.bar(features, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Anomaly Detection')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


for feature, importance in zip(features, feature_importance):
    print(f"{feature}: {importance:.4f}")


def predict_anomaly(packet_size, connection_duration, bytes_transferred, model):
    input_data = pd.DataFrame({'PacketSize': [packet_size],
                               'ConnectionDuration': [connection_duration],
                               'BytesTransferred': [bytes_transferred]})
    prediction = model.predict(input_data)[0]
    return 1 if prediction == -1 else 0

# Example usage
packet_size = 150
connection_duration = 120
bytes_transferred = 5000
prediction = predict_anomaly(packet_size, connection_duration, bytes_transferred, model)
print(f'Predicted anomaly for traffic with packet size {packet_size}, '
      f'connection duration {connection_duration}, and '
      f'bytes transferred {bytes_transferred}: {prediction}')



# ------------------ Recommendations ------------------
def recommend_action(packet_size, connection_duration, bytes_transferred, model):
    input_data = pd.DataFrame({'PacketSize': [packet_size], 
                               'ConnectionDuration': [connection_duration], 
                               'BytesTransferred': [bytes_transferred]})
    prediction = model.predict(input_data)[0]

    if prediction == -1:
        return 'Potential security threat detected. Recommend further investigation and possible blocking of the connection.'
    else:
        return 'No anomaly detected. Continue monitoring.'

# Example usage
packet_size = 250
connection_duration = 180
bytes_transferred = 8000
recommendation = recommend_action(packet_size, connection_duration, bytes_transferred, model)
print(recommendation)

import numpy as np
import pandas as pd


def generate_normal_traffic_examples(num_examples, base_packet_size, base_connection_duration, base_bytes_transferred):
    examples = []
    for _ in range(num_examples):
        packet_size = np.random.normal(base_packet_size, base_packet_size * 0.1)
        connection_duration = np.random.normal(base_connection_duration, base_connection_duration * 0.1)
        bytes_transferred = np.random.normal(base_bytes_transferred, base_bytes_transferred * 0.1)

        examples.append({
            'PacketSize': max(0, packet_size),
            'ConnectionDuration': max(0, connection_duration),
            'BytesTransferred': max(0, bytes_transferred)
        })
    return examples


def test_examples(examples, model):
    df = pd.DataFrame(examples)
    predictions = model.predict(df)
    return predictions


# Generate examples
normal_examples = generate_normal_traffic_examples(10, 150, 100, 100)

# Test examples
results = test_examples(normal_examples, model)

for i, (example, result) in enumerate(zip(normal_examples, results)):
    print(f"Example {i + 1}:")
    print(f"  Packet Size: {example['PacketSize']:.2f}")
    print(f"  Connection Duration: {example['ConnectionDuration']:.2f}")
    print(f"  Bytes Transferred: {example['BytesTransferred']:.2f}")
    print(f"  Prediction: {'Normal' if result == 1 else 'Anomaly'}")
    print()

# Calculate and print the percentage of normal traffic
normal_percentage = (results == 1).mean() * 100
print(f"Percentage of traffic classified as normal: {normal_percentage:.2f}%")



# 1. Scatter plot of normal vs anomalous traffic
plt.figure(figsize=(12, 8))
plt.scatter(df[df['Anomaly'] == 0]['PacketSize'], df[df['Anomaly'] == 0]['ConnectionDuration'],
            alpha=0.5, label='Normal', color='blue')
plt.scatter(df[df['Anomaly'] == 1]['PacketSize'], df[df['Anomaly'] == 1]['ConnectionDuration'],
            alpha=0.5, label='Anomalous', color='red')
plt.xlabel('Packet Size')
plt.ylabel('Connection Duration')
plt.title('Normal vs Anomalous Traffic')
plt.legend()
plt.show()


# 2. Distribution of features
fig, axs = plt.subplots(3, 1, figsize=(12, 15))
features = ['PacketSize', 'ConnectionDuration', 'BytesTransferred']
for i, feature in enumerate(features):
    axs[i].hist(df[df['Anomaly'] == 0][feature], bins=50, alpha=0.5, label='Normal', color='blue')
    axs[i].hist(df[df['Anomaly'] == 1][feature], bins=50, alpha=0.5, label='Anomalous', color='red')
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('Frequency')
    axs[i].legend()
plt.tight_layout()
plt.show()

# 3. Correlation heat map 
import seaborn as sns
plt.figure(figsize=(10, 8))
correlation_matrix = df[['PacketSize', 'ConnectionDuration', 'BytesTransferred', 'Anomaly']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



# 4. ROC Curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, model.decision_function(X_test))
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 5. Decision boundary visualization 
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c='blue', alpha=0.8, label='Normal')
plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c='red', alpha=0.8, label='Anomalous')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Decision Boundary (2D Projection)')
plt.legend()
plt.show()

