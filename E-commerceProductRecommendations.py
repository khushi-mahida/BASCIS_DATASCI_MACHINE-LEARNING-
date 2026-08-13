# Day 4 E-commerce Product Recommendations

# pip install pandas numpy scikit-learn seaborn matplotlib plotly

# ------------------ Synthetic Data ------------------
import pandas as pd
import numpy as np

users = np.random.choice(['User_' + str(i) for i in range(1, 101)], size=1000, replace=True)
products = np.random.choice(['Product_' + str(i) for i in range(1, 21)], size=1000, replace=True)
ratings = np.random.randint(1, 6, size=1000)

data = pd.DataFrame({'UserID': users, 'ProductID': products, 'Rating': ratings})
print("Synthetic e-commerce recommendation data created.")


# ------------------ E-commerce Product Recommendations ------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

user_product_matrix = data.pivot_table(index='UserID', columns='ProductID', values='Rating').fillna(0)
similarity_matrix = cosine_similarity(user_product_matrix)
print("E-commerce product recommendation system built.")

# ------------------ Predictions ------------------
def predict_rating(user_id, product_id, user_product_matrix, similarity_matrix):
    # Get the index of the user and product
    user_index = user_product_matrix.index.get_loc(user_id)
    product_index = user_product_matrix.columns.get_loc(product_id)

    # Get the similarity scores for the target user
    user_similarity_scores = similarity_matrix[user_index]

    # Get the ratings for the target product
    product_ratings = user_product_matrix.iloc[:, product_index]

    # Calculate the weighted average of ratings
    weighted_sum = np.dot(user_similarity_scores, product_ratings)
    sum_of_weights = np.sum(user_similarity_scores)

    if sum_of_weights == 0:
        return 0  # Avoid division by zero

    predicted_rating = weighted_sum / sum_of_weights
    return predicted_rating


# Example usage
user_id = 'User_1'
product_id = 'Product_1'
predicted_rating = predict_rating(user_id, product_id, user_product_matrix, similarity_matrix)
print(f'Predicted rating for {user_id} on {product_id}: {predicted_rating}')

# ------------------ Recommendations ------------------
def recommend_products(user_id, user_product_matrix, similarity_matrix, top_n=5):
    # Get the index of the user
    user_index = user_product_matrix.index.get_loc(user_id)

    # Get the similarity scores for the target user
    user_similarity_scores = similarity_matrix[user_index]

    # Predict ratings for all products
    predicted_ratings = []
    for product_id in user_product_matrix.columns:
        if user_product_matrix.at[user_id, product_id] == 0:  # Only predict for products not rated by the user
            product_index = user_product_matrix.columns.get_loc(product_id)
            product_ratings = user_product_matrix.iloc[:, product_index]
            weighted_sum = np.dot(user_similarity_scores, product_ratings)
            sum_of_weights = np.sum(user_similarity_scores)
            predicted_rating = weighted_sum / sum_of_weights if sum_of_weights != 0 else 0
            predicted_ratings.append((product_id, predicted_rating))

    # Sort products by predicted rating in descending order
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)

    # Return the top N recommended products
    recommended_products = [product_id for product_id, rating in predicted_ratings[:top_n]]
    return recommended_products

# Example usage
user_id = 'User_10'
recommended_products = recommend_products(user_id, user_product_matrix, similarity_matrix, top_n=3)
print(f'Recommended products for {user_id}: {recommended_products}')

# ------------------ Plots ------------------
import seaborn as sns
import matplotlib.pyplot as plt

# User-Product Rating Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(user_product_matrix, cmap='viridis', cbar=True)
plt.title('User-Product Rating Heatmap')
plt.xlabel('ProductID')
plt.ylabel('UserID')
plt.show()

# User Similarity Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(similarity_matrix, cmap='coolwarm', cbar=True)
plt.title('User Similarity Heatmap')
plt.xlabel('UserID')
plt.ylabel('UserID')
plt.show()

# Top-N Product Recommendations for a specific user
def plot_recommendations(user_id, user_product_matrix, similarity_matrix, top_n=5):
    recommended_products = recommend_products(user_id, user_product_matrix, similarity_matrix, top_n)
    ratings = [predict_rating(user_id, product_id, user_product_matrix, similarity_matrix) for product_id in recommended_products]

    plt.figure(figsize=(10, 6))
    plt.barh(recommended_products, ratings, color='skyblue')
    plt.xlabel('Predicted Rating')
    plt.ylabel('ProductID')
    plt.title(f'Top {top_n} Product Recommendations for {user_id}')
    plt.gca().invert_yaxis()
    plt.show()

# Example usage
plot_recommendations('User_10', user_product_matrix, similarity_matrix, top_n=5)

