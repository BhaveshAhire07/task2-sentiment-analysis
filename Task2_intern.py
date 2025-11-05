# Task 2: Sentiment Analysis with NLP for Codtech Internship
# Description: Perform sentiment analysis on customer reviews using TF-IDF vectorization and Logistic Regression.
#              Includes preprocessing, modeling, evaluation, and visualization.
# Author: [Your Name] | Date: October 25, 2025
# Dataset: IMDB Movie Reviews (download from Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

# Step 0: Import libraries
import pandas as pd  # Data handling
import numpy as np  # Numerical ops
import re  # Regex for cleaning
import nltk  # NLP toolkit
from nltk.corpus import stopwords  # Common words to remove
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF
from sklearn.model_selection import train_test_split  # Data split
from sklearn.linear_model import LogisticRegression  # Classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve  # Metrics
from sklearn.pipeline import Pipeline  # For streamlined workflow
import matplotlib.pyplot as plt  # Plots
from wordcloud import WordCloud  # Word clouds (pip install wordcloud)

# Download NLTK data (run once)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

print("Libraries loaded!")

# Step 1: Load and Prep Data
# Option 1: IMDB CSV (recommended; ~50k reviews)
try:
    df = pd.read_csv('IMDB Dataset.csv')  # Adjust path if needed
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})  # Binary: 1=pos, 0=neg
    X = df['review']  # Text column
    y = df['sentiment']  # Labels
    print(f"IMDB Dataset loaded: {df.shape[0]} reviews")
    print(df.head(2))
except FileNotFoundError:
    print("IMDB CSV not found. Using fallback: 20newsgroups (binary proxy).")
    # Option 2: Fallback with sklearn (sci.space=neg, rec.autos=pos)
    from sklearn.datasets import fetch_20newsgroups
    newsgroups = fetch_20newsgroups(subset='train', categories=['sci.space', 'rec.autos'], remove=('headers', 'footers', 'quotes'))
    X = newsgroups.data
    y = (newsgroups.target == 0).astype(int)  # 0=space (neg), 1=autos (pos)
    df = pd.DataFrame({'review': X, 'sentiment': y})
    print(f"Fallback Dataset: {len(X)} samples")

# Text Cleaning Function
def preprocess(text):
    # Lowercase, remove non-alpha, remove stopwords, join words
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())  # Remove punctuation/numbers
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Apply preprocessing
X_clean = [preprocess(text) for text in X]
print(f"Sample cleaned review: {X_clean[0][:100]}...")

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42, stratify=y)  # Stratify for balance
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Step 2: TF-IDF Vectorization + Model Pipeline
# Pipeline: Chains vectorizer + classifier for easy reuse
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),  # Unigrams + bigrams, top 5k features
    ('lr', LogisticRegression(random_state=42, max_iter=200))  # Logistic Regression
])

# Train the model
pipeline.fit(X_train, y_train)
print("Model trained!")

# Step 3: Predict and Evaluate
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Prob for positive class

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nModel Performance:")
print(f"Test Accuracy: {accuracy:.2f} ({accuracy*100:.1f}%)")
print(f"AUC-ROC: {auc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.yticks([0, 1], ['Negative', 'Positive'])
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_sentiment.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve_sentiment.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 4: Sample Predictions
sample_reviews = [
    "This movie was amazing and thrilling!",
    "Terrible plot, wasted my time."
]
for review in sample_reviews:
    clean_review = preprocess(review)
    pred = pipeline.predict([clean_review])[0]
    prob = pipeline.predict_proba([clean_review])[0][1]
    print(f"Review: '{review}' → Predicted: {'Positive' if pred == 1 else 'Negative'} (Prob: {prob:.2f})")

# Step 5: Visualization - Word Clouds (Top words in positive/negative reviews)
# Positive words
pos_reviews = ' '.join(df[df['sentiment'] == 1]['review'].apply(preprocess))
pos_cloud = WordCloud(width=800, height=400, background_color='white').generate(pos_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(pos_cloud, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Sentiment Word Cloud')
plt.savefig('positive_wordcloud.png', dpi=300, bbox_inches='tight')
plt.show()

# Negative words
neg_reviews = ' '.join(df[df['sentiment'] == 0]['review'].apply(preprocess))
neg_cloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(neg_cloud, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Sentiment Word Cloud')
plt.savefig('negative_wordcloud.png', dpi=300, bbox_inches='tight')
plt.show()

# End: Summary
print("\n--- Analysis Summary ---")
print(f"- Dataset: IMDB Reviews (balanced binary classification).")
print(f"- Performance: Accuracy {accuracy:.2f}, AUC {auc:.2f} (solid for text; TF-IDF captures context like 'great acting' vs. 'boring').")
print("- Key Insights: Preprocessing removes noise (e.g., stopwords); bigrams (ngram=2) help with phrases like 'not good'.")
print("- Pros: Logistic Reg fast/interpretible (coefficients show word impact). Cons: Ignores word order (use LSTM for advanced).")
print("- Files Saved: confusion_matrix_sentiment.png, roc_curve_sentiment.png, wordclouds (upload to GitHub).")
print("- Extensions: Tune C=1.0 via GridSearchCV; try CountVectorizer for comparison.")
