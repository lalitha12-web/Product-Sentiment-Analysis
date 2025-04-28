# === Imports ===
import pandas as pd
import re
import nltk
import spacy
import pickle as pk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import nltk
nltk.download('punkt_tab')

# === Downloads ===
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# === Load SpaCy model with fallback ===
try:
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
except OSError:
    raise OSError("SpaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")

# === Load Pickles ===
pickle_dir = 'pickle_file'
with open(os.path.join(pickle_dir, 'count_vector.pkl'), 'rb') as f:
    count_vector = pk.load(f)

with open(os.path.join(pickle_dir, 'tfidf_transformer.pkl'), 'rb') as f:
    tfidf_transformer = pk.load(f)

with open(os.path.join(pickle_dir, 'model.pkl'), 'rb') as f:
    model = pk.load(f)

with open(os.path.join(pickle_dir, 'user_final_rating.pkl'), 'rb') as f:
    recommend_matrix = pk.load(f)

# === Load Dataset ===
product_df = pd.read_csv(r'/Users/lalitha/Downloads/sentimental_analysis/sample30.csv', sep=",")

# === Text Cleaning Pipeline ===
stopword_list = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    return re.sub(pattern, '', text)

def normalize_text(text):
    text = remove_special_characters(text)
    words = word_tokenize(text.lower())
    words = [re.sub(r'[^\w\s]', '', w) for w in words if w not in stopword_list]
    words = [remove_special_characters(w, True) for w in words if w]
    return words

def lemmatize_words(words):
    return [lemmatizer.lemmatize(w, pos='v') for w in words]

def normalize_and_lemmatize(text):
    words = normalize_text(text)
    lemmas = lemmatize_words(words)
    return ' '.join(lemmas)

# === Sentiment Model Prediction ===
def model_predict(text_series):
    word_vector = count_vector.transform(text_series)
    tfidf_vector = tfidf_transformer.transform(word_vector)
    return model.predict(tfidf_vector)

# === Recommend Products ===
def recommend_products(user_name):
    if user_name not in recommend_matrix.index:
        raise ValueError(f"User '{user_name}' not found in recommendation matrix.")

    top_products = recommend_matrix.loc[user_name].sort_values(ascending=False).head(20)
    product_names = top_products.index.tolist()

    product_frame = product_df[product_df['name'].isin(product_names)]
    output_df = product_frame[['name', 'reviews_text']].copy()

    output_df['lemmatized_text'] = output_df['reviews_text'].apply(normalize_and_lemmatize)
    output_df['predicted_sentiment'] = model_predict(output_df['lemmatized_text'])

    return output_df

# === Get Top 5 Sentiment-Positive Products ===
def top5_products(df):
    total_product = df.groupby('name')['reviews_text'].count().reset_index(name='total_reviews')
    sentiment_count = df.groupby(['name', 'predicted_sentiment'])['reviews_text'].count().reset_index(name='sentiment_count')

    merged = pd.merge(sentiment_count, total_product, on='name')
    merged['percentage'] = (merged['sentiment_count'] / merged['total_reviews']) * 100
    merged = merged[merged['predicted_sentiment'] == 1].sort_values(by='percentage', ascending=False)

    return merged[['name']].head(5)
