# search_engine.py
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_assessments(csv_path="shl_assessments.csv"):
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} assessments from CSV")
    return df

def create_search_text(row):
    return (
        f"{row['name']} "
        f"{row['description']} "
        f"{row['job_levels']} "
        f"{row['test_type']}"
    )

def initialize_search_system():
    print("🚀 Initializing search system...")
    df = load_assessments()
    
    # Create search texts
    texts = [create_search_text(row) for _, row in df.iterrows()]
    
    # Build TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        max_features=5000
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    print("✅ Search system ready!")
    return vectorizer, tfidf_matrix, df

def search_assessments(query, model, index, df, top_k=5):
    # model = vectorizer, index = tfidf_matrix
    vectorizer = model
    tfidf_matrix = index
    
    # Transform query
    query_vector = vectorizer.transform([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    # Get top results
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
    "name": row["name"],
    "description": row["description"],
    "job_levels": row["job_levels"],
    "duration_minutes": int(row["duration_minutes"]),
    "test_type": row["test_type"],
    "remote_testing": row["remote_testing"],
    "adaptive": row["adaptive"],
    "url": row["url"],
    "similarity_score": round(float(similarities[idx]), 3)

        })
    
    return results