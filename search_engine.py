# search_engine.py
# This file builds a FAISS index from our embeddings
# and provides a function to search for relevant assessments.

import faiss
import numpy as np
import pandas as pd
from data_loader import load_assessments, generate_embeddings

# -------------------------------------------------------
# STEP A: Build the FAISS index
# -------------------------------------------------------
def build_faiss_index(embeddings: np.ndarray):
    """
    Builds a FAISS index from our embeddings.
    
    Think of this like building a smart filing cabinet.
    We put all 20 assessment vectors inside it.
    Later we can ask it: "which vectors are closest to this query?"
    
    We use IndexFlatL2 which measures straight-line distance
    between vectors. Simple and reliable for small datasets.
    """
    # Get the size of each embedding vector (384 in our case)
    dimension = embeddings.shape[1]
    
    # Create a flat L2 index (L2 = straight line distance)
    index = faiss.IndexFlatL2(dimension)
    
    # Add all our assessment embeddings to the index
    # FAISS requires float32 format
    index.add(embeddings.astype(np.float32))
    
    print(f"✅ FAISS index built with {index.ntotal} assessments")
    return index


# -------------------------------------------------------
# STEP B: Search the index
# -------------------------------------------------------
def search_assessments(
    query: str,
    model,
    index,
    df: pd.DataFrame,
    top_k: int = 5
):
    """
    Takes a recruiter's query (plain English text),
    converts it to a vector, and finds the top_k
    most similar assessments in the FAISS index.
    
    Parameters:
        query   → the recruiter's question as a string
        model   → our sentence-transformer model
        index   → the FAISS index we built
        df      → our dataframe with assessment details
        top_k   → how many results to return (default 5)
    
    Returns:
        A list of dictionaries, each with assessment details
    """
    # Step 1: Convert the query text into a vector
    query_vector = model.encode([query])
    
    # Step 2: Search FAISS for the top_k closest vectors
    # distances → how far each result is (lower = better match)
    # indices   → which row numbers in our dataframe matched
    distances, indices = index.search(
        query_vector.astype(np.float32),
        top_k
    )
    
    # Step 3: Build results list
    results = []
    for i, idx in enumerate(indices[0]):
        # Skip invalid indices (shouldn't happen but safe to check)
        if idx == -1:
            continue
            
        # Get the assessment row from our dataframe
        row = df.iloc[idx]
        
        # Convert distance to a similarity score (0 to 1)
        # Lower distance = higher similarity
        similarity = float(1 / (1 + distances[0][i]))
        
        results.append({
            "name": row["name"],
            "description": row["description"],
            "job_levels": row["job_levels"],
            "duration_minutes": int(row["duration_minutes"]),
            "test_type": row["test_type"],
            "remote_testing": row["remote_testing"],
            "adaptive": row["adaptive"],
            "similarity_score": round(similarity, 3)
        })
    
    return results


# -------------------------------------------------------
# STEP C: Initialize everything together
# -------------------------------------------------------
def initialize_search_system():
    """
    This function loads everything we need:
    1. The CSV data
    2. The AI embedding model
    3. The FAISS index
    
    We call this ONCE when the server starts.
    Then we reuse the same model and index for every search.
    This is important — loading the model takes a few seconds,
    so we don't want to reload it on every request!
    """
    print("🚀 Initializing search system...")
    
    # Load assessments from CSV
    df = load_assessments()
    
    # Generate embeddings for all assessments
    from data_loader import generate_embeddings
    model, embeddings, texts = generate_embeddings(df)
    
    # Build FAISS index
    index = build_faiss_index(embeddings)
    
    print("✅ Search system ready!")
    return model, index, df


# -------------------------------------------------------
# STEP D: Test everything works
# -------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("Testing FAISS search engine...")
    print("=" * 50)
    
    # Initialize the system
    model, index, df = initialize_search_system()
    
    # Test queries — these simulate what a recruiter might type
    test_queries = [
        "I need a test for logical reasoning and problem solving",
        "Looking for personality assessment for sales role",
        "Assessment for customer service representative",
        "Numerical and data analysis test for finance role",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 40)
        
        results = search_assessments(query, model, index, df, top_k=3)
        
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['name']}")
            print(f"     Type: {result['test_type']}")
            print(f"     Duration: {result['duration_minutes']} mins")
            print(f"     Similarity: {result['similarity_score']}")