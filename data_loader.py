# data_loader.py
# This file does two things:
# 1. Loads the SHL assessments from our CSV file
# 2. Converts the text descriptions into embeddings (vectors/numbers)

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------
# STEP A: Load the CSV file
# -------------------------------------------------------
def load_assessments(csv_path: str = "shl_assessments.csv"):
    """
    Reads the CSV file and returns it as a pandas DataFrame.
    A DataFrame is like an Excel table in Python.
    """
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} assessments from CSV")
    return df


# -------------------------------------------------------
# STEP B: Create text for embedding
# -------------------------------------------------------
def create_embedding_text(row) -> str:
    """
    For each assessment row, we combine multiple columns
    into one sentence. This gives the AI more context
    when searching for matches.
    
    Example output:
    "Verify - Numerical Reasoning. Measures ability to interpret
    numerical data. Job levels: Graduate, Manager. Type: Ability & Aptitude."
    """
    return (
        f"{row['name']}. "
        f"{row['description']} "
        f"Job levels: {row['job_levels']}. "
        f"Test type: {row['test_type']}. "
        f"Duration: {row['duration_minutes']} minutes."
    )


# -------------------------------------------------------
# STEP C: Generate embeddings using sentence-transformers
# -------------------------------------------------------
def generate_embeddings(df: pd.DataFrame):
    """
    Converts each assessment's text into a vector (list of numbers).
    We use a pre-trained AI model called 'all-MiniLM-L6-v2'.
    
    This model is small, fast, and works great for semantic search.
    It will be downloaded automatically the first time (~80MB).
    """
    print("🤖 Loading AI embedding model...")
    
    # Load the pre-trained model
    # This downloads automatically on first run
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("📝 Creating text for each assessment...")
    
    # Create a combined text for each assessment
    texts = [create_embedding_text(row) for _, row in df.iterrows()]
    
    print("⚡ Generating embeddings (converting text to numbers)...")
    
    # Convert all texts to embeddings at once
    # This returns a 2D array: one row per assessment, 384 numbers per row
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print(f"✅ Generated embeddings with shape: {embeddings.shape}")
    print(f"   → {embeddings.shape[0]} assessments")
    print(f"   → {embeddings.shape[1]} numbers per assessment")
    
    return model, embeddings, texts


# -------------------------------------------------------
# STEP D: Test everything works
# -------------------------------------------------------
if __name__ == "__main__":
    # This block only runs when you run this file directly
    # It will NOT run when another file imports this file
    
    print("=" * 50)
    print("Testing data loader...")
    print("=" * 50)
    
    # Load the CSV
    df = load_assessments()
    
    # Show the first 3 assessments
    print("\n📋 First 3 assessments:")
    print(df[['name', 'test_type', 'duration_minutes']].head(3))
    
    # Generate embeddings
    model, embeddings, texts = generate_embeddings(df)
    
    print("\n🎉 Everything working! Ready for Phase 3.")