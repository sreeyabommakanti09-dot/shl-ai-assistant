# data_loader.py
import pandas as pd

def load_assessments(csv_path="shl_assessments.csv"):
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} assessments from CSV")
    return df

def create_embedding_text(row):
    return (
        f"{row['name']}. "
        f"{row['description']} "
        f"Job levels: {row['job_levels']}. "
        f"Test type: {row['test_type']}. "
        f"Duration: {row['duration_minutes']} minutes."
    )