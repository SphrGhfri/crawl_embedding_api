import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine

app = FastAPI()

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# Function to load embeddings and corresponding rows from a CSV file
def load_embeddings_and_data(file_path):
    df = pd.read_csv(file_path)
    embeddings = {}
    data_rows = {}
    for _, row in df.iterrows():
        index = int(row["Row"])
        embedding = np.array(row[1:-1]).astype(float)  # Exclude last column (data row)
        data_row = row.iloc[-1]  # Last column is the original data row
        embeddings[index] = embedding
        data_rows[index] = data_row
    return embeddings, data_rows


# Load embeddings and corresponding data rows
embeddings, data_rows = load_embeddings_and_data("embeddings.csv")


class Query(BaseModel):
    text: str


# Function to convert text to embedding
def text_to_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    return embedding.numpy()


# Endpoint to get the most similar row
@app.post("/find_related_row/")
async def find_related_row(query: Query):
    query_embedding = text_to_embedding(query.text)

    # Ensure query embedding has the same shape as loaded embeddings
    existing_embedding_shape = next(iter(embeddings.values())).shape
    if query_embedding.shape != existing_embedding_shape:
        raise HTTPException(
            status_code=400,
            detail="Embedding shape mismatch. Check your model or data processing.",
        )

    # Find the top 4 most similar embeddings
    top_n = 3
    distances = []

    for index, emb in embeddings.items():
        distance = cosine(query_embedding, emb)
        distances.append((distance, index))

    # Sort distances and get the top N closest rows
    distances.sort(key=lambda x: x[0])
    top_closest_rows = [index for _, index in distances[:top_n]]

    # Prepare the response with the top 4 closest rows
    results = [{"row_number": row + 1, "related_row": data_rows[row]} for row in top_closest_rows]
    return {"top_rows": results}
