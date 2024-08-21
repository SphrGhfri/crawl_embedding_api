import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Text
from dotenv import load_dotenv

app = FastAPI()

# Load environment variables from .env file
load_dotenv()

# Get DATABASE_URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the environment.")

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Define the table for embeddings
class EmbeddingTable(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True, index=True)
    embedding = Column(Text, nullable=False)
    data_row = Column(Text, nullable=False)


# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


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

    # Database session
    db_session = SessionLocal()

    # Query the embeddings from the database
    embedding_records = db_session.query(EmbeddingTable).all()
    db_session.close()

    best_match = None
    best_distance = float("inf")

    for record in embedding_records:
        db_embedding = np.array(record.embedding.split(",")).astype(float)
        distance = cosine(query_embedding, db_embedding)

        # Check if the current distance is smaller than the best distance found so far
        if distance < best_distance:
            best_distance = distance
            best_match = {"row_number": record.id, "related_row": record.data_row}

    # Return the best matching row
    if best_match:
        return {"best_match": best_match}
    else:
        raise HTTPException(status_code=404, detail="No matching row found.")
