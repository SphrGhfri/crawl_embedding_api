import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Initialize FastAPI application
app = FastAPI()

# Load environment variables from .env file
load_dotenv()

# Retrieve the database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the environment.")

# SQLAlchemy setup: Create database engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Define the EmbeddingTable model for storing embeddings
class EmbeddingTable(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    embedding = Column(Text, nullable=False)
    data_row = Column(Text, nullable=False)


# Load the pre-trained model and tokenizer for generating embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# Pydantic model for validating the query input
class Query(BaseModel):
    text: str


# Function to convert text to an embedding vector
def text_to_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    return embedding.numpy()


# API endpoint to find the most similar row based on the input text
@app.post("/find_related_row/")
async def find_related_row(query: Query):
    query_embedding = text_to_embedding(query.text)

    # Start a database session
    db_session = SessionLocal()

    # Query all embeddings from the database
    embedding_records = db_session.query(EmbeddingTable).all()
    db_session.close()

    best_match = None
    best_distance = float("inf")

    # Iterate through all records to find the closest embedding
    for record in embedding_records:
        db_embedding = np.array(record.embedding.split(",")).astype(float)
        distance = cosine(query_embedding, db_embedding)

        # Update the best match if a closer embedding is found
        if distance < best_distance:
            best_distance = distance
            best_match = {"related_row": record.data_row, "row_number": record.id}

    # Return the best matching row or raise a 404 error if no match is found
    if best_match:
        return {"best_match": best_match}
    else:
        raise HTTPException(status_code=404, detail="No matching row found.")
