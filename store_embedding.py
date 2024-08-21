import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve the DATABASE_URL from environment variables
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

# Drop all existing tables and recreate them
Base.metadata.drop_all(bind=engine)  # Drop all existing tables
Base.metadata.create_all(bind=engine)  # Recreate the tables

# Function to read a Markdown file
def read_markdown_file(file_path: str) -> str:
    """Reads the content of a markdown file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Function to parse a Markdown table into a DataFrame
def markdown_to_dataframe(markdown_text: str) -> pd.DataFrame:
    """Converts markdown table text into a pandas DataFrame."""
    # Extract the table section from the markdown text
    table_start = markdown_text.find("|")
    table_end = markdown_text.rfind("|", table_start) + 1
    table_markdown = markdown_text[table_start:table_end].strip().split("\n")

    # Clean and process the table text
    headers = table_markdown[0].split("|")[1:]  # Skip the first header
    data = [row.split("|")[1:] for row in table_markdown[2:]]  # Skip the first data column

    # Create a DataFrame from the processed data
    df = pd.DataFrame(data, columns=[header.strip() for header in headers])
    return df

# Load pre-trained model and tokenizer for generating embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to convert text to embeddings
def text_to_embedding(text: str) -> np.ndarray:
    """Converts a text string into a numerical embedding."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()

# Function to process DataFrame rows and convert them to embeddings
def process_table_to_embeddings(table_df: pd.DataFrame) -> list:
    """Converts DataFrame rows to embeddings and pairs them with original text."""
    embeddings = []
    for _, row in table_df.iterrows():
        text = " ".join(row.astype(str))  # Combine all columns into one text
        embedding = text_to_embedding(text)
        embeddings.append((embedding, text))
    return embeddings

# Function to save embeddings to the database
def save_embeddings_to_db(embeddings: list):
    """Saves a list of embeddings and corresponding text rows to the database."""
    db_session = SessionLocal()
    for embedding, data_row in embeddings:
        embedding_str = ",".join(map(str, embedding))
        db_record = EmbeddingTable(embedding=embedding_str, data_row=data_row)
        db_session.add(db_record)
    db_session.commit()
    db_session.close()

# Main execution flow
if __name__ == "__main__":
    markdown_text = read_markdown_file("crawled_table.md")
    table_df = markdown_to_dataframe(markdown_text)
    embeddings = process_table_to_embeddings(table_df)
    save_embeddings_to_db(embeddings)
