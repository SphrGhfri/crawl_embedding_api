import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Function to read Markdown file
def read_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to parse Markdown table into a DataFrame, excluding the first column
def markdown_to_dataframe(markdown_text):
    # Extract the table part of the Markdown
    table_start = markdown_text.find('|')
    table_end = markdown_text.rfind('|', table_start) + 1
    table_markdown = markdown_text[table_start:table_end]

    # Convert Markdown table to DataFrame
    table_markdown = table_markdown.strip().split('\n')
    table_markdown = [row.strip() for row in table_markdown if row.strip()]
    
    # Exclude the first column by processing the rows
    headers = table_markdown[0].split('|')[1:]  # Skip the first header
    data = [row.split('|')[1:] for row in table_markdown[2:]]  # Skip the first data column
    
    df = pd.DataFrame(data, columns=[header.strip() for header in headers])
    return df

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Function to convert text to embeddings
def text_to_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()

# Process table rows and convert to embeddings
def process_table_to_embeddings(table_df):
    embeddings = {}
    for index, row in table_df.iterrows():
        text = ' | '.join(row.astype(str))  # Combine all columns into one text
        embeddings[index] = text_to_embedding(text)
    return embeddings

# Save embeddings to CSV, including the original data rows
def save_embeddings_to_csv(embeddings, table_df, file_path):
    embedding_list = []
    for key, embedding in embeddings.items():
        row_data = ' | '.join(table_df.loc[key].astype(str))  # Retrieve the correct row using `loc`
        embedding_list.append([key] + embedding.tolist() + [row_data])
    df = pd.DataFrame(embedding_list, columns=['Row'] + [f'feature_{i}' for i in range(len(embedding))] + ['DataRow'])
    df.to_csv(file_path, index=False)

# Example usage
markdown_text = read_markdown_file('main_table_output.md')
table_df = markdown_to_dataframe(markdown_text)
embeddings = process_table_to_embeddings(table_df)
save_embeddings_to_csv(embeddings, table_df, 'embeddings.csv')
