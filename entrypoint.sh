#!/bin/bash
set -e

echo "Step 1: Running store_embedding.py to process and store the embeddings..."
python store_embedding.py
echo "store_embedding.py finished."

echo "Step 2: Starting the FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8000
