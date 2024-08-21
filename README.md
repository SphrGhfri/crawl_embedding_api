# Qavanin.ir Data Crawler and Embedding API

This repository is an interview task designed to crawl the main table from [qavanin.ir](https://qavanin.ir), convert the data into Markdown format, embed the table's data, and export the embedded data to a PostgreSQL database. The repository also includes a FastAPI API that accepts text input and returns the most similar row from the embedded data.

## Features

- **Crawl Data**: Scrapes the first 5 pages of the main table from [qavanin.ir](https://qavanin.ir) and converts the data into Markdown format.
- **Embed Data**: Converts the table data into embeddings using a pre-trained transformer model.
- **Database Integration**: Exports the embedded data to a PostgreSQL database.
- **API**: Provides a FastAPI endpoint that accepts text input and returns the most similar row from the database based on the embeddings.

## Getting Started

### Prerequisites

- Python 3.8 or later
- PostgreSQL database (for storing embeddings)
- Docker (optional, for containerized deployment)

### Installation
You can choose between Local or Docker Installation.
#### Local Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/SphrGhfri/crawl_embedding_api.git
   cd crawl_embedding_api
2. **Install the required Python packages**:
   ```bash
   pip install -r requirements.txt
3. **Set up your environment**:
- In .env file choose for local or docker (Remember add .env to .gitignore):
   ```bash
   DATABASE_URL=postgresql://user:password123@localhost:5432/dbname
4. **Run the Crawler**:
   ```bash
   python crawler.py
5. **Store Embeddings**:
   ```bash
   python store_embedding.py
6. **Run the FastAPI application**:
   ```bash
   uvicorn main:app --reload
#### Docker Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/SphrGhfri/crawl_embedding_api.git
   cd crawl_embedding_api
2. **Install the required Python packages**:
   ```bash
   pip install -r requirements.txt
3. **Set up your environment**:
- In .env file choose for local or docker (Remember add .env to .gitignore):
   ```bash
   DATABASE_URL=postgresql://user:password123@db:5432/dbname
4. **Run the Crawler**:
   ```bash
   python crawler.py
5. **Run Docker Compose**:
   ```bash
   docker compose up -d
### Usage
- Once the application is running, you can access the API at http://localhost:8000/docs/find_related_row/.
### Recreate the Embedding Database:
To recreate the embeddings database and start fresh:
   ```bash
   docker compose up -d --force-recreate
   ```

### Notes
- Uncomment `DATABASE_URL`: If you switch between local and Docker environments, uncomment or modify the `DATABASE_URL` in your `.env` file accordingly.

- `.env` File: Ensure that your `.env` file is excluded from version control by adding it to `.gitignore`.

### Contributing
Feel free to fork this repository, make changes, and submit pull requests. All contributions are welcome!