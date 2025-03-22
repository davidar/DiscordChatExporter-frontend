# Vector Search for Discord Chat Exporter

This module adds semantic search capabilities to Discord Chat Exporter using HuggingFace embeddings and Qdrant vector database.

## Prerequisites

Before using this module, you need to install and run:

1. **Qdrant** - Vector database for storing message embeddings
   - Download from: [Qdrant Website](https://qdrant.tech/documentation/quick-start/)
   - Default port: 6333

2. **HuggingFace Dependencies** - For generating high-quality text embeddings
   - Python package: `sentence-transformers`
   - Model used: `BAAI/bge-small-en-v1.5` (will be downloaded automatically on first use)

## Setup

1. Make sure MongoDB is running and contains your Discord chat data
2. Install the required dependencies from `requirements.txt`
3. Run the setup check script:
   ```
   python setup_vector_search.py
   ```
4. Build the vector index for your Discord messages:
   ```
   # Index all guilds
   python -m src.vector_search.build_index
   
   # Or index a specific guild
   python -m src.vector_search.build_index --guild_id YOUR_GUILD_ID
   
   # Limit the number of messages to index
   python -m src.vector_search.build_index --max_messages 10000
   ```
5. Start the FastAPI server:
   ```
   python dev.py  # For development
   # or
   python prod.py  # For production
   ```

## API Endpoints

### Semantic Search

```
GET /api/guild/semantic_search?guild_id=YOUR_GUILD_ID&query=YOUR_QUERY&limit=10&fetch_full_messages=true
```

Parameters:
- `guild_id` (required): ID of the guild to search in
- `query` (required): Semantic search query
- `limit` (optional): Maximum number of results to return (default: 10)
- `fetch_full_messages` (optional): Whether to include full message objects in results (default: true)

### Build Index

```
POST /api/guild/build_index?guild_id=YOUR_GUILD_ID&max_messages=10000&force_rebuild=false
```

Parameters:
- `guild_id` (required): ID of the guild to index
- `max_messages` (optional): Maximum number of messages to index
- `force_rebuild` (optional): Whether to force rebuild the index (default: false)

## How It Works

1. Discord messages are fetched from MongoDB
2. Messages are embedded using the `BAAI/bge-small-en-v1.5` model via SentenceTransformer
3. Embeddings are stored in Qdrant vector database with metadata and content hashes
4. Search queries are embedded using the same model and compared to stored embeddings
5. Results are ranked by similarity and returned with configurable filters
