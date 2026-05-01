# vector-db

A minimal Python vector database example.

## Features

- Store vectors with IDs and metadata
- Search by cosine similarity or Euclidean distance
- Persist records to SQLite

## Files

- `vector_db/store.py` - core vector database implementation
- `example.py` - demonstration script to insert and search vectors
- `requirements.txt` - dependency hints

## Usage

1. Run the example:

```bash
python example.py
```

2. Use the API in your own code:

```python
from vector_db import VectorDatabase

db = VectorDatabase("vectors.db")
db.add("item-1", [0.1, 0.2, 0.3], {"label": "example"})
results = db.search([0.1, 0.2, 0.3], top_k=5, metric="cosine")
```

## Notes

- This implementation is intentionally simple and suitable for prototypes.
- For production-grade vector search, consider FAISS, Milvus, or Pinecone.
