from vector_db import VectorDatabase


def main() -> None:
    db = VectorDatabase("vectors.db")

    # Insert a handful of vectors with metadata.
    db.add("doc-1", [0.1, 0.2, 0.3], {"title": "Introduction"})
    db.add("doc-2", [0.9, 0.8, 0.7], {"title": "Advanced Concepts"})
    db.add("doc-3", [0.2, 0.1, 0.5], {"title": "Search Basics"})

    query = [0.2, 0.1, 0.3]
    results = db.search(query, top_k=3, metric="cosine")

    print("Top search results:")
    for record, score in results:
        print(f"- {record['id']} (score={score:.4f}, title={record['metadata'].get('title')})")

    # Persist to disk explicitly if needed.
    db.save()


if __name__ == "__main__":
    main()
