import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

Vector = List[float]
Record = Dict[str, Any]


def _dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_similarity(a: Vector, b: Vector) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


def _euclidean_distance(a: Vector, b: Vector) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class VectorDatabase:
    """Simple vector database with SQLite persistence."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self._items: Dict[str, Record] = {}
        if self.path is not None:
            self._path = self.path
            self._connect()
            self._load()
        else:
            self._conn = None

    def _connect(self) -> None:
        assert self.path is not None
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, vector TEXT NOT NULL, metadata TEXT)"
        )
        self._conn.commit()

    def _load(self) -> None:
        assert self._conn is not None
        cursor = self._conn.execute("SELECT id, vector, metadata FROM vectors")
        self._items = {}
        for item_id, vector_text, metadata_text in cursor:
            vector = json.loads(vector_text)
            metadata = json.loads(metadata_text) if metadata_text else {}
            self._items[item_id] = {
                "id": item_id,
                "vector": vector,
                "metadata": metadata,
            }

    def add(self, item_id: str, vector: Vector, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add or update a vector record."""
        metadata = metadata or {}
        self._items[item_id] = {
            "id": item_id,
            "vector": vector,
            "metadata": metadata,
        }
        if self._conn is not None:
            self._conn.execute(
                "REPLACE INTO vectors (id, vector, metadata) VALUES (?, ?, ?)",
                (item_id, json.dumps(vector), json.dumps(metadata)),
            )
            self._conn.commit()

    def delete(self, item_id: str) -> None:
        """Remove a vector record by ID."""
        if item_id in self._items:
            del self._items[item_id]
        if self._conn is not None:
            self._conn.execute("DELETE FROM vectors WHERE id = ?", (item_id,))
            self._conn.commit()

    def get(self, item_id: str) -> Optional[Record]:
        return self._items.get(item_id)

    def search(
        self,
        query_vector: Vector,
        top_k: int = 5,
        metric: str = "cosine",
    ) -> List[Tuple[Record, float]]:
        """Search nearest neighbors for a query vector."""
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("metric must be 'cosine' or 'euclidean'")

        results: List[Tuple[Record, float]] = []
        for record in self._items.values():
            vector = record["vector"]
            if metric == "cosine":
                score = _cosine_similarity(query_vector, vector)
                results.append((record, score))
            else:
                score = _euclidean_distance(query_vector, vector)
                results.append((record, score))

        if metric == "cosine":
            results.sort(key=lambda item: item[1], reverse=True)
        else:
            results.sort(key=lambda item: item[1])
        return results[:top_k]

    def all(self) -> List[Record]:
        return list(self._items.values())

    def save(self, path: Optional[str] = None) -> None:
        """Persist the database to disk."""
        target_path = Path(path) if path else self.path
        if target_path is None:
            raise ValueError("No path provided for save().")
        target_conn = sqlite3.connect(target_path)
        target_conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, vector TEXT NOT NULL, metadata TEXT)"
        )
        for item in self._items.values():
            target_conn.execute(
                "REPLACE INTO vectors (id, vector, metadata) VALUES (?, ?, ?)",
                (item["id"], json.dumps(item["vector"]), json.dumps(item["metadata"])),
            )
        target_conn.commit()
        target_conn.close()
