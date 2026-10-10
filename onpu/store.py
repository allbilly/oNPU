from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

DEFAULTS = {"context": 1024, "max_tokens": 128, "temperature": 0.6,
            "profile": "CPU", "realize": True, "idle_timeout": 600, "memory_fraction": 0.8}


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        (root / "models").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(root / "onpu.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, data TEXT, updated REAL, PRIMARY KEY(kind,id))")
        self.db.commit()

    def put(self, kind: str, item: dict) -> dict:
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?,?)",
                            (kind, item["id"], json.dumps(item), time.time()))
            if kind == "metrics":
                self.db.execute("DELETE FROM records WHERE kind='metrics' AND id NOT IN "
                                "(SELECT id FROM records WHERE kind='metrics' ORDER BY updated DESC LIMIT 500)")
        return item

    def get(self, kind: str, key: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT data FROM records WHERE kind=? AND id=?", (kind, key)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, kind: str, limit: int = 500) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT data FROM records WHERE kind=? ORDER BY updated DESC LIMIT ?", (kind, limit)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete(self, kind: str, key: str):
        with self.lock, self.db:
            self.db.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, key))

    def settings(self) -> dict:
        return {**DEFAULTS, **(self.get("settings", "global") or {})}

    def close(self):
        self.db.close()
