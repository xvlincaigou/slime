from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


@dataclass
class RetrievedDoc:
    docid: str
    text: str
    url: str
    score: float


class BrowseCompPlusBM25Retriever:
    def __init__(self, corpus_root: str | os.PathLike[str]):
        self.corpus_root = Path(corpus_root)
        self.docs: list[dict[str, str]] = []
        self.doc_tokens: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self.docid_to_doc: dict[str, dict[str, str]] = {}

    def load(self) -> None:
        jsonl_path = self.corpus_root / "corpus.jsonl"
        tsv_path = self.corpus_root / "corpus.tsv"

        if jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    doc = {
                        "docid": str(row["docid"]),
                        "text": row.get("text", ""),
                        "url": row.get("url", ""),
                    }
                    self.docs.append(doc)
        elif tsv_path.exists():
            with tsv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    doc = {
                        "docid": str(row["docid"]),
                        "text": row.get("text", ""),
                        "url": row.get("url", ""),
                    }
                    self.docs.append(doc)
        else:
            raise FileNotFoundError(
                f"Neither corpus.jsonl nor corpus.tsv exists under {self.corpus_root}."
            )

        self.doc_tokens = [tokenize(doc["text"]) for doc in self.docs]
        self.bm25 = BM25Okapi(self.doc_tokens)
        self.docid_to_doc = {doc["docid"]: doc for doc in self.docs}

    def search(self, query: str, topk: int = 5) -> list[RetrievedDoc]:
        if self.bm25 is None:
            self.load()

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        assert self.bm25 is not None
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:topk]

        results = []
        for idx in top_indices:
            doc = self.docs[idx]
            results.append(
                RetrievedDoc(
                    docid=doc["docid"],
                    text=doc["text"],
                    url=doc["url"],
                    score=float(scores[idx]),
                )
            )
        return results


_RETRIEVER: BrowseCompPlusBM25Retriever | None = None
_RETRIEVER_LOCK = Lock()


def get_retriever(corpus_root: str | os.PathLike[str]) -> BrowseCompPlusBM25Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        with _RETRIEVER_LOCK:
            if _RETRIEVER is None:
                retriever = BrowseCompPlusBM25Retriever(corpus_root=corpus_root)
                retriever.load()
                _RETRIEVER = retriever
    assert _RETRIEVER is not None
    return _RETRIEVER
