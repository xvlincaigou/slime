from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def stable_sort_key(query_id: str) -> str:
    return hashlib.md5(query_id.encode("utf-8")).hexdigest()


def build_prompt(query: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": query}]


def build_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Performs a search on a knowledge source. Returns the top-5 results with docid, score, and snippet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def normalize_record(row: dict) -> dict:
    gold_docs = row.get("gold_docs") or row.get("evidence_docs") or []
    reward_model = {
        "answer": row["answer"],
        "query_id": str(row["query_id"]),
        "gold_docids": [str(doc["docid"]) for doc in gold_docs],
        "gold_urls": [doc.get("url", "") for doc in gold_docs],
    }
    metadata = {
        "query_id": str(row["query_id"]),
        "answer": row["answer"],
        "gold_docids": reward_model["gold_docids"],
        "gold_urls": reward_model["gold_urls"],
    }
    return {
        "prompt": build_prompt(row["query"]),
        "reward_model": json.dumps(reward_model, ensure_ascii=False),
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "tools": json.dumps(build_tools(), ensure_ascii=False),
    }


def load_rows(dataset_root: Path) -> list[dict]:
    parquet_paths = sorted(dataset_root.glob("train-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No train-*.parquet shards found under {dataset_root}.")

    frames = [pd.read_parquet(path) for path in parquet_paths]
    df = pd.concat(frames, ignore_index=True)
    records = df.to_dict(orient="records")
    records.sort(key=lambda row: stable_sort_key(str(row["query_id"])))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="Path containing BrowseComp-Plus train parquet shards.")
    parser.add_argument("--output-dir", required=True, help="Directory to write train/eval parquet files.")
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(dataset_root)
    eval_count = max(1, int(len(rows) * args.eval_ratio))
    train_rows = rows[:-eval_count]
    eval_rows = rows[-eval_count:]

    train_df = pd.DataFrame([normalize_record(row) for row in train_rows])
    eval_df = pd.DataFrame([normalize_record(row) for row in eval_rows])

    train_path = output_dir / "train.parquet"
    eval_path = output_dir / "eval.parquet"
    train_df.to_parquet(train_path, index=False)
    eval_df.to_parquet(eval_path, index=False)

    print(
        json.dumps(
            {
                "train_path": str(train_path),
                "eval_path": str(eval_path),
                "num_train": len(train_df),
                "num_eval": len(eval_df),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
