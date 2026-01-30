# run_agent_hybrid.py
import json
import argparse
from typing import Any, Dict

from agent.graph_hybrid import build_hybrid_graph


def run_batch(input_path: str, out_path: str):
    graph = build_hybrid_graph()

    outputs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = item["id"]
            question = item["question"]
            format_hint = item.get("format_hint", "")

            # initial state
            state = {
                "id": qid,
                "question": question,
                "format_hint": format_hint,
            }

            final_state = graph.invoke(state)

            outputs.append(
                {
                    "id": qid,
                    "final_answer": final_state.get("final_answer"),
                    "sql": final_state.get("sql_query", ""),
                    "confidence": final_state.get("confidence", 0.0),
                    "explanation": final_state.get("explanation", ""),
                    "citations": final_state.get("citations", []),
                }
            )

    with open(out_path, "w", encoding="utf-8") as f:
        for obj in outputs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, help="Input JSONL with questions")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    args = parser.parse_args()

    run_batch(args.batch, args.out)


if __name__ == "__main__":
    main()
