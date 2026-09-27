import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import ROOT, Settings
from app.services.rag import build_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    settings = Settings.from_env()
    agent = build_agent(settings)
    cases = json.loads((ROOT / "tests/eval_cases.json").read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        if case["split"] != args.split:
            continue
        started = time.perf_counter()
        result = agent.ask(case["question"])
        grounded = bool(result.citations) and all(
            c.is_active and c.doc_version == "2.0"
            and c.quote == agent.retriever.by_id[c.chunk_id].text for c in result.citations
        )
        keyword_pass = all(word in result.answer.casefold() for word in case["must_include"])
        if case["expected"] == "answer":
            passed = grounded and keyword_pass and result.reason_code in {"answered", "extractive_baseline"}
        else:
            passed = not result.citations and result.reason_code in {
                "unsafe_request", "sensitive_input", "private_data_request", "action_not_supported",
                "out_of_scope", "no_relevant_context",
            }
        row = {**case, "passed": passed, "grounded": grounded,
               "seconds": round(time.perf_counter()-started, 3), "response": result.model_dump()}
        rows.append(row)
        print(case["id"], "PASS" if passed else "FAIL", result.reason_code, flush=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": settings.provider, "model": settings.model, "split": args.split,
        "threshold": settings.min_similarity, "passed": sum(r["passed"] for r in rows),
        "total": len(rows),
        "warning": "Pemeriksaan kata kunci hanya smoke test; periksa kutipan dan relevansi secara manual.",
        "rows": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Summary:", report["passed"], "/", report["total"], flush=True)
    if not rows or report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
