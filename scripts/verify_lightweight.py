"""Uji startup, RAM, API, dan retrieval tanpa memanggil penyedia LLM.

Jalankan: python -m scripts.verify_lightweight --output reports/lightweight.json
"""
import argparse
import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from app.services.runtime_stats import memory_mib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    initial_memory = memory_mib()
    # Jangan memakai key atau provider .env pengguna selama verifikasi offline.
    os.environ.update(APP_ENV="local", LLM_PROVIDER="extractive", API_ACCESS_TOKEN="")
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.config import ROOT, Settings
    (ROOT / ".cache").mkdir(exist_ok=True)
    directory = tempfile.mkdtemp(prefix="verify-tfidf-", dir=ROOT / ".cache")
    settings = Settings(provider="extractive", db_path=directory)
    application = create_app(settings=settings)
    with TestClient(application) as client:
        startup_seconds = round(time.perf_counter() - started, 3)
        startup_memory = memory_mib()
        retriever = application.state.agent.retriever
        cases = json.loads((ROOT / "tests/eval_cases.json").read_text(encoding="utf-8"))
        rows = []
        for case in cases:
            q = case["question"].lower()
            sop_only = any(w in q for w in ("akses", "akun", "sementara", "hari yang sama", "hari ini"))
            hits = retriever.search(case["question"], sop_only=sop_only)
            rows.append({
                "id": case["id"], "question": case["question"], "expected": case["expected"],
                "expected_section": case["section"],
                "found_expected_section": (any(case["section"].casefold() in p.section.casefold()
                                               for p, _, _ in hits) if case["expected"] == "answer" else None),
                "hits": [{"id": p.id, "section": p.section, "score": round(sim, 4)} for p, _, sim in hits],
            })
        responses = {
            "health": client.get("/health"), "ready": client.get("/ready"),
            "faq": client.post("/ask", json={"question": "Apakah saya bisa mengirim permintaan melalui WhatsApp?"}),
            "no_context": client.post("/ask", json={"question": "Bagaimana budidaya anggrek?"}),
            "phone_missing": client.post("/ask", json={"question": "Berapa nomor telepon Service Desk?"}),
        }
        api = {key: {"status": value.status_code, "body": value.json()} for key, value in responses.items()}
        answers = [row for row in rows if row["expected"] == "answer"]
        heavy = [name for name in ("torch", "transformers", "sentence_transformers", "fastembed", "onnxruntime")
                 if name in sys.modules]
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "python": platform.python_version(),
            "pipeline": "local-tfidf-chroma", "llm_called": False,
            "startup_seconds_including_imports": startup_seconds,
            "initial_memory": initial_memory, "startup_memory": startup_memory,
            "after_queries_memory": memory_mib(), "heavy_modules_loaded": heavy,
            "chunks": len(retriever.chunks), "dimensions": len(retriever.model.vocabulary),
            "threshold": settings.min_similarity,
            "retrieval_passed": sum(row["found_expected_section"] for row in answers),
            "retrieval_total": len(answers),
            "refusal_questions_with_candidates": sum(bool(row["hits"]) for row in rows if row["expected"] == "refuse"),
            "warning": "Historical questions, not a new holdout. Retrieval only, not LLM answer accuracy. Windows RAM is not Linux container RAM.",
            "api": api, "rows": rows,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key not in {"api", "rows"}}, indent=2))
        print("Missing sections:", [row["id"] for row in answers if not row["found_expected_section"]])
        assert not heavy, "Model runtime neural tidak boleh dimuat."
        assert all(result["status"] == 200 for result in api.values()), "Smoke test HTTP gagal."
        assert api["no_context"]["body"]["reason_code"] == "no_relevant_context"
        assert api["phone_missing"]["body"]["reason_code"] == "no_relevant_context"
        assert api["faq"]["body"]["citations"]
        assert report["retrieval_passed"] == report["retrieval_total"], "Ada regresi pencarian; periksa laporan."


if __name__ == "__main__":
    main()
