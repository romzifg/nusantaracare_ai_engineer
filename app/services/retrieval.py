import hashlib
import json
import re
import time

import chromadb
from chromadb.config import Settings as ChromaSettings
from app.services.lexical import STOP, TfidfEncoder, tokens, word_count
from app.services.runtime_stats import log_stage

from app.config import ROOT
from app.services.knowledge import SOURCE_NAME, SOURCE_SHA256, PIPELINE_VERSION, parse_document, make_chunks


def terms(text):
    """Kecocokan kata literal sebagai pelengkap vektor yang sudah dinormalisasi."""
    return set(re.findall(r"\w+", text.lower())) - STOP


class Retriever:
    def __init__(self, settings):
        started = time.perf_counter()
        self.settings = settings
        self.header, self.passages = parse_document(ROOT / "data/raw_docs" / SOURCE_NAME)
        self.by_id = {p.id: p for p in self.passages}
        self.chunks = make_chunks(self.passages, self.header, word_count)
        for chunk in self.chunks:
            chunk["embedding_text"] = chunk["metadata"]["section_title"] + "\n" + chunk["text"]
        self.model = TfidfEncoder([c["embedding_text"] for c in self.chunks])
        log_stage("tfidf_ready", started)
        signature = hashlib.sha256(json.dumps(
            [SOURCE_SHA256, PIPELINE_VERSION, self.model.vocabulary, self.model.idf, 112, 18]
        ).encode()).hexdigest()[:16]
        self.client = chromadb.PersistentClient(
            path=settings.db_path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        log_stage("chroma_opened", started)
        self.collection = self.client.get_or_create_collection(
            "nusantaracare_" + signature, embedding_function=None,
            metadata={"source_sha256": SOURCE_SHA256, "pipeline": PIPELINE_VERSION},
            configuration={"hnsw": {"space": "cosine"}},
        )
        expected = {c["id"] for c in self.chunks}
        existing = set(self.collection.get(include=[])["ids"])
        if existing - expected:
            raise ValueError("Indeks berisi ID sumber berbeda. Pilih CHROMA_PATH baru untuk rebuild.")
        missing = [chunk for chunk in self.chunks if chunk["id"] not in existing]
        if missing:
            # Lanjutkan indeks yang terputus tanpa menghapus vektor yang sudah lengkap.
            # Batasi vektor sementara di RAM; tidak menyalin seluruh korpus sekaligus.
            for start in range(0, len(missing), 16):
                batch = missing[start:start + 16]
                self.collection.upsert(
                    ids=[c["id"] for c in batch], documents=[c["text"] for c in batch],
                    metadatas=[c["metadata"] for c in batch],
                    embeddings=[self.model.encode(c["embedding_text"]) for c in batch],
                )
        log_stage("index_ready", started)

    def search(self, question, sop_only=False):
        # Kecocokan kata membantu mengurutkan hasil; similarity tetap harus melewati ambang.
        filters = {"$and": [
            {"is_active": {"$eq": True}},
            {"doc_version": {"$eq": "2.0"}},
            {"authority": {"$in": ["primary"] if sop_only else ["primary", "faq"]}},
        ]}
        query_vector = self.model.encode(question)
        if not any(query_vector):
            return []
        query_vectors = [query_vector]
        result = self.collection.query(query_embeddings=query_vectors, n_results=self.settings.top_k,
                                       where=filters, include=["metadatas", "distances"])
        scored = {}
        qterms = terms(" ".join(tokens(question)))
        for metas, distances in zip(result["metadatas"], result["distances"]):
            for meta, distance in zip(metas, distances):
                similarity = 1-float(distance)
                if similarity < self.settings.min_similarity:
                    continue
                p = self.by_id[meta["passage_id"]]
                lexical = len(qterms & terms(p.text + " " + p.section)) / max(1, len(qterms))
                score = 0.85*similarity + 0.15*lexical
                if p.authority == "primary":
                    score += 0.02
                if p.id not in scored or score > scored[p.id][1]:
                    scored[p.id] = (p, score, similarity)
        hits = sorted(scored.values(), key=lambda item: item[1], reverse=True)[:6]
        return hits

    def evidence(self, hits, question):
        """Sertakan paragraf prasyarat agar model mendapat konteks yang cukup."""
        passages = [h[0] for h in hits]
        q = question.lower()
        sections = {p.section for p in passages[:2]}
        for p in self.passages:
            if not p.active or p.authority != "primary":
                continue
            needs_approval = ("akses" in q or "akun" in q) and p.section.endswith("Langkah Service Desk")
            if p.section in sections or needs_approval:
                if p not in passages:
                    passages.append(p)
        # Jangan potong paragraf di tengah kalimat saat membatasi panjang konteks.
        budget, output = 20000, []
        for p in passages:
            if len(p.text) <= budget:
                output.append(p)
                budget -= len(p.text)
        return output
