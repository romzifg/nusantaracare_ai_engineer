import hashlib
import json
import math
import re

import chromadb
from chromadb.config import Settings as ChromaSettings
from fastembed import TextEmbedding

from app.config import ROOT
from app.services.knowledge import SOURCE_NAME, SOURCE_SHA256, PIPELINE_VERSION, parse_document, make_chunks

STOP = set("apa apakah bagaimana berapa saya yang dan atau di ke dari untuk dengan dalam pada itu ini bisa boleh kapan siapa saja harus jika kalau ada tidak sebuah tolong jelaskan".split())


def terms(text):
    return set(re.findall(r"\w+", text.lower())) - STOP


class Retriever:
    def __init__(self, settings):
        self.settings = settings
        self.header, self.passages = parse_document(ROOT / "data/raw_docs" / SOURCE_NAME)
        self.by_id = {p.id: p for p in self.passages}
        self.model = TextEmbedding(
            model_name=settings.embedding_model,
            cache_dir=settings.embedding_cache,
            threads=1,
            providers=["CPUExecutionProvider"],
            enable_cpu_mem_arena=False,
        )
        self.chunks = make_chunks(self.passages, self.header, self.model.token_count)
        for chunk in self.chunks:
            chunk["embedding_text"] = chunk["metadata"]["section_title"] + "\n" + chunk["text"]
        signature = hashlib.sha256(json.dumps(
            [SOURCE_SHA256, PIPELINE_VERSION, settings.embedding_model, 112, 18]
        ).encode()).hexdigest()[:16]
        self.client = chromadb.PersistentClient(
            path=settings.db_path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            "nusantaracare_" + signature, embedding_function=None,
            metadata={"source_sha256": SOURCE_SHA256, "pipeline": PIPELINE_VERSION},
            configuration={"hnsw": {"space": "cosine"}},
        )
        expected = {c["id"] for c in self.chunks}
        if set(self.collection.get(include=[])["ids"]) != expected:
            # Jangan gunakan indeks parsial; buat CHROMA_PATH baru untuk membangun ulang.
            if self.collection.count():
                raise ValueError("Indeks parsial/berbeda. Pilih CHROMA_PATH baru untuk rebuild.")
            vectors = self._normalize(self.model.embed(
                [c["embedding_text"] for c in self.chunks], batch_size=1, parallel=None
            ))
            self.collection.upsert(
                ids=[c["id"] for c in self.chunks], documents=[c["text"] for c in self.chunks],
                metadatas=[c["metadata"] for c in self.chunks], embeddings=vectors
            )

    @staticmethod
    def _normalize(vectors):
        normalized = []
        for vector in vectors:
            values = [float(value) for value in vector]
            norm = math.sqrt(sum(value * value for value in values))
            normalized.append([value / max(norm, 1e-12) for value in values])
        return normalized

    def search(self, question, sop_only=False):
        # Kecocokan kata membantu mengurutkan hasil; similarity tetap harus melewati ambang.
        aliases = {"wa": "WhatsApp", "nyangkut": "gangguan", "login": "akses akun",
                   "lelet": "gangguan layanan", "laptop": "laptop perlengkapan"}
        expanded = " ".join(aliases.get(word.lower(), word) for word in question.split())
        if "kata sandi" in question.lower() or "password" in question.lower():
            expanded += " password kredensial autentikasi"
        queries = [expanded]
        if "akses" in question.lower() or "akun" in question.lower():
            queries.append(expanded + " persetujuan Atasan Langsung Pemilik Layanan")
        filters = {"$and": [
            {"is_active": {"$eq": True}},
            {"doc_version": {"$eq": "2.0"}},
            {"authority": {"$in": ["primary"] if sop_only else ["primary", "faq"]}},
        ]}
        query_vectors = self._normalize(self.model.query_embed(queries, batch_size=1, parallel=None))
        result = self.collection.query(query_embeddings=query_vectors, n_results=self.settings.top_k,
                                       where=filters, include=["metadatas", "distances"])
        scored = {}
        qterms = terms(expanded)
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
