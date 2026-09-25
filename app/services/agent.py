"""Alur pencarian yang memeriksa pertanyaan dan kutipan sebelum menjawab."""
import re
from app.schemas import Answer, Citation
from app.services.guardrails import normalize, refusal_reason
from app.services.llm import LLMUnavailable, InvalidSelection


def refuse(reason, mode, route="none", trace=None):
    messages = {
        "unsafe_request": "Permintaan ditolak. Sistem tidak mengikuti instruksi untuk mengabaikan kebijakan atau membuka konfigurasi rahasia.",
        "sensitive_input": "Jangan kirim kredensial melalui API ini. Hapus kredensial dari pertanyaan dan ikuti prosedur pelaporan keamanan.",
        "private_data_request": "API ini tidak memiliki akses ke tiket pribadi. Informasi tiket karyawan lain tidak dapat ditampilkan.",
        "action_not_supported": "API ini hanya membaca panduan; tidak dapat membuat, menutup, mengirim, atau mengeskalasi tiket.",
        "model_unavailable": "Model belum tersedia. Jawaban belum dapat dibuat; silakan coba kembali setelah layanan model aktif.",
        "invalid_model_output": "Keluaran model tidak lolos pemeriksaan bukti. Sistem tidak memberikan jawaban yang belum tervalidasi.",
    }
    return Answer(answer=messages.get(reason, "Tidak ditemukan dalam dokumen."),
                  confidence_label="low", reason_code=reason, mode=mode,
                  route=route, trace=trace or ["guard"])


def direct_faq_answer(question, passage):
    """True when the user asks the same question as a FAQ, possibly in shorter form."""
    match = re.match(r"\*\*T:\s*(.*?)\*\*\s*\nJ:\s*", passage.text, flags=re.S)
    if not match:
        return False
    asked = normalize(question).rstrip(" ?.!。？")
    faq_question = normalize(match.group(1)).rstrip(" ?.!。？")
    if len(asked) < 12:
        return False
    return re.search(rf"(?<!\w){re.escape(asked)}(?!\w)", faq_question) is not None


def answer_text(passage):
    """Keep FAQ wording concise while citations preserve the complete source text."""
    if passage.authority == "faq":
        match = re.match(r"\*\*T:.*?\*\*\s*\nJ:\s*", passage.text, flags=re.S)
        if match:
            return passage.text[match.end():].strip()
    return passage.text


class Agent:
    def __init__(self, retriever, selector, settings):
        self.retriever, self.selector, self.settings = retriever, selector, settings

    def ask(self, question):
        mode = self.settings.provider
        reason = refusal_reason(question)
        if reason:
            return refuse(reason, mode)
        q = normalize(question)
        route = "search_sop" if any(w in q for w in ("akses", "akun", "sementara", "hari yang sama", "hari ini")) else "search_active_policy"
        trace = ["guard_passed", route]
        hits = self.retriever.search(question, sop_only=route == "search_sop")
        if not hits:
            return refuse("no_relevant_context", mode, route, trace)
        evidence = self.retriever.evidence(hits, question)
        # Rujukan ke direktori belum tentu memuat nomor telepon yang diminta.
        asks_phone_value = re.search(r"(berapa|sebutkan|tuliskan).{0,30}nomor.{0,20}(telepon|telp)", q)
        if asks_phone_value and not any(
            re.search(r"(?:\+62|0)\d[\d\s()\-]{6,}\d", p.text) for p in evidence
        ):
            return refuse("no_relevant_context", mode, route, trace + ["required_detail_missing"])
        # FAQ yang cocok langsung tidak perlu melewati LLM atau membawa paragraf berulang.
        direct_faq = next((p for p in evidence if p.authority == "faq"
                           and direct_faq_answer(question, p)), None)
        if direct_faq:
            selected = [direct_faq]
        else:
            try:
                chosen = self.selector.select(question, evidence)
            except LLMUnavailable:
                return refuse("model_unavailable", mode, route, trace + ["model_failed"])
            except InvalidSelection:
                return refuse("invalid_model_output", mode, route, trace + ["validation_failed"])
            if chosen.abstain:
                return refuse("no_relevant_context", mode, route, trace + ["abstained"])
            by_id = {p.id: p for p in evidence}
            selected = [by_id[i] for i in chosen.selected_ids]
        # Untuk pengecualian, sertakan juga syarat dan persetujuannya.
        critical_sections = {
            p.section for p in selected
            if p.section.endswith(("Pengecualian Permintaan Hari yang Sama", "Langkah Service Desk",
                                    "Ketentuan Khusus Status Menunggu Pemohon"))
        }
        for p in evidence:
            if p.section in critical_sections and p not in selected:
                selected.append(p)
        if any(not p.active or p.authority not in {"primary", "faq"} for p in selected):
            return refuse("invalid_model_output", mode, route, trace + ["validation_failed"])
        citations = []
        for p in selected:
            m = p.metadata(self.retriever.header)
            citations.append(Citation(chunk_id=p.id, doc_id=m["doc_id"], doc_title=m["doc_title"],
                doc_version=p.version, effective_date=m["effective_date"], is_active=p.active,
                section_title=p.section, line_start=p.line_start, line_end=p.line_end, quote=p.text))
        # Jawaban disusun dari kutipan sumber, bukan teks bebas dari model.
        answer = "\n\n".join(f"{answer_text(p)} [{i}]" for i, p in enumerate(selected, 1))
        similarity_by_id = {p.id: similarity for p, _, similarity in hits}
        confidence = "high" if mode != "extractive" and all(
            similarity_by_id.get(p.id, 0) >= 0.65 for p in selected
        ) else "medium"
        return Answer(answer=answer, confidence_label=confidence,
            reason_code="extractive_baseline" if mode == "extractive" else "answered",
            citations=citations, route=route, mode=mode,
            trace=trace + ["select_evidence", "citations_verified"])
