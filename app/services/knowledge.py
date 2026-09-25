"""Baca dokumen sumber dan tandai bagian aktif serta arsip."""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

SOURCE_NAME = "nusantaracare_panduan_operasional_internal_v2.md"
SOURCE_SHA256 = "4c8aa6e896425547ec55392a7a8df97142d20e3d93b480aa29494f0e6f514db8"
PIPELINE_VERSION = "nc-ops-v2-title-aware-faq-answer"


@dataclass
class Passage:
    id: str
    text: str
    section: str
    section_id: str
    line_start: int
    line_end: int
    version: str
    active: bool
    authority: str

    def metadata(self, header):
        return {
            "doc_id": header["doc_id"],
            "doc_title": header["doc_title"],
            "doc_version": self.version,
            "effective_date": "2025-01-01" if not self.active else str(header["effective_date"]),
            "effective_until": "2026-06-30" if not self.active else "",
            "is_active": self.active,
            "section_title": self.section,
            "section_id": self.section_id,
            "passage_id": self.id,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "authority": self.authority,
        }


def parse_document(path: Path, verify=True):
    raw = path.read_bytes()
    if verify and hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Dokumen sumber berubah. Verifikasi dokumen asli, jangan ubah hash sembarang.")
    lines = raw.decode("utf-8-sig").splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("Frontmatter wajib tersedia.")
    end = lines.index("---", 1)
    header = yaml.safe_load("\n".join(lines[1:end]))
    if header["doc_id"] != "NC-OPS-001" or str(header["doc_version"]) != "2.0":
        raise ValueError("Identitas sumber tidak sesuai.")
    passages = []
    h2, h3 = "", ""
    start, buffer = None, []

    def flush():
        nonlocal start, buffer
        if buffer:
            text = "\n".join(buffer)
            active = "NONAKTIF" not in h3.upper()
            section = " / ".join(x for x in (h2, h3) if x)
            authority = (
                "archive" if not active else
                "summary" if h2 in {"Lampiran Matriks Keputusan", "Riwayat Perubahan dan Arsip Kebijakan"} else
                "example" if "Contoh" in h3 else
                "faq" if h2 == "FAQ Operasional" else "primary"
            )
            sid = hashlib.sha256(section.encode()).hexdigest()[:12]
            pid = f"NC-{start:04d}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"
            passages.append(Passage(pid, text, section, sid, start, start+len(buffer)-1,
                                    "2.0" if active else "1.4", active, authority))
        buffer, start = [], None

    for number, line in enumerate(lines[end+1:], start=end+2):
        if line.startswith("#"):
            flush()
            if line.startswith("## "):
                h2, h3 = line[3:].strip(), ""
            elif line.startswith("### "):
                h3 = line[4:].strip()
        elif not line.strip():
            flush()
        else:
            if start is None:
                start = number
            buffer.append(line)
    flush()
    return header, passages


def make_chunks(passages, header, tokenizer, max_tokens=96, overlap=18):
    """Potong teks berdasarkan token sambil menjaga kutipan tetap sama dengan sumber."""
    if overlap >= max_tokens or overlap < 0:
        raise ValueError("Overlap harus lebih kecil dari ukuran chunk.")
    chunks = []
    for p in passages:
        offsets = tokenizer(p.text, add_special_tokens=False, truncation=False,
                            max_length=100000, return_offsets_mapping=True)["offset_mapping"]
        for start in range(0, len(offsets), max_tokens-overlap):
            stop = min(start+max_tokens, len(offsets))
            text = p.text[offsets[start][0]:offsets[stop-1][1]]
            chunks.append({
                "id": f"{p.id}-c{start:04d}", "text": text,
                "metadata": p.metadata(header),
            })
            if stop == len(offsets):
                break
    return chunks
